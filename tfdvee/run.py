import html
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .frontend import caption_model, visual_model, load_features
from . import calibration
from .temporal import confidence, contrast, event_statistics, multiscale, segment


def concept_f1(first, second):
    first, second = set(first), set(second)
    return 2 * len(first & second) / max(1, len(first) + len(second))


def extract_concepts(text, concepts):
    return sorted(concept for concept in concepts if re.search(
        r"(?<!\w)" + re.escape(concept.casefold()) + r"(?!\w)", text.casefold()))


def run(args):
    if args.confidence_calibration and args.variant in ("no_calibration", "frame_retrieval"):
        raise ValueError("Strength-only ablations do not consume a confidence calibration artifact")
    if args.temperature <= 0 or args.pre_context <= 0 or min(args.windows) < 0:
        raise ValueError("Invalid temperature or temporal context")
    if not 0 <= args.evidence_alpha <= 1 or not 0 <= args.confidence_threshold <= 1:
        raise ValueError("Evidence mixing and confidence thresholds must be in [0,1]")
    vocabulary = json.loads(Path(args.vocabulary).read_text(encoding="utf-8"))
    states = vocabulary["deviations"]
    if not states or len({state["id"] for state in states}) != len(states):
        raise ValueError("A nonempty vocabulary with unique deviation IDs is required")
    video = load_features(args.features)
    normal = load_features(args.normal)
    if video["clip_model"] != normal["clip_model"]:
        raise ValueError("Video and commissioning embeddings must use the same visual encoder")
    if Path(args.features).resolve() == Path(args.normal).resolve():
        raise ValueError("Commissioning and event videos must be separate inputs")
    registered = calibration.context(args, video["clip_model"], vocabulary)
    confidence_fit = (calibration.load(args.confidence_calibration, registered, args.video_id, args.device_group)
                      if args.confidence_calibration else None)
    model = visual_model(video["clip_model"], args.device)
    definitions = [vocabulary["normal"]] + states
    prompts = [f"{vocabulary['asset']}, {state['state']}; visible cue: {state['cue']}" for state in definitions]
    if args.variant == "generic_prompts":
        prompts = [f"{state['state']}" for state in definitions]
    prompt_features = model.text(prompts)
    if args.variant in ("frame_retrieval", "no_normal_contrast"):
        signal = video["features"] @ prompt_features[1:].T / args.temperature
        normal_signal = normal["features"] @ prompt_features[1:].T / args.temperature
    else:
        signal = contrast(video["features"], prompt_features, normal["features"], args.temperature, args.normal_coefficient)
        normal_signal = contrast(normal["features"], prompt_features, normal["features"], args.temperature, args.normal_coefficient)
    windows = [0] if args.variant in ("frame_retrieval", "no_temporal") else args.windows
    filtered, scales = multiscale(signal, normal_signal, windows, video["fps"])
    if args.variant == "frame_retrieval":
        candidates = segment(filtered, args.high, args.high - 1e-6, 0, 0, video["fps"])
    else:
        candidates = segment(filtered, args.high, args.low, args.persistence, args.merge_gap, video["fps"])
    captioner = None
    all_concepts = {concept for state in definitions for concept in state["concepts"]}
    records = []
    for candidate in candidates:
        state = states[candidate["state"]]
        statistics = event_statistics(candidate, filtered, scales, max(1, int(args.pre_context * video["fps"])))
        safety = bool(state.get("safety", False))
        if args.variant not in ("no_transition", "frame_retrieval") and statistics["transition"] <= args.transition_threshold and not safety and not statistics["no_pre_context"]:
            continue
        indices = statistics["evidence_indices"]
        images = []
        for index in indices:
            with Image.open(video["root"] / video["paths"][index]) as opened:
                images.append(opened.convert("RGB"))
        if captioner is None:
            captioner = caption_model(args.caption_model, args.device)
        caption = captioner.describe(images, vocabulary["asset"], state["state"])
        concepts = extract_concepts(caption, all_concepts)
        if args.variant in ("no_evidence", "frame_retrieval"):
            evidence_score = 0.0
        else:
            caption_feature = model.text([caption])[0]
            minimum = float((video["features"][indices] @ caption_feature).min())
            evidence_score = args.evidence_alpha * minimum + (1 - args.evidence_alpha) * concept_f1(concepts, state["concepts"])
        if args.variant in ("no_calibration", "frame_retrieval"):
            score = confidence(statistics["strength"], 0, 0, 0, [1, 0, 0, 0])
        else:
            score = confidence(statistics["strength"], evidence_score, statistics["disagreement"],
                               statistics["ambiguity"], args.confidence_coefficients)
            if confidence_fit:
                score = calibration.predict(dict(statistics, evidence_consistency=evidence_score), confidence_fit)
        weak_evidence = evidence_score < args.evidence_threshold and args.variant not in ("no_evidence", "frame_retrieval")
        review = score < args.confidence_threshold or weak_evidence or safety or statistics["no_pre_context"]
        if args.variant == "no_referral":
            review = False
        start_index, end_index = candidate["start"], candidate["end"]
        start = float(video["timestamps"][start_index])
        end = min(video["duration"], float(video["timestamps"][end_index - 1] + 1 / video["fps"]))
        event_id = f"{args.video_id}:{args.variant}:{state['id']}:{start:.3f}-{end:.3f}"
        records.append(dict(id=event_id, state=state["id"], start=start, end=end, confidence=score,
                            review=bool(review), safety=safety, caption=caption, concepts=concepts,
                            evidence_consistency=evidence_score,
                            evidence_frames=[str(video["root"] / video["paths"][i]) for i in indices],
                            **statistics))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(video_id=args.video_id, device_group=args.device_group, duration=video["duration"],
                   variant=args.variant, condition=args.condition, events=records, config=vars(args),
                   calibration_context=registered,
                   confidence_calibration=confidence_fit)
    (output / "events.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "signals.npz", timestamps=video["timestamps"], contrast=signal,
                        filtered=filtered, scales=scales, states=np.array([state["id"] for state in states]))
    blocks = []
    for event in records:
        images = "".join(f'<img width="250" src="{html.escape(os.path.relpath(path, output))}">' for path in event["evidence_frames"])
        blocks.append(f'<article><h2>{html.escape(event["id"])}</h2><p>{html.escape(event["caption"])}</p>'
                      f'<p>Confidence: {event["confidence"]:.3f}; review: {event["review"]}</p>{images}</article>')
    page = '<!doctype html><meta charset="utf-8"><title>TF-DVEE evidence</title><style>body{font:16px sans-serif;margin:2rem;max-width:1100px}article{margin-bottom:2rem}img{margin:4px}</style>' + "".join(blocks)
    (output / "evidence.html").write_text(page, encoding="utf-8")
