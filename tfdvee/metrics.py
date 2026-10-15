import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score


def temporal_iou(first, second):
    intersection = max(0, min(first["end"], second["end"]) - max(first["start"], second["start"]))
    union = max(first["end"], second["end"]) - min(first["start"], second["start"])
    return intersection / union if union > 0 else 0.0


def match(predictions, truth, threshold):
    used = set()
    output = []
    for prediction in sorted(predictions, key=lambda event: -event["confidence"]):
        possible = [(index, event) for index, event in enumerate(truth)
                    if index not in used and event["video_id"] == prediction["video_id"]
                    and event["state"] == prediction["state"]]
        chosen = max(possible, key=lambda item: temporal_iou(prediction, item[1])) if possible else None
        if chosen is not None and temporal_iou(prediction, chosen[1]) >= threshold:
            used.add(chosen[0])
            output.append((prediction, chosen[1]))
        else:
            output.append((prediction, None))
    return output, used


def average_precision(predictions, truth, threshold):
    states = sorted({event["state"] for event in truth})
    scores = []
    for state in states:
        target = [event for event in truth if event["state"] == state]
        guessed = [event for event in predictions if event["state"] == state]
        pairs, _ = match(guessed, target, threshold)
        correct = np.array([truth_event is not None for _, truth_event in pairs], dtype=float)
        if not len(correct):
            scores.append(0)
            continue
        recall = np.cumsum(correct) / len(target)
        precision = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        increments = np.diff(np.concatenate(([0], recall)))
        scores.append(float(np.sum(increments * precision)))
    return float(np.mean(scores)) if scores else None


def evaluate_events(predictions, truth, duration, ratings=None, reviews=None):
    pairs, used = match(predictions, truth, 0.5)
    confidence = np.array([prediction["confidence"] for prediction, _ in pairs])
    correct = np.array([target is not None for _, target in pairs], dtype=float)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = float(sum(np.mean(bins == b) * abs(confidence[bins == b].mean() - correct[bins == b].mean())
                    for b in range(15) if np.any(bins == b))) if len(pairs) else None
    risk = 1 - np.cumsum(correct) / np.arange(1, len(pairs) + 1) if pairs else np.array([])
    tp = fp = fn = 0
    for prediction, target in pairs:
        a, b = set(prediction.get("concepts", [])), set(target.get("concepts", [])) if target else set()
        tp += len(a & b)
        fp += len(a - b)
        fn += len(b - a)
    for index, target in enumerate(truth):
        if index not in used:
            fn += len(set(target.get("concepts", [])))
    accepted = [event for event in predictions if not event["review"]]
    accepted_pairs, _ = match(accepted, truth, 0.5)
    result = dict(map50=average_precision(predictions, truth, 0.5),
                  map75=average_precision(predictions, truth, 0.75),
                  concept_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                  ece=ece, aurc=float(risk.mean()) if len(risk) else None,
                  false_alarms_per_hour=float(sum(target is None for _, target in pairs) / (duration / 3600)),
                  automated_coverage=len(accepted) / len(predictions) if predictions else 0,
                  automated_risk=float(np.mean([target is None for _, target in accepted_pairs])) if accepted_pairs else None,
                  evidence_sufficiency=None, unsupported_concepts_per_event=None,
                  median_review_minutes_per_12min_video=None, event_count=len(predictions),
                  ground_truth_events=len(truth), duration_seconds=duration)
    if ratings:
        rows = [ratings[event["id"]] for event in predictions if event["id"] in ratings]
        if rows:
            result["evidence_sufficiency"] = float(np.mean([int(row["evidence_sufficient"]) for row in rows]))
            result["unsupported_concepts_per_event"] = float(np.mean([int(row["unsupported_concepts"]) for row in rows]))
            result["rated_events"] = len(rows)
    if reviews:
        result["median_review_minutes_per_12min_video"] = float(np.median(reviews))
    return result, dict(coverage=(np.arange(1, len(risk) + 1) / len(risk)).tolist() if len(risk) else [], risk=risk.tolist())


def evaluate_files(paths, truth_path, output, ratings_path=None, reviews_path=None):
    videos = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    truth_data = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    targets = truth_data["videos"]
    if len({video["video_id"] for video in videos}) != len(videos):
        raise ValueError("One prediction method/run per video is required")
    if len({video["variant"] for video in videos}) != 1:
        raise ValueError("Evaluate each method separately")
    if len({video.get("condition", "nominal") for video in videos}) != 1:
        raise ValueError("Evaluate each disturbance condition separately")
    predictions, truth = [], []
    duration = 0
    by_device = {}
    for video in videos:
        video_id = video["video_id"]
        target = targets[video_id]
        if target["device_group"] != video["device_group"] or video["duration"] <= 0:
            raise ValueError("Prediction/annotation device group or duration mismatch")
        duration += video["duration"]
        current_predictions = [dict(event, video_id=video_id) for event in video["events"]]
        current_truth = [dict(event, video_id=video_id) for event in target["events"]]
        if len({event["id"] for event in current_predictions}) != len(current_predictions):
            raise ValueError(f"Duplicate prediction event IDs in {video_id}")
        for event in current_predictions + current_truth:
            if not 0 <= event["start"] < event["end"] <= video["duration"] + 1e-6:
                raise ValueError(f"Invalid event boundaries in {video_id}")
        if any(not 0 <= event["confidence"] <= 1 for event in current_predictions):
            raise ValueError("Confidence must be in [0,1]")
        predictions.extend(current_predictions)
        truth.extend(current_truth)
        group = by_device.setdefault(video["device_group"], dict(predictions=[], truth=[], duration=0, video_ids=[]))
        group["predictions"].extend(current_predictions)
        group["truth"].extend(current_truth)
        group["duration"] += video["duration"]
        group["video_ids"].append(video_id)
    ratings = None
    if ratings_path:
        with Path(ratings_path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        ratings = {row["event_id"]: row for row in rows}
        if len(ratings) != len(rows):
            raise ValueError("Duplicate event IDs in evidence ratings")
        if any(int(row["evidence_sufficient"]) not in (0, 1) or int(row["unsupported_concepts"]) < 0 for row in rows):
            raise ValueError("Evidence ratings require a binary judgment and nonnegative concept count")
    reviews = []
    by_video_review = {}
    if reviews_path:
        with Path(reviews_path).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                length = next(video["duration"] for video in videos if video["video_id"] == row["video_id"])
                minutes = float(row["review_seconds"]) / length * 12
                if minutes < 0:
                    raise ValueError("Review time cannot be negative")
                reviews.append(minutes)
                by_video_review.setdefault(row["video_id"], []).append(minutes)
    overall, curve = evaluate_events(predictions, truth, duration, ratings, reviews)
    devices = {}
    for name, group in by_device.items():
        group_reviews = [value for video_id in group["video_ids"] for value in by_video_review.get(video_id, [])]
        devices[name] = evaluate_events(group["predictions"], group["truth"], group["duration"], ratings, group_reviews)[0]
    payload = dict(variant=videos[0]["variant"], condition=videos[0].get("condition", "nominal"),
                   overall=overall, devices=devices, risk_coverage=curve)
    if ratings:
        agreement = {}
        for field in ("asset", "state", "evidence", "safety"):
            complete = [row for row in ratings.values() if row.get(f"{field}_r1", "") and row.get(f"{field}_r2", "")]
            if complete:
                first = [row[f"{field}_r1"] for row in complete]
                second = [row[f"{field}_r2"] for row in complete]
                kappa = cohen_kappa_score(first, second)
                agreement[field] = dict(agreement=float(np.mean(np.array(first) == second)),
                                        kappa=float(kappa) if np.isfinite(kappa) else None,
                                        adjudicated_items=sum(row.get("adjudicated", "0") == "1" for row in complete), items=len(complete))
        payload["annotation_agreement"] = agreement
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def compare(first_path, second_path, output, resamples=10000):
    first = json.loads(Path(first_path).read_text(encoding="utf-8"))
    second = json.loads(Path(second_path).read_text(encoding="utf-8"))
    common = sorted(set(first["devices"]) & set(second["devices"]))
    if len(common) < 2:
        raise ValueError("At least two matched device groups required")
    rng = np.random.default_rng(0)
    rows = []
    for metric in ("map50", "map75", "concept_f1", "ece", "aurc", "false_alarms_per_hour", "evidence_sufficiency",
                   "unsupported_concepts_per_event", "median_review_minutes_per_12min_video"):
        pairs = [(first["devices"][group][metric], second["devices"][group][metric]) for group in common]
        if any(a is None or b is None for a, b in pairs):
            continue
        difference = np.array([a - b for a, b in pairs])
        bootstrap = rng.choice(difference, (resamples, len(difference)), replace=True).mean(1)
        rows.append(dict(metric=metric, mean_difference=float(difference.mean()),
                         ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                         pvalue=float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0, device_groups=common))
    previous = 0
    for index, row in enumerate(sorted(rows, key=lambda item: item["pvalue"])):
        previous = max(previous, min(1, (len(rows) - index) * row["pvalue"]))
        row["holm_pvalue"] = previous
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
