"""Fit a frozen event-confidence postprocessor on device-disjoint development data."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .metrics import match


CONTEXT_KEYS = ("variant", "temperature", "normal_coefficient", "windows", "high", "low",
                "persistence", "merge_gap", "pre_context", "transition_threshold",
                "evidence_alpha", "caption_model")


def context(args, clip_model, vocabulary):
    return dict(parameters={key: getattr(args, key) for key in CONTEXT_KEYS},
                clip_model=clip_model, vocabulary=vocabulary)


def features(event):
    values = np.array([event["strength"], event["evidence_consistency"],
                       -event["disagreement"], -event["ambiguity"]], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Confidence features must be finite")
    return values


def predict(event, artifact):
    weights = np.asarray(artifact["coefficients"], dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Calibration requires four finite nonnegative coefficients")
    intercept = float(artifact["intercept"])
    if not np.isfinite(intercept):
        raise ValueError("Calibration intercept must be finite")
    return float(expit(intercept + features(event) @ weights))


def load(path, registered, video_id, device_group):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("protocol") != "tfdvee.confidence.v1" or artifact["context"] != registered:
        raise ValueError("Confidence calibration differs from the frozen detector/frontend configuration")
    if video_id in artifact["video_ids"] or device_group in artifact["device_groups"]:
        raise ValueError("Development and test video/device groups must be disjoint")
    return artifact


def fit(paths, truth_path, output, regularization=0.001):
    """Fit monotone logistic event correctness; never train vision-language models."""
    if regularization <= 0:
        raise ValueError("Use positive confidence regularization")
    videos = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))["videos"]
    if not videos or len({video["video_id"] for video in videos}) != len(videos):
        raise ValueError("Use one development prediction per video")
    registered = videos[0]["calibration_context"]
    rows, labels, groups = [], [], []
    for video in videos:
        video_id, group = video["video_id"], video["device_group"]
        if video["calibration_context"] != registered or video.get("confidence_calibration"):
            raise ValueError("Fit one frozen uncalibrated detector configuration at a time")
        if truth[video_id]["device_group"] != group:
            raise ValueError("Development prediction/annotation device mismatch")
        predictions = [dict(event, video_id=video_id) for event in video["events"]]
        targets = [dict(event, video_id=video_id) for event in truth[video_id]["events"]]
        pairs, _ = match(predictions, targets, 0.5)
        for event, target in pairs:
            rows.append(features(event))
            labels.append(int(target is not None))
            groups.append(group)
    if set(labels) != {0, 1}:
        raise ValueError("Confidence fitting needs both correct and incorrect development events")
    design, labels = np.asarray(rows), np.asarray(labels)
    # Give each device equal weight, rather than weighting long device recordings more.
    weights = np.array([1 / groups.count(group) for group in groups])
    weights /= weights.sum()

    def objective(parameters):
        logits = parameters[0] + design @ parameters[1:]
        value = np.sum(weights * (np.logaddexp(0, logits) - labels * logits))
        value += regularization * np.square(parameters[1:]).sum() / 2
        residual = weights * (expit(logits) - labels)
        gradient = np.r_[residual.sum(), design.T @ residual + regularization * parameters[1:]]
        return float(value), gradient

    solution = minimize(objective, np.zeros(5), jac=True, method="L-BFGS-B",
                        bounds=[(-20, 20)] + [(0, None)] * 4)
    if not solution.success or not np.isfinite(solution.fun):
        raise RuntimeError("Confidence calibration failed: " + solution.message)
    artifact = dict(protocol="tfdvee.confidence.v1", context=registered,
                    intercept=float(solution.x[0]), coefficients=solution.x[1:].tolist(),
                    video_ids=[video["video_id"] for video in videos],
                    device_groups=sorted({video["device_group"] for video in videos}), development_events=len(labels),
                    regularization=regularization, target="same-state one-to-one temporal IoU >= 0.50",
                    interpretation="device-balanced development logistic fit; no distribution-free risk guarantee")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
