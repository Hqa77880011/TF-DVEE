import numpy as np
from scipy.special import expit


VARIANTS = ("tfdvee", "frame_retrieval", "no_temporal", "no_normal_contrast",
            "no_transition", "no_evidence", "no_calibration", "generic_prompts", "no_referral")


def contrast(features, prompt_features, normal_features, temperature=0.07, coefficient=1):
    prototype = normal_features.mean(0)
    prototype /= max(np.linalg.norm(prototype), 1e-8)
    scores = features @ prompt_features.T / temperature
    difference = scores[:, 1:] - scores[:, :1]
    return difference + coefficient * (1 - features @ prototype)[:, None]


def multiscale(signal, normal_signal, windows, fps):
    location = np.median(normal_signal, axis=0)
    scale = 1.4826 * np.median(np.abs(normal_signal - location), axis=0)
    scale = np.maximum(scale, 1e-3)
    values = []
    for seconds in windows:
        steps = max(1, int(round(seconds * fps)) + 1)
        aggregated = np.empty_like(signal)
        for time in range(len(signal)):
            aggregated[time] = np.median(signal[max(0, time - steps + 1):time + 1], axis=0)
        values.append((aggregated - location) / scale)
    values = np.stack(values)
    return values.max(0), values


def segment(signal, high, low, persistence, merge_gap, fps):
    if high <= low or persistence < 0 or merge_gap < 0:
        raise ValueError("Require high > low and nonnegative persistence/merge gap")
    delay = max(1, int(round(persistence * fps)))
    gap = max(0, int(round(merge_gap * fps)))
    events = []
    for state in range(signal.shape[1]):
        current = None
        below = 0
        intervals = []
        for time, value in enumerate(signal[:, state]):
            if current is None:
                if value >= high:
                    current, below = time, 0
            elif value < low:
                below += 1
                if below >= delay:
                    intervals.append([current, time - below + 1])
                    current, below = None, 0
            else:
                below = 0
        if current is not None:
            intervals.append([current, len(signal)])
        merged = []
        for start, end in intervals:
            if merged and start - merged[-1][1] <= gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        events.extend(dict(state=state, start=start, end=end) for start, end in merged if end > start)
    return sorted(events, key=lambda event: (event["start"], event["state"]))


def event_statistics(event, signal, scales, pre_steps):
    start, end, state = event["start"], event["end"], event["state"]
    inside = signal[start:end, state]
    context = signal[max(0, start - pre_steps):start, state]
    transition = float(inside.mean() - context.mean()) if len(context) else 0.0
    peak = start + int(inside.argmax())
    differences = np.diff(signal[max(0, start - 1):end, state], prepend=signal[max(0, start - 1), state])
    change = max(start, max(0, start - 1) + int(differences.argmax()))
    change = min(change, end - 1)
    pre = max(0, start - max(1, pre_steps // 2))
    means = signal[start:end].mean(0)
    others = np.delete(means, state)
    margin = max(0, float(means[state] - others.max())) if len(others) else max(0, float(means[state]))
    disagreement = float(scales[:, start:end, state].std(0).mean())
    return dict(transition=transition, no_pre_context=len(context) == 0,
                strength=float(inside.mean()), disagreement=disagreement,
                ambiguity=float(np.exp(-margin)), evidence_indices=[peak, change, pre])


def confidence(strength, consistency, disagreement, ambiguity, coefficients):
    first, second, third, fourth = coefficients
    return float(expit(first * strength + second * consistency - third * disagreement - fourth * ambiguity))
