"""Actual rendered commissioning/event videos with separate device/event annotations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def render(time, rng, normal=False, tone=100):
    image = np.full((64, 64, 3), tone, dtype=np.float32)
    image[18:45, 20:44] += 25
    # Physical fluid and unrelated signage are separate rendered scene mechanisms.
    fluid = not normal and 10 <= time < 22
    panel = not normal and 34 <= time < 44
    if fluid:
        image[43:61, 27:38] = [35, 65, 220]
    if panel:
        image[2:18, 2:24] = [35, 65, 220]
    return np.clip(image + rng.normal(0, 3, image.shape), 0, 255).astype(np.uint8)


def video(path, seed, normal, tone):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPG video writer is unavailable")
    try:
        for frame in range(120):
            writer.write(cv2.cvtColor(render(frame / 2, rng, normal, tone), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def generate(output, seed):
    output.mkdir(parents=True, exist_ok=False)
    vocabulary = dict(asset="pump", normal=dict(state="operating normally", cue="dry gray housing",
                      concepts=["pump", "dry", "housing"]), deviations=[
                      dict(id="visible_leak", state="visible leakage", cue="blue liquid below housing",
                           concepts=["pump", "blue", "liquid"], safety=False)])
    (output / "states.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    dev, test = {}, {}
    for index, device in enumerate(("dev-A", "dev-B", "test-C")):
        tone = 90 + index * 10
        video(output / f"normal-{device}.avi", seed + index * 10, True, tone)
        video(output / f"event-{device}.avi", seed + index * 10 + 1, False, tone)
        target = dict(device_group=device, events=[dict(state="visible_leak", start=10, end=22,
                      concepts=["pump", "blue", "liquid"])])
        (test if device.startswith("test") else dev)[device] = target
    (output / "development_truth.json").write_text(json.dumps(dict(videos=dev), indent=2), encoding="utf-8")
    (output / "test_truth.json").write_text(json.dumps(dict(videos=test), indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(source="finite_scene_videos",
        visual_encoder="finite_rgb_v1", captioner="finite_rgb_rules_v1",
        labels="physical fluid interval; unrelated blue signage is negative",
        calibration="two development devices, held-out test device; no target label enters the frontend"), indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    generate(Path(args.output), args.seed)


if __name__ == "__main__":
    main()
