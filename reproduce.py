"""Finite RGB videos, closed-vocabulary visual/caption features and held-out-device event confidence calibration."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def build_steps(args):
    output = (ROOT / args.output).resolve()
    generated, encoded = output / "generated", output / "encoded"
    steps = [["finite_data.py", "--output", str(generated), "--seed", "19"]]
    for device in ("dev-A", "dev-B", "test-C"):
        for role in ("normal", "event"):
            steps.append(["-m", "tfdvee", "prepare", "--video", str(generated / f"{role}-{device}.avi"),
                          "--clip-model", "finite_rgb_v1", "--fps", "1",
                          "--output", str(encoded / f"{role}-{device}")])

    def detect(device, destination, calibrated=False):
        command = ["-m", "tfdvee", "run", "--features", str(encoded / f"event-{device}/features.npz"),
                   "--normal", str(encoded / f"normal-{device}/features.npz"),
                   "--vocabulary", str(generated / "states.json"),
                   "--caption-model", "finite_rgb_rules_v1",
                   "--video-id", device, "--device-group", device,
                   "--output", str(output / destination)]
        if calibrated:
            command += ["--confidence-calibration", str(output / "confidence.json")]
        return command

    steps += [detect(device, device) for device in ("dev-A", "dev-B")]
    steps.append(["-m", "tfdvee", "calibrate", "--predictions",
                  str(output / "dev-A/events.json"), str(output / "dev-B/events.json"),
                  "--truth", str(generated / "development_truth.json"),
                  "--output", str(output / "confidence.json")])
    steps.append(detect("test-C", "events", calibrated=True))
    steps.append(["-m", "tfdvee", "evaluate", "--predictions", str(output / "events/events.json"),
                  "--truth", str(generated / "test_truth.json"),
                  "--output", str(output / "metrics.json")])
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/finite-setting")
    parser.add_argument("--plan", action="store_true",
                        help="Print every command without generating videos, fitting or evaluating")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
