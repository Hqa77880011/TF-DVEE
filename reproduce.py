"""Frozen local CLIP/BLIP2, 1/3/7s causal windows and 3/1.5 hysteresis; monotone confidence fit on device-disjoint annotated development events."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "tfdvee", "calibrate", "--predictions", *args.development_predictions, "--truth", args.development_truth, "--output", str(output / "confidence.json")],
        ["-m", "tfdvee", "run", "--features", args.features, "--normal", args.normal, "--vocabulary", args.vocabulary, "--caption-model", args.caption_model, "--video-id", args.video_id, "--device-group", args.device_group, "--confidence-calibration", str(output / "confidence.json"), "--output", str(output / "events")],
        ["-m", "tfdvee", "evaluate", "--predictions", str(output / "events/events.json"), "--truth", args.truth, "--output", str(output / "metrics.json")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='outputs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--features', required=True, type=absolute_path)
    parser.add_argument('--normal', required=True, type=absolute_path)
    parser.add_argument('--vocabulary', required=True, type=absolute_path)
    parser.add_argument('--caption-model', required=True, type=absolute_path)
    parser.add_argument('--video-id', required=True)
    parser.add_argument('--device-group', required=True)
    parser.add_argument("--development-predictions", nargs="+", required=True, type=absolute_path)
    parser.add_argument('--development-truth', required=True, type=absolute_path)
    parser.add_argument('--truth', required=True, type=absolute_path)
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
