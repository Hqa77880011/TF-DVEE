import argparse
import json
from pathlib import Path

import numpy as np

from .frontend import prepare_video
from .metrics import compare, evaluate_files
from .run import run
from .temporal import VARIANTS
from .calibration import fit as fit_confidence


def plot(signals, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if signals:
        with np.load(signals, allow_pickle=False) as archive:
            timestamps = archive["timestamps"]
            filtered = archive["filtered"]
            contrast = archive["contrast"]
            states = archive["states"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for state, name in enumerate(states):
            axes[0].plot(timestamps, contrast[:, state], label=name)
            axes[1].plot(timestamps, filtered[:, state], label=name)
        axes[0].set_ylabel("Normal-reference contrast")
        axes[1].set(xlabel="Time (s)", ylabel="Multiscale response")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / "signals.png", dpi=180)
        plt.close(fig)
    if metrics:
        records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in metrics]
        fig, ax = plt.subplots(figsize=(6, 4))
        for record in records:
            curve = record["risk_coverage"]
            ax.plot(curve["coverage"], curve["risk"], label=record["variant"])
        ax.set(xlabel="Coverage", ylabel="Event error risk")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(destination / "risk_coverage.png", dpi=180)
        plt.close(fig)
        for metric in ("map50", "map75", "concept_f1", "ece", "false_alarms_per_hour", "evidence_sufficiency"):
            available = [record for record in records if record["overall"][metric] is not None]
            if not available:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([record["variant"] for record in available],
                   [record["overall"][metric] for record in available], color="#518393")
            ax.set_ylabel(metric.replace("_", " "))
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(destination / f"{metric}.png", dpi=180)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="python -m tfdvee")
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare")
    data.add_argument("--video", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    data.add_argument("--device", default="cpu")
    data.add_argument("--fps", type=float, default=1)
    data.add_argument("--batch-size", type=int, default=32)
    data.add_argument("--crop", type=int, nargs=4)
    inference = commands.add_parser("run")
    inference.add_argument("--features", required=True)
    inference.add_argument("--normal", required=True)
    inference.add_argument("--vocabulary", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--video-id", required=True)
    inference.add_argument("--device-group", required=True)
    inference.add_argument("--condition", default="nominal")
    inference.add_argument("--variant", choices=VARIANTS, default="tfdvee")
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--caption-model", default="Salesforce/blip2-flan-t5-xl")
    inference.add_argument("--temperature", type=float, default=0.07)
    inference.add_argument("--normal-coefficient", type=float, default=1)
    inference.add_argument("--windows", type=float, nargs="+", default=[1, 3, 7])
    inference.add_argument("--high", type=float, default=3)
    inference.add_argument("--low", type=float, default=1.5)
    inference.add_argument("--persistence", type=float, default=2)
    inference.add_argument("--merge-gap", type=float, default=2)
    inference.add_argument("--pre-context", type=float, default=10)
    inference.add_argument("--transition-threshold", type=float, default=0.5)
    inference.add_argument("--evidence-alpha", type=float, default=0.5)
    inference.add_argument("--confidence-coefficients", type=float, nargs=4, default=[0.5, 2, 0.5, 1])
    inference.add_argument("--confidence-calibration", help="Device-disjoint development artifact from calibrate")
    inference.add_argument("--confidence-threshold", type=float, default=0.7)
    inference.add_argument("--evidence-threshold", type=float, default=0.3)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--predictions", nargs="+", required=True)
    calibration.add_argument("--truth", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--regularization", type=float, default=0.001)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--predictions", nargs="+", required=True)
    evaluation.add_argument("--truth", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--ratings")
    evaluation.add_argument("--reviews")
    paired = commands.add_parser("compare")
    paired.add_argument("--first", required=True)
    paired.add_argument("--second", required=True)
    paired.add_argument("--output", required=True)
    paired.add_argument("--resamples", type=int, default=10000)
    visual = commands.add_parser("plot")
    visual.add_argument("--signals")
    visual.add_argument("--metrics", nargs="+")
    visual.add_argument("--output", default="outputs/figures")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_video(args.video, args.output, args.clip_model, args.device, args.fps, args.crop, args.batch_size)
    elif args.command == "run":
        run(args)
    elif args.command == "evaluate":
        evaluate_files(args.predictions, args.truth, args.output, args.ratings, args.reviews)
    elif args.command == "calibrate":
        fit_confidence(args.predictions, args.truth, args.output, args.regularization)
    elif args.command == "compare":
        compare(args.first, args.second, args.output, args.resamples)
    else:
        if not args.signals and not args.metrics:
            parser.error("plot requires --signals or --metrics")
        plot(args.signals, args.metrics, args.output)


if __name__ == "__main__":
    main()
