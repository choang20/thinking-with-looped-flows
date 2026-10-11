"""Command-line experiment for the CPU-only looped-flow demonstration."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import make_dataset
from .flow import EvaluationRow, TrainingConfig, evaluate_steps, train_denoiser
from .metrics import write_json, write_metrics_csv
from .model import DenoiserConfig


def _parse_steps(value: str) -> list[int]:
    steps = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not steps or any(step < 1 for step in steps):
        raise argparse.ArgumentTypeError("steps must be positive integers")
    return steps


def _plot_results(rows: list[EvaluationRow], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    variants = sorted({row.variant for row in rows})
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for variant in variants:
        selected = [row for row in rows if row.variant == variant]
        selected.sort(key=lambda row: row.flow_steps)
        axis.plot(
            [row.flow_steps for row in selected],
            [row.accuracy * 100.0 for row in selected],
            marker="o",
            label=variant,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(sorted({row.flow_steps for row in rows}))
    axis.set_xlabel("Probability-flow integration steps (more inference compute)")
    axis.set_ylabel("Exact puzzle accuracy (%)")
    axis.set_title("Inference-time scaling on held-out 4x4 Sudoku")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_summary(rows: list[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment summary",
        "",
        "This result was generated locally from synthetic 4x4 Sudoku data.",
        "The same trained weights are evaluated with different probability-flow",
        "integration grids; increasing `flow_steps` is the inference-time",
        "scaling axis.",
        "",
        "| Variant | Steps | Exact accuracy | Model calls | Mean call latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.variant} | {row.flow_steps} | "
            f"{row.accuracy * 100:.1f}% | {row.model_calls} | "
            f"{row.mean_latency_ms:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: accuracy is the primary measure. Latency is",
            "machine-dependent and is included to show the cost of spending",
            "more inference compute.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> list[EvaluationRow]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = make_dataset(args.train_size, seed=args.seed)
    validation = make_dataset(args.validation_size, seed=args.seed + 1)
    test = make_dataset(args.test_size, seed=args.seed + 2)
    del validation

    model_config = DenoiserConfig(hidden=args.hidden)
    training_config = TrainingConfig(
        steps=args.train_steps,
        flow_steps=args.train_flow_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    recurrent_model, recurrent_losses = train_denoiser(
        train,
        model_config=model_config,
        training_config=training_config,
    )
    no_state_model, no_state_losses = train_denoiser(
        train,
        model_config=DenoiserConfig(
            hidden=args.hidden,
            use_state=False,
        ),
        training_config=training_config,
    )
    rows = evaluate_steps(
        recurrent_model,
        test,
        step_values=args.eval_steps,
        seed=args.seed + 3,
        variant="looped-flow",
    )
    rows.extend(
        evaluate_steps(
            no_state_model,
            test,
            step_values=args.eval_steps,
            seed=args.seed + 3,
            variant="no-state-flow",
        )
    )
    write_metrics_csv(rows, output_dir / "metrics.csv")
    _write_summary(rows, output_dir / "summary.md")
    _plot_results(rows, output_dir / "accuracy_vs_steps.png")
    write_json(
        {
            "task": "unique 4x4 Sudoku completion",
            "seed": args.seed,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "dataset_sizes": {
                "train": train.size,
                "validation": args.validation_size,
                "test": test.size,
            },
            "model": asdict(model_config),
            "training": asdict(training_config),
            "evaluation_steps": args.eval_steps,
            "final_training_loss": {
                "looped-flow": recurrent_losses[-1],
                "no-state-flow": no_state_losses[-1],
            },
            "notes": [
                "All data is generated locally; no cloud or downloaded dataset is used.",
                "Latency depends on the CPU and Python/NumPy build.",
                "The Qwen adapter is optional and is not used for the core result.",
            ],
        },
        output_dir / "metadata.json",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--validation-size", type=int, default=160)
    parser.add_argument("--test-size", type=int, default=240)
    parser.add_argument("--train-steps", type=int, default=1200)
    parser.add_argument("--train-flow-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--eval-steps",
        type=_parse_steps,
        default=[2, 4, 8, 16, 32],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)
    rows = run_experiment(args)
    best = max(rows, key=lambda row: row.accuracy)
    print(
        f"best={best.variant} steps={best.flow_steps} "
        f"accuracy={best.accuracy:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
