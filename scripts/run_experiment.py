#!/usr/bin/env python3
"""Run the offline CPU scaling experiment and publish complete artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from looped_flows_cpu import __version__
from looped_flows_cpu.experiment import ExperimentConfig, run_experiment


def _git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def _svg_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _render_scaling_svg(
    metrics: Sequence[dict[str, Any]],
    width: int = 780,
    height: int = 440,
) -> str:
    plot_left, plot_top = 74, 54
    plot_width, plot_height = 620, 290
    steps = sorted({int(row["steps"]) for row in metrics})
    series = [
        ("recurrent", "exact_accuracy", "#0b6e4f", "Recurrent exact"),
        ("recurrent", "cell_accuracy", "#2f9e44", "Recurrent cell"),
        ("no_recurrent_state", "exact_accuracy", "#9c2c77", "Baseline exact"),
        ("no_recurrent_state", "cell_accuracy", "#d9480f", "Baseline cell"),
    ]
    points = []
    for model, metric, color, label in series:
        values = {
            int(row["steps"]): float(row[metric])
            for row in metrics
            if row["model"] == model
        }
        coordinates = []
        for step in steps:
            x = plot_left + (
                0
                if len(steps) == 1
                else (step - steps[0]) / (steps[-1] - steps[0]) * plot_width
            )
            y = plot_top + (1.0 - values[step]) * plot_height
            coordinates.append(f"{x:.1f},{y:.1f}")
        points.append(
            f'<polyline points="{" ".join(coordinates)}" '
            f'stroke="{color}" fill="none" stroke-width="3"/>'
        )
        for step in steps:
            x = plot_left + (
                0
                if len(steps) == 1
                else (step - steps[0]) / (steps[-1] - steps[0]) * plot_width
            )
            y = plot_top + (1.0 - values[step]) * plot_height
            points.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            )
    grid = []
    for tick in range(0, 6):
        value = tick / 5
        y = plot_top + (1.0 - value) * plot_height
        grid.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" '
            f'x2="{plot_left + plot_width}" y2="{y:.1f}" '
            'stroke="#d9e2ec" stroke-width="1"/>'
        )
        grid.append(
            f'<text x="{plot_left - 12}" y="{y + 4:.1f}" '
            'text-anchor="end" font-size="12" fill="#52606d">'
            f"{value:.1f}</text>"
        )
    for step in steps:
        x = plot_left + (
            0
            if len(steps) == 1
            else (step - steps[0]) / (steps[-1] - steps[0]) * plot_width
        )
        grid.append(
            f'<text x="{x:.1f}" y="{plot_top + plot_height + 24}" '
            f'text-anchor="middle" font-size="12" fill="#52606d">{step}</text>'
        )
    legend = []
    for index, (_, _, color, label) in enumerate(series):
        x = plot_left + (index % 2) * 280
        y = height - 54 + (index // 2) * 20
        legend.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 24}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
            f'<text x="{x + 32}" y="{y + 4}" font-size="12" '
            f'fill="#243b53">{_svg_escape(label)}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
viewBox="0 0 {width} {height}" role="img"
aria-labelledby="title description">
<title id="title">Inference-time scaling on 4x4 Sudoku</title>
<desc id="description">Exact and cell accuracy for recurrent and no-recurrent models at increasing Euler step counts.</desc>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-size="18"
font-family="sans-serif" font-weight="600" fill="#102a43">Inference-time scaling</text>
{"".join(grid)}
<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}"
y2="{plot_top + plot_height}" stroke="#243b53" stroke-width="1.5"/>
<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}"
stroke="#243b53" stroke-width="1.5"/>
{"".join(points)}
<text x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 48}"
text-anchor="middle" font-size="13" fill="#243b53">Euler inference steps</text>
<text x="18" y="{plot_top + plot_height / 2}" text-anchor="middle"
font-size="13" fill="#243b53" transform="rotate(-90 18 {plot_top + plot_height / 2})">Accuracy</text>
{"".join(legend)}
</svg>
"""


def _metadata(config: ExperimentConfig, command: str) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "versions": {
            "looped_flows_cpu": __version__,
            "python": sys.version.split()[0],
        },
        "execution": {"device": "CPU", "numeric_dtype": "float32"},
        "seed": config.seed,
        "command": command,
        "git_revision": _git_revision(),
    }


def _write_artifacts(
    output_dir: Path,
    metrics: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".inference-scaling-", dir=output_dir
    ) as staging_name:
        staging = Path(staging_name)
        (staging / "metrics.json").write_text(
            json.dumps(list(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        fieldnames = [
            "model",
            "steps",
            "examples",
            "exact_accuracy",
            "cell_accuracy",
            "model_calls",
            "wall_clock_ms",
        ]
        with (staging / "metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "inference_scaling.svg").write_text(
            _render_scaling_svg(metrics),
            encoding="utf-8",
        )
        for name in (
            "metrics.json",
            "metrics.csv",
            "summary.json",
            "metadata.json",
            "inference_scaling.svg",
        ):
            os.replace(staging / name, output_dir / name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_mapping(
        json.loads(args.config.read_text(encoding="utf-8"))
    )
    command = shlex.join([sys.executable, *sys.argv])
    result = run_experiment(config)
    metadata = _metadata(config, command)
    _write_artifacts(args.output, result.metrics, result.summary, metadata)
    print(json.dumps({"output": str(args.output), "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
