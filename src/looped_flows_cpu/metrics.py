"""Small result-writing helpers with no dataframe dependency."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .flow import EvaluationRow


def write_metrics_csv(rows: Iterable[EvaluationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "flow_steps",
        "accuracy",
        "model_calls",
        "elapsed_seconds",
        "mean_latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["mean_latency_ms"] = row.mean_latency_ms
            writer.writerow(values)


def write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
