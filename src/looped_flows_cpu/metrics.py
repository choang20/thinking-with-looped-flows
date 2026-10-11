"""Metric records and aggregation for inference scaling runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ExampleMetrics:
    exact_correct: bool
    correct_cells: int
    total_cells: int
    model_calls: int
    wall_clock_ms: float


def aggregate_metrics(
    model: str,
    steps: int,
    records: Iterable[ExampleMetrics],
) -> dict[str, float | int | str]:
    """Aggregate per-example outcomes without averaging ratios incorrectly."""

    materialized = list(records)
    if not materialized:
        raise ValueError("cannot aggregate an empty metric collection")
    total_cells = sum(record.total_cells for record in materialized)
    if total_cells <= 0:
        raise ValueError("metric records must contain cells")
    return {
        "model": model,
        "steps": steps,
        "examples": len(materialized),
        "exact_accuracy": sum(
            1 for record in materialized if record.exact_correct
        )
        / len(materialized),
        "cell_accuracy": sum(
            record.correct_cells for record in materialized
        )
        / total_cells,
        "model_calls": sum(record.model_calls for record in materialized),
        "wall_clock_ms": sum(
            record.wall_clock_ms for record in materialized
        ),
    }
