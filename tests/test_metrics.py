from pathlib import Path

from looped_flows_cpu.flow import EvaluationRow
from looped_flows_cpu.metrics import write_metrics_csv


def test_metrics_csv_contains_scaling_columns(tmp_path: Path) -> None:
    output = tmp_path / "metrics.csv"
    write_metrics_csv(
        [
            EvaluationRow(
                variant="test",
                flow_steps=4,
                accuracy=0.5,
                model_calls=12,
                elapsed_seconds=0.12,
            )
        ],
        output,
    )
    content = output.read_text(encoding="utf-8")
    assert "flow_steps" in content
    assert "mean_latency_ms" in content
    assert "0.5" in content
