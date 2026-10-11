from looped_flows_cpu.metrics import ExampleMetrics, aggregate_metrics


def test_metric_aggregation_uses_global_cell_denominator():
    result = aggregate_metrics(
        "recurrent",
        8,
        [
            ExampleMetrics(True, 16, 16, 8, 1.5),
            ExampleMetrics(False, 8, 16, 8, 2.5),
        ],
    )

    assert result["model"] == "recurrent"
    assert result["steps"] == 8
    assert result["examples"] == 2
    assert result["exact_accuracy"] == 0.5
    assert result["cell_accuracy"] == 0.75
    assert result["model_calls"] == 16
    assert result["wall_clock_ms"] == 4.0
