import numpy as np

from looped_flows_cpu.data import make_dataset
from looped_flows_cpu.flow import (
    TrainingConfig,
    evaluate_steps,
    integrate_probability_flow,
    train_denoiser,
)
from looped_flows_cpu.model import DenoiserConfig


def test_inference_is_deterministic_for_a_fixed_seed() -> None:
    dataset = make_dataset(5, seed=8)
    model, _ = train_denoiser(
        dataset,
        model_config=DenoiserConfig(hidden=12),
        training_config=TrainingConfig(
            steps=3,
            flow_steps=2,
            batch_size=5,
            seed=4,
        ),
    )
    first = integrate_probability_flow(model, dataset, steps=4, seed=9)
    second = integrate_probability_flow(model, dataset, steps=4, seed=9)
    assert np.array_equal(first, second)


def test_evaluation_counts_model_calls() -> None:
    dataset = make_dataset(5, seed=10)
    model, _ = train_denoiser(
        dataset,
        model_config=DenoiserConfig(hidden=12),
        training_config=TrainingConfig(
            steps=2,
            flow_steps=2,
            batch_size=5,
            seed=5,
        ),
    )
    rows = evaluate_steps(
        model,
        dataset,
        step_values=[2, 8],
        seed=2,
        variant="test",
    )
    assert [row.model_calls for row in rows] == [10, 40]
