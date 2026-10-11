"""End-to-end training and inference evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

from .dataset import CELL_COUNT, make_dataset
from .flow import euler_probability_flow
from .metrics import ExampleMetrics, aggregate_metrics
from .model import CategoricalDenoiser, ModelConfig


@dataclass(frozen=True)
class DatasetConfig:
    train_size: int = 48
    test_size: int = 24
    min_clues: int = 8
    max_clues: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 18
    times: int = 5
    learning_rate: float = 0.08


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1729
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig(recurrent=True)
    baseline: ModelConfig = ModelConfig(recurrent=False)
    inference_steps: tuple[int, ...] = (4, 8, 16, 32)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ExperimentConfig":
        dataset_value = value.get("dataset", {})
        training_value = value.get("training", {})
        model_value = value.get("model", {})
        baseline_value = value.get("baseline", {})
        inference_steps = tuple(value.get("inference_steps", [4, 8, 16, 32]))
        if not inference_steps or any(step < 1 for step in inference_steps):
            raise ValueError("inference_steps must contain positive integers")
        training = TrainingConfig(
            epochs=int(training_value.get("epochs", 18)),
            times=int(training_value.get("times", 5)),
            learning_rate=float(training_value.get("learning_rate", 0.08)),
        )
        common_model = {
            "hidden_size": 24,
            "candidate_bias": 1.5,
            "epochs": training.epochs,
            "learning_rate": training.learning_rate,
        }
        model = ModelConfig(
            **{
                **common_model,
                **model_value,
                "recurrent": True,
            }
        )
        baseline = ModelConfig(
            **{
                **common_model,
                **baseline_value,
                "recurrent": False,
            }
        )
        return cls(
            seed=int(value.get("seed", 1729)),
            dataset=DatasetConfig(
                train_size=int(dataset_value.get("train_size", 48)),
                test_size=int(dataset_value.get("test_size", 24)),
                min_clues=int(dataset_value.get("min_clues", 8)),
                max_clues=int(dataset_value.get("max_clues", 10)),
            ),
            training=training,
            model=model,
            baseline=baseline,
            inference_steps=inference_steps,
        )


@dataclass(frozen=True)
class ExperimentResult:
    metrics: list[dict[str, float | int | str]]
    summary: dict[str, Any]


def _evaluate(
    model_name: str,
    model: CategoricalDenoiser,
    examples: Sequence,
    steps: int,
) -> dict[str, float | int | str]:
    records: list[ExampleMetrics] = []
    for example in examples:
        started = time.perf_counter()
        output = euler_probability_flow(model, example.puzzle, steps)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        correct_cells = sum(
            predicted == target
            for predicted, target in zip(output.prediction, example.solution)
        )
        records.append(
            ExampleMetrics(
                exact_correct=output.prediction == example.solution,
                correct_cells=correct_cells,
                total_cells=CELL_COUNT,
                model_calls=output.model_calls,
                wall_clock_ms=elapsed_ms,
            )
        )
    return aggregate_metrics(model_name, steps, records)


def _ordered_training_times(count: int, seed: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("training time count must be positive")
    import random

    rng = random.Random(seed)
    values = sorted(rng.random() for _ in range(count))
    return tuple(round(value, 6) for value in values)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Train both models and evaluate every requested inference step count."""

    train, test = make_dataset(
        train_size=config.dataset.train_size,
        test_size=config.dataset.test_size,
        seed=config.seed,
        min_clues=config.dataset.min_clues,
        max_clues=config.dataset.max_clues,
    )
    training_times = _ordered_training_times(
        config.training.times, config.seed + 11
    )
    recurrent_model = CategoricalDenoiser(config.model, seed=config.seed + 101)
    baseline_model = CategoricalDenoiser(config.baseline, seed=config.seed + 202)
    recurrent_history = recurrent_model.fit(train, training_times)
    baseline_history = baseline_model.fit(train, training_times)

    metrics: list[dict[str, float | int | str]] = []
    for model_name, model in (
        ("recurrent", recurrent_model),
        ("no_recurrent_state", baseline_model),
    ):
        for steps in config.inference_steps:
            metrics.append(_evaluate(model_name, model, test, steps))

    best: dict[str, dict[str, float | int | str]] = {}
    for model_name in ("recurrent", "no_recurrent_state"):
        candidates = [row for row in metrics if row["model"] == model_name]
        best[model_name] = max(
            candidates,
            key=lambda row: (
                float(row["exact_accuracy"]),
                float(row["cell_accuracy"]),
                -int(row["steps"]),
            ),
        )
    summary = {
        "seed": config.seed,
        "train_examples": len(train),
        "test_examples": len(test),
        "training_times": list(training_times),
        "inference_steps": list(config.inference_steps),
        "best": best,
        "final_training_loss": {
            "recurrent": recurrent_history[-1],
            "no_recurrent_state": baseline_history[-1],
        },
    }
    return ExperimentResult(metrics=metrics, summary=summary)
