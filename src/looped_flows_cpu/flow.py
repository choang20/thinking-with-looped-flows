"""Probability-flow integration and local training utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .data import LENGTH, PuzzleDataset, VOCAB
from .model import Adam, DenoiserConfig, StatefulDenoiser, accumulate_gradients


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 1_200
    flow_steps: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    sigma: float = 1.0
    seed: int = 7


def _initial_noise(
    targets: np.ndarray,
    clue_mask: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
) -> np.ndarray:
    noise = (
        sigma
        * rng.standard_normal(targets.shape).astype(np.float32)
    )
    return np.where(clue_mask[:, :, None], targets, noise)


def train_denoiser(
    dataset: PuzzleDataset,
    *,
    model_config: DenoiserConfig,
    training_config: TrainingConfig,
) -> tuple[StatefulDenoiser, list[float]]:
    """Train with ordered local denoising objectives and detached state."""

    if training_config.flow_steps < 1:
        raise ValueError("flow_steps must be positive")
    if training_config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    model = StatefulDenoiser(model_config, seed=training_config.seed)
    optimizer = Adam(
        model.parameters,
        learning_rate=training_config.learning_rate,
    )
    rng = np.random.default_rng(training_config.seed + 1)
    targets = dataset.target_one_hot
    conditions = dataset.condition_one_hot
    unknown_mask = dataset.unknown_mask
    losses: list[float] = []

    for _ in range(training_config.steps):
        batch_indices = rng.integers(
            0, dataset.size, size=training_config.batch_size
        )
        batch_targets = targets[batch_indices]
        batch_conditions = conditions[batch_indices]
        batch_clue_mask = dataset.clue_mask[batch_indices]
        batch_unknown_mask = unknown_mask[batch_indices]
        x0 = _initial_noise(
            batch_targets,
            batch_clue_mask,
            rng,
            training_config.sigma,
        )
        times = np.sort(
            rng.uniform(
                0.0,
                1.0,
                size=(training_config.batch_size, training_config.flow_steps),
            ),
            axis=1,
        ).astype(np.float32)
        state = model.zero_state(training_config.batch_size)
        gradients = {
            name: np.zeros_like(value)
            for name, value in model.parameters.items()
        }
        local_losses: list[float] = []

        for flow_index in range(training_config.flow_steps):
            time_values = times[:, flow_index]
            interpolant = (
                (1.0 - time_values[:, None, None]) * x0
                + time_values[:, None, None] * batch_targets
            )
            probabilities, next_state, cache = model.forward(
                interpolant,
                batch_conditions,
                batch_clue_mask,
                time_values,
                state,
            )
            del probabilities
            local_loss, local_gradients = model.backward(
                cache,
                batch_targets,
                batch_unknown_mask,
            )
            local_losses.append(local_loss)
            accumulate_gradients(gradients, local_gradients)
            # This is the local-loss approximation: future losses use the
            # state value, but not its gradient history.
            state = next_state.copy()

        for name in gradients:
            gradients[name] /= float(training_config.flow_steps)
        optimizer.step(gradients)
        losses.append(float(np.mean(local_losses)))

    return model, losses


def integrate_probability_flow(
    model: StatefulDenoiser,
    dataset: PuzzleDataset,
    *,
    steps: int,
    seed: int,
    sigma: float = 1.0,
    carry_state: bool = True,
) -> np.ndarray:
    """Integrate the learned flow and return integer predictions."""

    if steps < 1:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    targets = dataset.target_one_hot
    x = _initial_noise(targets, dataset.clue_mask, rng, sigma)
    state = model.zero_state(dataset.size)
    condition = dataset.condition_one_hot
    clue_mask = dataset.clue_mask

    for index in range(steps):
        t = np.float32(index / steps)
        dt = np.float32(1.0 / steps)
        time_values = np.full(dataset.size, t, dtype=np.float32)
        probabilities, next_state = model.predict(
            x,
            condition,
            clue_mask,
            time_values,
            state,
        )
        velocity = (probabilities - x) / max(1.0 - t, 1e-6)
        x = x + dt * velocity
        x = np.where(clue_mask[:, :, None], targets, x)
        state = next_state if carry_state else model.zero_state(dataset.size)

    return np.argmax(x, axis=-1).astype(np.int64)


@dataclass(frozen=True)
class EvaluationRow:
    variant: str
    flow_steps: int
    accuracy: float
    model_calls: int
    elapsed_seconds: float

    @property
    def mean_latency_ms(self) -> float:
        return self.elapsed_seconds * 1_000.0 / max(self.model_calls, 1)


def evaluate_steps(
    model: StatefulDenoiser,
    dataset: PuzzleDataset,
    *,
    step_values: list[int],
    seed: int,
    variant: str,
    carry_state: bool = True,
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for steps in step_values:
        started = time.perf_counter()
        predictions = integrate_probability_flow(
            model,
            dataset,
            steps=steps,
            seed=seed,
            carry_state=carry_state,
        )
        elapsed = time.perf_counter() - started
        accuracy = float(
            np.mean(np.all(predictions == dataset.solutions, axis=1))
        )
        rows.append(
            EvaluationRow(
                variant=variant,
                flow_steps=steps,
                accuracy=accuracy,
                model_calls=dataset.size * steps,
                elapsed_seconds=elapsed,
            )
        )
    return rows
