"""A tiny NumPy stateful categorical denoiser.

This is intentionally transparent rather than optimized.  The model is a
single shared tanh block whose hidden activation is also the recurrent state.
During training, callers detach the state between denoising times, matching
the paper's local-loss/no-full-BPTT idea.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DenoiserConfig:
    length: int = 16
    vocab: int = 4
    hidden: int = 96
    use_state: bool = True


class StatefulDenoiser:
    """Small stateful denoiser with manual gradients for offline CPU training."""

    def __init__(self, config: DenoiserConfig, *, seed: int = 0) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        state_width = config.hidden if config.use_state else 0
        input_width = (
            config.length * config.vocab
            + config.length * config.vocab
            + config.length
            + 2
            + state_width
        )
        scale = np.sqrt(2.0 / input_width)
        self.parameters: dict[str, np.ndarray] = {
            "input_weight": (
                rng.standard_normal((input_width, config.hidden)).astype(np.float32)
                * scale
            ),
            "input_bias": np.zeros(config.hidden, dtype=np.float32),
            "output_weight": (
                rng.standard_normal(
                    (config.hidden, config.length * config.vocab)
                ).astype(np.float32)
                * np.sqrt(2.0 / config.hidden)
            ),
            "output_bias": np.zeros(
                config.length * config.vocab, dtype=np.float32
            ),
        }

    def zero_state(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.config.hidden), dtype=np.float32)

    def _features(
        self,
        x: np.ndarray,
        condition: np.ndarray,
        clue_mask: np.ndarray,
        time: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        pieces = [
            x.reshape(x.shape[0], -1),
            condition.reshape(condition.shape[0], -1),
            clue_mask.astype(np.float32),
            np.stack((time, 1.0 - time), axis=1).astype(np.float32),
        ]
        if self.config.use_state:
            pieces.append(state.astype(np.float32))
        return np.concatenate(pieces, axis=1)

    def forward(
        self,
        x: np.ndarray,
        condition: np.ndarray,
        clue_mask: np.ndarray,
        time: np.ndarray,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Return probabilities, next state, and a cache for ``backward``."""

        features = self._features(x, condition, clue_mask, time, state)
        pre_activation = (
            features @ self.parameters["input_weight"]
            + self.parameters["input_bias"]
        )
        next_state = np.tanh(pre_activation).astype(np.float32)
        logits = (
            next_state @ self.parameters["output_weight"]
            + self.parameters["output_bias"]
        ).reshape(-1, self.config.length, self.config.vocab)
        logits = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        cache = {
            "features": features,
            "next_state": next_state,
            "probabilities": probabilities,
        }
        return probabilities, next_state, cache

    def backward(
        self,
        cache: dict[str, np.ndarray],
        targets: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Compute cross-entropy loss and gradients for one local objective."""

        probabilities = cache["probabilities"]
        valid_count = float(max(loss_mask.sum(), 1.0))
        log_probabilities = np.log(np.clip(probabilities, 1e-7, 1.0))
        loss = float(
            -np.sum(loss_mask[:, :, None] * targets * log_probabilities)
            / valid_count
        )

        grad_logits = (
            (probabilities - targets)
            * loss_mask[:, :, None]
            / valid_count
        ).reshape(probabilities.shape[0], -1)
        next_state = cache["next_state"]
        features = cache["features"]
        gradients = {
            "output_weight": next_state.T @ grad_logits,
            "output_bias": grad_logits.sum(axis=0),
        }
        grad_state = grad_logits @ self.parameters["output_weight"].T
        grad_pre_activation = grad_state * (1.0 - next_state**2)
        gradients["input_weight"] = features.T @ grad_pre_activation
        gradients["input_bias"] = grad_pre_activation.sum(axis=0)
        return loss, gradients

    def predict(
        self,
        x: np.ndarray,
        condition: np.ndarray,
        clue_mask: np.ndarray,
        time: np.ndarray,
        state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities, next_state, _ = self.forward(
            x, condition, clue_mask, time, state
        )
        return probabilities, next_state


class Adam:
    """Minimal Adam optimizer for the NumPy model."""

    def __init__(
        self,
        parameters: dict[str, np.ndarray],
        *,
        learning_rate: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        self.first_moment = {
            name: np.zeros_like(value) for name, value in parameters.items()
        }
        self.second_moment = {
            name: np.zeros_like(value) for name, value in parameters.items()
        }

    def step(self, gradients: dict[str, np.ndarray]) -> None:
        self.step_count += 1
        bias_correction_1 = 1.0 - self.beta1**self.step_count
        bias_correction_2 = 1.0 - self.beta2**self.step_count
        for name, parameter in self.parameters.items():
            gradient = gradients[name].astype(np.float32)
            self.first_moment[name] = (
                self.beta1 * self.first_moment[name]
                + (1.0 - self.beta1) * gradient
            )
            self.second_moment[name] = (
                self.beta2 * self.second_moment[name]
                + (1.0 - self.beta2) * gradient**2
            )
            first = self.first_moment[name] / bias_correction_1
            second = self.second_moment[name] / bias_correction_2
            parameter -= self.learning_rate * first / (
                np.sqrt(second) + self.epsilon
            )


def accumulate_gradients(
    total: dict[str, np.ndarray],
    update: dict[str, np.ndarray],
) -> None:
    for name in total:
        total[name] += update[name]
