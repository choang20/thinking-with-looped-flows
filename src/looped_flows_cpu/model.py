"""A small stateful categorical denoiser trained with pure-Python SGD."""

from __future__ import annotations

from dataclasses import dataclass
from array import array
import math
import random
from typing import Sequence

from .dataset import (
    CELL_COUNT,
    DIGITS,
    SudokuExample,
    candidate_mask,
    initial_state,
)
from .flow import interpolant_state
from .numeric import float32

_STATE_WIDTH = CELL_COUNT * len(DIGITS)
_CLUE_WIDTH = CELL_COUNT * len(DIGITS)
_MASK_WIDTH = CELL_COUNT
_CANDIDATE_WIDTH = CELL_COUNT * len(DIGITS)
_POSITION_WIDTH = CELL_COUNT * CELL_COUNT
_INPUT_WIDTH = (
    _STATE_WIDTH
    + _CLUE_WIDTH
    + _MASK_WIDTH
    + _CANDIDATE_WIDTH
    + _POSITION_WIDTH
    + 1
)
_OUTPUT_WIDTH = _STATE_WIDTH


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 24
    learning_rate: float = 0.08
    epochs: int = 18
    recurrent: bool = True
    candidate_bias: float = 1.5

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.epochs < 1:
            raise ValueError("hidden_size and epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class ForwardCache:
    features: tuple[float, ...]
    previous_state: tuple[float, ...]
    hidden: tuple[float, ...]
    logits: tuple[float, ...]
    candidate: tuple[float, ...]


class CategoricalDenoiser:
    """Task-specific denoiser with an optional detached recurrent state."""

    def __init__(self, config: ModelConfig, seed: int) -> None:
        self.config = config
        self._rng = random.Random(seed)
        hidden = config.hidden_size
        input_scale = math.sqrt(6.0 / (_INPUT_WIDTH + hidden))
        output_scale = math.sqrt(6.0 / (hidden + _OUTPUT_WIDTH))
        self._input_weights = [
            array("f", [
                self._rng.uniform(-input_scale, input_scale)
                for _ in range(_INPUT_WIDTH)
            ])
            for _ in range(hidden)
        ]
        self._recurrent_weights = (
            [
                array("f", [
                    self._rng.uniform(-input_scale, input_scale)
                    for _ in range(hidden)
                ])
                for _ in range(hidden)
            ]
            if config.recurrent
            else []
        )
        self._hidden_bias = array("f", [0.0] * hidden)
        self._output_weights = [
            array("f", [
                self._rng.uniform(-output_scale, output_scale)
                for _ in range(hidden)
            ])
            for _ in range(_OUTPUT_WIDTH)
        ]
        self._output_bias = array("f", [0.0] * _OUTPUT_WIDTH)

    @property
    def recurrent(self) -> bool:
        return self.config.recurrent

    def _features(
        self,
        state: Sequence[Sequence[float]],
        puzzle: Sequence[int],
        time: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        candidates = candidate_mask(puzzle)
        features: list[float] = [
            value
            for cell in range(CELL_COUNT)
            for value in state[cell]
        ]
        clue_features = [
            1.0 if puzzle[cell] != -1 and digit == puzzle[cell] else 0.0
            for cell in range(CELL_COUNT)
            for digit in DIGITS
        ]
        features = features[:_STATE_WIDTH] + clue_features
        features.extend(1.0 if value != -1 else 0.0 for value in puzzle)
        candidate_features = [
            value for cell in candidates for value in cell
        ]
        features.extend(candidate_features)
        features.extend(
            1.0 if cell == position else 0.0
            for cell in range(CELL_COUNT)
            for position in range(CELL_COUNT)
        )
        features.append(time)
        if len(features) != _INPUT_WIDTH:
            raise AssertionError(f"unexpected feature width: {len(features)}")
        return tuple(features), tuple(candidate_features)

    @staticmethod
    def _softmax(logits: Sequence[float]) -> list[float]:
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        total = sum(exponentials)
        return [float32(value / total) for value in exponentials]

    def _forward(
        self,
        state: Sequence[Sequence[float]],
        puzzle: Sequence[int],
        time: float,
        recurrent_state: Sequence[float] | None,
    ) -> tuple[list[float], list[float], ForwardCache]:
        features, candidate = self._features(state, puzzle, time)
        hidden: list[float] = []
        previous = (
            list(recurrent_state)
            if self.config.recurrent and recurrent_state is not None
            else [0.0] * self.config.hidden_size
        )
        for hidden_index in range(self.config.hidden_size):
            activation = self._hidden_bias[hidden_index]
            activation += sum(
                weight * feature
                for weight, feature in zip(
                    self._input_weights[hidden_index], features
                )
            )
            if self.config.recurrent:
                activation += sum(
                    weight * value
                    for weight, value in zip(
                        self._recurrent_weights[hidden_index], previous
                    )
                )
            hidden.append(float32(math.tanh(activation)))

        logits: list[float] = []
        for output_index in range(_OUTPUT_WIDTH):
            value = self._output_bias[output_index] + sum(
                weight * activation
                for weight, activation in zip(
                    self._output_weights[output_index], hidden
                )
            )
            value += self.config.candidate_bias * (
                1.0 if candidate[output_index] else -1.0
            )
            logits.append(float32(value))
        return (
            logits,
            hidden,
            ForwardCache(
                features=features,
                previous_state=tuple(previous),
                hidden=tuple(hidden),
                logits=tuple(logits),
                candidate=candidate,
            ),
        )

    def predict(
        self,
        state: Sequence[Sequence[float]],
        puzzle: Sequence[int],
        time: float,
        recurrent_state: Sequence[float] | None = None,
    ) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...] | None]:
        """Predict one categorical distribution per cell without mutation."""

        logits, hidden, _ = self._forward(state, puzzle, time, recurrent_state)
        probabilities = tuple(
            tuple(
                self._softmax(
                    logits[cell * len(DIGITS) : (cell + 1) * len(DIGITS)]
                )
            )
            for cell in range(CELL_COUNT)
        )
        return probabilities, tuple(hidden) if self.config.recurrent else None

    def _apply_gradient(
        self,
        cache: ForwardCache,
        probabilities: Sequence[float],
        target: Sequence[Sequence[float]],
    ) -> float:
        hidden_size = self.config.hidden_size
        d_logits = [0.0] * _OUTPUT_WIDTH
        loss = 0.0
        for cell in range(CELL_COUNT):
            offset = cell * len(DIGITS)
            for digit in DIGITS:
                target_value = target[cell][digit]
                probability = max(probabilities[offset + digit], 1e-12)
                loss -= target_value * math.log(probability)
                d_logits[offset + digit] = (
                    probabilities[offset + digit] - target_value
                ) / CELL_COUNT

        d_output_weights = [
            [0.0] * hidden_size for _ in range(_OUTPUT_WIDTH)
        ]
        d_output_bias = [0.0] * _OUTPUT_WIDTH
        d_hidden = [0.0] * hidden_size
        for output_index, gradient in enumerate(d_logits):
            d_output_bias[output_index] = gradient
            for hidden_index in range(hidden_size):
                d_output_weights[output_index][hidden_index] = (
                    gradient * cache.hidden[hidden_index]
                )
                d_hidden[hidden_index] += (
                    gradient * self._output_weights[output_index][hidden_index]
                )

        d_hidden_pre = [
            d_hidden[index] * (1.0 - cache.hidden[index] ** 2)
            for index in range(hidden_size)
        ]
        for hidden_index in range(hidden_size):
            for input_index in range(_INPUT_WIDTH):
                self._input_weights[hidden_index][input_index] -= (
                    self.config.learning_rate
                    * d_hidden_pre[hidden_index]
                    * cache.features[input_index]
                )
            self._hidden_bias[hidden_index] -= (
                self.config.learning_rate * d_hidden_pre[hidden_index]
            )
            if self.config.recurrent:
                for previous_index in range(hidden_size):
                    self._recurrent_weights[hidden_index][previous_index] -= (
                        self.config.learning_rate
                        * d_hidden_pre[hidden_index]
                        * cache.previous_state[previous_index]
                    )
        for output_index in range(_OUTPUT_WIDTH):
            for hidden_index in range(hidden_size):
                self._output_weights[output_index][hidden_index] -= (
                    self.config.learning_rate
                    * d_output_weights[output_index][hidden_index]
                )
            self._output_bias[output_index] -= (
                self.config.learning_rate * d_output_bias[output_index]
            )
        return loss / CELL_COUNT

    def fit(
        self,
        examples: Sequence[SudokuExample],
        training_times: Sequence[float],
    ) -> list[float]:
        """Train with ordered time samples and detached recurrent state."""

        if not examples:
            raise ValueError("at least one training example is required")
        if not training_times or list(training_times) != sorted(training_times):
            raise ValueError("training_times must be a non-empty sorted sequence")
        history: list[float] = []
        for _ in range(self.config.epochs):
            total_loss = 0.0
            updates = 0
            for example in examples:
                detached_state: tuple[float, ...] | None = None
                for time in training_times:
                    state = interpolant_state(
                        example.puzzle, example.solution, time
                    )
                    logits, hidden, cache = self._forward(
                        state,
                        example.puzzle,
                        time,
                        detached_state,
                    )
                    probabilities = [
                        probability
                        for cell in range(CELL_COUNT)
                        for probability in self._softmax(
                            logits[cell * len(DIGITS) : (cell + 1) * len(DIGITS)]
                        )
                    ]
                    target = tuple(
                        tuple(
                            1.0 if digit == value else 0.0
                            for digit in DIGITS
                        )
                        for value in example.solution
                    )
                    total_loss += self._apply_gradient(
                        cache, probabilities, target
                    )
                    updates += 1
                    # Copying the hidden state intentionally stops gradients
                    # from flowing through earlier sampled times.
                    detached_state = tuple(hidden) if self.config.recurrent else None
            history.append(total_loss / updates)
        return history
