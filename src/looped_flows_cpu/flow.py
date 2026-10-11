"""Interpolants and Euler probability-flow inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .dataset import (
    BLANK,
    CELL_COUNT,
    DIGITS,
    GRID_SIZE,
    initial_state,
    one_hot_solution,
)
from .numeric import float32

if TYPE_CHECKING:
    from .model import CategoricalDenoiser


State = tuple[tuple[float, ...], ...]


def _validate_time(time: float) -> None:
    if not 0.0 <= time <= 1.0:
        raise ValueError("time must be in [0, 1]")


def interpolant_state(
    puzzle: Sequence[int],
    solution: Sequence[int],
    time: float,
) -> State:
    """Linearly interpolate the categorical state from clues to the solution."""

    _validate_time(time)
    start = initial_state(puzzle)
    target = one_hot_solution(solution)
    return tuple(
        tuple(
            float32(
                (1.0 - time) * start[cell][digit]
                + time * target[cell][digit]
            )
            for digit in DIGITS
        )
        for cell in range(CELL_COUNT)
    )


def interpolant_velocity(
    puzzle: Sequence[int],
    solution: Sequence[int],
    time: float,
) -> State:
    """Return the exact time derivative of :func:`interpolant_state`."""

    _validate_time(time)
    start = initial_state(puzzle)
    target = one_hot_solution(solution)
    return tuple(
        tuple(
            float32(target[cell][digit] - start[cell][digit])
            for digit in DIGITS
        )
        for cell in range(CELL_COUNT)
    )


def _renormalize(values: Sequence[float]) -> tuple[float, ...]:
    clipped = [max(0.0, value) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        return (1.0 / GRID_SIZE,) * GRID_SIZE
    return tuple(float32(value / total) for value in clipped)


def decode_state(state: State) -> tuple[int, ...]:
    return tuple(
        max(DIGITS, key=lambda digit: state[cell][digit])
        for cell in range(CELL_COUNT)
    )


@dataclass(frozen=True)
class InferenceOutput:
    prediction: tuple[int, ...]
    final_state: State
    model_calls: int


def euler_probability_flow(
    model: "CategoricalDenoiser",
    puzzle: Sequence[int],
    steps: int,
) -> InferenceOutput:
    """Integrate the learned categorical vector field with Euler steps."""

    if steps < 1:
        raise ValueError("steps must be positive")
    state = initial_state(puzzle)
    recurrent_state: tuple[float, ...] | None = None
    step_size = 1.0 / steps

    for step in range(steps):
        time = step / steps
        probabilities, recurrent_state = model.predict(
            state=state,
            puzzle=puzzle,
            time=time,
            recurrent_state=recurrent_state,
        )
        next_rows: list[tuple[float, ...]] = []
        for cell, clue in enumerate(puzzle):
            if clue != BLANK:
                next_rows.append(
                    tuple(1.0 if digit == clue else 0.0 for digit in DIGITS)
                )
                continue
            next_rows.append(
                _renormalize(
                    state[cell][digit]
                    + step_size * (probabilities[cell][digit] - state[cell][digit])
                    for digit in DIGITS
                )
            )
        state = tuple(next_rows)

    return InferenceOutput(
        prediction=decode_state(state),
        final_state=state,
        model_calls=steps,
    )
