"""Deterministic 4x4 Sudoku completion data and state encodings."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .numeric import float32

GRID_SIZE = 4
BOX_SIZE = 2
CELL_COUNT = GRID_SIZE * GRID_SIZE
DIGITS = tuple(range(GRID_SIZE))
BLANK = -1

Puzzle = tuple[int, ...]


@dataclass(frozen=True)
class SudokuExample:
    """A partially observed Sudoku board and its exact completion."""

    puzzle: Puzzle
    solution: Puzzle

    def __post_init__(self) -> None:
        if len(self.puzzle) != CELL_COUNT or len(self.solution) != CELL_COUNT:
            raise ValueError(f"Sudoku boards must contain {CELL_COUNT} cells")
        if not is_valid_puzzle(self.puzzle):
            raise ValueError("puzzle contains invalid or duplicate clues")
        if not is_valid_solution(self.solution):
            raise ValueError("solution is not a valid 4x4 Sudoku")
        if any(
            clue != BLANK and clue != solved
            for clue, solved in zip(self.puzzle, self.solution)
        ):
            raise ValueError("puzzle clue does not match solution")


def _base_solution() -> list[list[int]]:
    return [
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [3, 2, 1, 0],
    ]


def _permuted_solution(rng: random.Random) -> Puzzle:
    """Generate a valid board using only Sudoku-preserving permutations."""

    board = _base_solution()
    digits = list(DIGITS)
    rng.shuffle(digits)

    band_order = [0, 1]
    rng.shuffle(band_order)
    row_order: list[int] = []
    for band in band_order:
        rows = [band * BOX_SIZE + offset for offset in range(BOX_SIZE)]
        rng.shuffle(rows)
        row_order.extend(rows)

    stack_order = [0, 1]
    rng.shuffle(stack_order)
    col_order: list[int] = []
    for stack in stack_order:
        columns = [stack * BOX_SIZE + offset for offset in range(BOX_SIZE)]
        rng.shuffle(columns)
        col_order.extend(columns)

    return tuple(
        digits[board[row][column]]
        for row in row_order
        for column in col_order
    )


def _unit_indices() -> Iterable[tuple[int, ...]]:
    for row in range(GRID_SIZE):
        yield tuple(row * GRID_SIZE + column for column in range(GRID_SIZE))
    for column in range(GRID_SIZE):
        yield tuple(row * GRID_SIZE + column for row in range(GRID_SIZE))
    for box_row in range(0, GRID_SIZE, BOX_SIZE):
        for box_column in range(0, GRID_SIZE, BOX_SIZE):
            yield tuple(
                (box_row + row_offset) * GRID_SIZE
                + box_column
                + column_offset
                for row_offset in range(BOX_SIZE)
                for column_offset in range(BOX_SIZE)
            )


def is_valid_solution(board: Sequence[int]) -> bool:
    """Return whether ``board`` is a complete valid 4x4 Sudoku."""

    if len(board) != CELL_COUNT or any(value not in DIGITS for value in board):
        return False
    expected = set(DIGITS)
    return all({board[index] for index in unit} == expected for unit in _unit_indices())


def is_valid_puzzle(board: Sequence[int]) -> bool:
    """Return whether clues are in range and do not conflict in any unit."""

    if len(board) != CELL_COUNT or any(
        value != BLANK and value not in DIGITS for value in board
    ):
        return False
    return all(
        len([board[index] for index in unit if board[index] != BLANK])
        == len({board[index] for index in unit if board[index] != BLANK})
        for unit in _unit_indices()
    )


def is_valid_completion(puzzle: Sequence[int], completion: Sequence[int]) -> bool:
    """Return whether a completion is valid and preserves every clue."""

    return (
        len(puzzle) == CELL_COUNT
        and len(completion) == CELL_COUNT
        and is_valid_solution(completion)
        and all(
            clue == BLANK or clue == value
            for clue, value in zip(puzzle, completion)
        )
    )


def generate_example(
    rng: random.Random,
    min_clues: int = 8,
    max_clues: int = 10,
) -> SudokuExample:
    """Generate one deterministic puzzle from a transformed base solution."""

    if not 0 <= min_clues <= max_clues <= CELL_COUNT:
        raise ValueError("clue bounds must satisfy 0 <= min <= max <= 16")
    solution = _permuted_solution(rng)
    clue_count = rng.randint(min_clues, max_clues)
    clue_indices = set(rng.sample(range(CELL_COUNT), clue_count))
    puzzle = tuple(
        solution[index] if index in clue_indices else BLANK
        for index in range(CELL_COUNT)
    )
    return SudokuExample(puzzle=puzzle, solution=solution)


def make_dataset(
    train_size: int,
    test_size: int,
    seed: int,
    min_clues: int = 8,
    max_clues: int = 10,
) -> tuple[list[SudokuExample], list[SudokuExample]]:
    """Create reproducible train and test splits without external data."""

    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be positive")
    rng = random.Random(seed)
    train = [
        generate_example(rng, min_clues=min_clues, max_clues=max_clues)
        for _ in range(train_size)
    ]
    test = [
        generate_example(rng, min_clues=min_clues, max_clues=max_clues)
        for _ in range(test_size)
    ]
    return train, test


def initial_state(puzzle: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    """Encode clues as one-hot distributions and blanks as uniform."""

    if not is_valid_puzzle(puzzle):
        raise ValueError("cannot encode an invalid puzzle")
    uniform = 1.0 / GRID_SIZE
    rows: list[tuple[float, ...]] = []
    for value in puzzle:
        rows.append(
            tuple(
                float32(1.0 if digit == value else 0.0)
                for digit in DIGITS
            )
            if value != BLANK
            else (float32(uniform),) * GRID_SIZE
        )
    return tuple(rows)


def candidate_mask(puzzle: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    """Encode values allowed by the currently visible Sudoku clues."""

    if not is_valid_puzzle(puzzle):
        raise ValueError("cannot compute candidates for an invalid puzzle")
    masks: list[tuple[float, ...]] = []
    units = tuple(_unit_indices())
    for index, value in enumerate(puzzle):
        if value != BLANK:
            masks.append(tuple(1.0 if digit == value else 0.0 for digit in DIGITS))
            continue
        forbidden = {
            puzzle[other]
            for unit in units
            if index in unit
            for other in unit
            if puzzle[other] != BLANK
        }
        allowed = tuple(0.0 if digit in forbidden else 1.0 for digit in DIGITS)
        masks.append(allowed if any(allowed) else (1.0,) * GRID_SIZE)
    return tuple(masks)


def one_hot_solution(solution: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    if not is_valid_solution(solution):
        raise ValueError("cannot encode an invalid solution")
    return tuple(
        tuple(1.0 if digit == value else 0.0 for digit in DIGITS)
        for value in solution
    )


def render_board(board: Sequence[int], blank: str = ".") -> str:
    """Render a flat board as a compact human-readable grid."""

    if len(board) != CELL_COUNT:
        raise ValueError(f"board must contain {CELL_COUNT} cells")
    lines = []
    for row in range(GRID_SIZE):
        values = board[row * GRID_SIZE : (row + 1) * GRID_SIZE]
        lines.append(" ".join(blank if value == BLANK else str(value + 1) for value in values))
    return "\n".join(lines)
