"""Synthetic 4x4 Sudoku data used by the local experiment.

The dataset is deliberately small and generated locally.  Every puzzle has a
unique solution, so exact-match accuracy is a meaningful metric rather than a
proxy based on a potentially ambiguous completion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID = 4
VOCAB = 4
LENGTH = GRID * GRID

_BASE_SOLUTION = np.array(
    [
        [0, 1, 2, 3],
        [2, 3, 0, 1],
        [1, 0, 3, 2],
        [3, 2, 1, 0],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class PuzzleDataset:
    """A batch of fixed-length categorical reasoning problems."""

    puzzles: np.ndarray
    solutions: np.ndarray
    clue_mask: np.ndarray

    @property
    def size(self) -> int:
        return int(self.solutions.shape[0])

    @property
    def target_one_hot(self) -> np.ndarray:
        return np.eye(VOCAB, dtype=np.float32)[self.solutions]

    @property
    def condition_one_hot(self) -> np.ndarray:
        condition = np.zeros_like(self.target_one_hot)
        rows = np.arange(self.size)[:, None]
        positions = np.arange(LENGTH)[None, :]
        condition[rows, positions, self.puzzles] = (
            self.clue_mask.astype(np.float32)
        )
        return condition

    @property
    def unknown_mask(self) -> np.ndarray:
        return (~self.clue_mask).astype(np.float32)


def is_valid_solution(grid: np.ndarray) -> bool:
    """Return whether a 4x4 grid obeys row, column, and 2x2-box constraints."""

    grid = np.asarray(grid)
    if grid.shape != (GRID, GRID) or set(np.unique(grid)) != set(range(VOCAB)):
        return False
    expected = set(range(VOCAB))
    for row in grid:
        if set(row.tolist()) != expected:
            return False
    for column in grid.T:
        if set(column.tolist()) != expected:
            return False
    for row in range(0, GRID, 2):
        for column in range(0, GRID, 2):
            if set(grid[row : row + 2, column : column + 2].ravel().tolist()) != expected:
                return False
    return True


def _make_solution(rng: np.random.Generator) -> np.ndarray:
    """Create a varied valid 4x4 Sudoku using symmetry-preserving transforms."""

    row_bands = rng.permutation(2)
    rows = np.concatenate(
        [2 * band + rng.permutation(2) for band in row_bands]
    )
    column_stacks = rng.permutation(2)
    columns = np.concatenate(
        [2 * stack + rng.permutation(2) for stack in column_stacks]
    )
    digits = rng.permutation(VOCAB)
    return digits[_BASE_SOLUTION[np.ix_(rows, columns)]]


def _count_solutions(puzzle: np.ndarray, limit: int = 2) -> int:
    """Count solutions, stopping at ``limit`` because uniqueness is all we need."""

    grid = np.asarray(puzzle, dtype=np.int64).copy()
    count = 0

    def search() -> None:
        nonlocal count
        if count >= limit:
            return
        best_cell: tuple[int, int] | None = None
        best_candidates: list[int] | None = None
        for row in range(GRID):
            for column in range(GRID):
                if grid[row, column] != -1:
                    continue
                used = set(grid[row, :].tolist())
                used.update(grid[:, column].tolist())
                used.update(
                    grid[
                        row // 2 * 2 : row // 2 * 2 + 2,
                        column // 2 * 2 : column // 2 * 2 + 2,
                    ].ravel().tolist()
                )
                candidates = [
                    digit for digit in range(VOCAB) if digit not in used
                ]
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_cell = (row, column)
                    best_candidates = candidates
        if best_cell is None:
            count += 1
            return
        row, column = best_cell
        for digit in best_candidates or []:
            grid[row, column] = digit
            search()
            grid[row, column] = -1

    search()
    return count


def _make_unique_puzzle(
    solution: np.ndarray,
    rng: np.random.Generator,
    clue_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(LENGTH)
    for _ in range(100):
        clue_positions = rng.choice(positions, size=clue_count, replace=False)
        puzzle = np.full((GRID, GRID), -1, dtype=np.int64)
        flat = puzzle.reshape(-1)
        flat[clue_positions] = solution.reshape(-1)[clue_positions]
        if _count_solutions(puzzle) == 1:
            return puzzle, np.isin(positions, clue_positions).reshape(GRID, GRID)
    # The 4x4 task is easy to make unique; this is a defensive fallback that
    # makes data generation explicit rather than silently returning ambiguity.
    for clue_count_fallback in range(clue_count + 1, LENGTH + 1):
        clue_positions = rng.choice(
            positions, size=clue_count_fallback, replace=False
        )
        puzzle = np.full((GRID, GRID), -1, dtype=np.int64)
        flat = puzzle.reshape(-1)
        flat[clue_positions] = solution.reshape(-1)[clue_positions]
        if _count_solutions(puzzle) == 1:
            return puzzle, np.isin(positions, clue_positions).reshape(GRID, GRID)
    raise RuntimeError("Could not construct a unique 4x4 Sudoku puzzle")


def make_dataset(
    size: int,
    *,
    seed: int,
    clue_count: int = 8,
) -> PuzzleDataset:
    """Generate ``size`` unique 4x4 Sudoku puzzles deterministically."""

    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 < clue_count < LENGTH:
        raise ValueError(f"clue_count must be between 1 and {LENGTH - 1}")

    rng = np.random.default_rng(seed)
    puzzles = np.empty((size, GRID, GRID), dtype=np.int64)
    solutions = np.empty_like(puzzles)
    clue_masks = np.empty_like(puzzles, dtype=bool)
    for index in range(size):
        solution = _make_solution(rng)
        puzzle, clue_mask = _make_unique_puzzle(solution, rng, clue_count)
        puzzles[index] = puzzle
        solutions[index] = solution
        clue_masks[index] = clue_mask

    return PuzzleDataset(
        puzzles=puzzles.reshape(size, LENGTH),
        solutions=solutions.reshape(size, LENGTH),
        clue_mask=clue_masks.reshape(size, LENGTH),
    )
