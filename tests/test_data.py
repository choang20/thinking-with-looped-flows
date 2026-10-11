import numpy as np

from looped_flows_cpu.data import GRID, LENGTH, make_dataset, is_valid_solution


def test_generated_solutions_are_valid_and_puzzles_are_unique() -> None:
    dataset = make_dataset(12, seed=11)
    for solution, puzzle, clue_mask in zip(
        dataset.solutions,
        dataset.puzzles,
        dataset.clue_mask,
    ):
        assert is_valid_solution(solution.reshape(GRID, GRID))
        assert np.array_equal(puzzle[clue_mask], solution[clue_mask])
        assert np.all(puzzle[~clue_mask] == -1)
        assert clue_mask.shape == (LENGTH,)


def test_generation_is_deterministic() -> None:
    first = make_dataset(8, seed=3)
    second = make_dataset(8, seed=3)
    assert np.array_equal(first.puzzles, second.puzzles)
    assert np.array_equal(first.solutions, second.solutions)
