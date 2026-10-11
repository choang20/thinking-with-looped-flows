from looped_flows_cpu.dataset import (
    BLANK,
    CELL_COUNT,
    candidate_mask,
    initial_state,
    is_valid_completion,
    is_valid_puzzle,
    is_valid_solution,
    make_dataset,
)


def test_generated_examples_are_valid_and_deterministic():
    first_train, first_test = make_dataset(8, 4, seed=41)
    second_train, second_test = make_dataset(8, 4, seed=41)

    assert first_train == second_train
    assert first_test == second_test
    for example in first_train + first_test:
        assert is_valid_puzzle(example.puzzle)
        assert is_valid_solution(example.solution)
        assert is_valid_completion(example.puzzle, example.solution)
        assert len(example.puzzle) == CELL_COUNT


def test_state_and_candidates_have_categorical_shapes():
    train, _ = make_dataset(1, 1, seed=7)
    puzzle = train[0].puzzle
    state = initial_state(puzzle)
    candidates = candidate_mask(puzzle)

    assert len(state) == CELL_COUNT
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in state)
    assert all(sum(row) >= 1.0 for row in candidates)
    assert any(value == BLANK for value in puzzle)
