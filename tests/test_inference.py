from looped_flows_cpu.dataset import make_dataset
from looped_flows_cpu.flow import euler_probability_flow
from looped_flows_cpu.model import CategoricalDenoiser, ModelConfig


def test_inference_is_deterministic_and_counts_euler_steps():
    train, _ = make_dataset(2, 1, seed=19)
    model = CategoricalDenoiser(
        ModelConfig(hidden_size=8, epochs=2, learning_rate=0.05),
        seed=23,
    )
    model.fit(train, (0.1, 0.5, 0.9))
    puzzle = train[0].puzzle

    four_a = euler_probability_flow(model, puzzle, steps=4)
    four_b = euler_probability_flow(model, puzzle, steps=4)
    eight = euler_probability_flow(model, puzzle, steps=8)

    assert four_a == four_b
    assert four_a.model_calls == 4
    assert eight.model_calls == 8
    assert len(four_a.prediction) == 16
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in eight.final_state)
