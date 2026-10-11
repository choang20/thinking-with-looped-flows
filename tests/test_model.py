import numpy as np

from looped_flows_cpu.data import make_dataset
from looped_flows_cpu.model import DenoiserConfig, StatefulDenoiser


def test_denoiser_shapes_and_probabilities() -> None:
    dataset = make_dataset(3, seed=5)
    model = StatefulDenoiser(DenoiserConfig(hidden=16), seed=2)
    state = model.zero_state(dataset.size)
    probabilities, next_state, cache = model.forward(
        dataset.target_one_hot,
        dataset.condition_one_hot,
        dataset.clue_mask,
        np.full(dataset.size, 0.25, dtype=np.float32),
        state,
    )
    assert probabilities.shape == (3, 16, 4)
    assert next_state.shape == (3, 16)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    loss, gradients = model.backward(
        cache,
        dataset.target_one_hot,
        dataset.unknown_mask,
    )
    assert np.isfinite(loss)
    assert all(value.shape == model.parameters[name].shape for name, value in gradients.items())
