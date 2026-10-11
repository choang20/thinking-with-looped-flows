from looped_flows_cpu.dataset import make_dataset
from looped_flows_cpu.flow import interpolant_state, interpolant_velocity


def test_interpolant_endpoints_and_exact_velocity():
    train, _ = make_dataset(1, 1, seed=12)
    example = train[0]
    start = interpolant_state(example.puzzle, example.solution, 0.0)
    midpoint = interpolant_state(example.puzzle, example.solution, 0.5)
    end = interpolant_state(example.puzzle, example.solution, 1.0)
    velocity = interpolant_velocity(example.puzzle, example.solution, 0.37)

    for cell in range(16):
        for digit in range(4):
            assert end[cell][digit] == float(
                digit == example.solution[cell]
            )
            expected_midpoint = 0.5 * (start[cell][digit] + end[cell][digit])
            assert abs(midpoint[cell][digit] - expected_midpoint) < 1e-9
            assert abs(
                velocity[cell][digit]
                - (end[cell][digit] - start[cell][digit])
            ) < 1e-9
            assert abs(sum(midpoint[cell]) - 1.0) < 1e-9
