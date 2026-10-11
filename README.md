# Thinking with Looped Flows: CPU inference scaling demo

This repository is a small, offline experiment that makes one idea measurable:
**the same denoiser can spend more inference compute by taking more probability-flow
solver steps.** It completes deterministic 4x4 Sudoku puzzles with a task-specific
categorical denoiser, then compares a recurrent-state model with a separately
configured no-recurrent-state baseline at 4, 8, 16, and 32 Euler steps.

The project intentionally uses only the Python standard library at runtime. The
model is a hand-sized multilayer perceptron trained from scratch with seeded
stochastic-gradient updates. It runs on the CPU, keeps parameters and flow states
in float32, and never downloads a dataset, model checkpoint, or remote artifact.

## Setup and exact commands

Use Python 3.10 or newer. The runtime package has no third-party dependencies;
`pytest` is the only development-time tool.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --no-build-isolation --no-deps -e .
python3 -m pytest
python3 scripts/run_experiment.py \
  --config configs/smoke.json \
  --output results
```

The final command writes the complete run to `results/`. It is safe to run from
the repository root without installation because the runner adds `src/` to its
local import path. If `pytest` is not already supplied by the development
environment, install the approved test package through that environment; the
experiment itself remains dependency-free and offline.

## What inference-time scaling means here

Each blank Sudoku cell starts as a uniform categorical distribution. At time
`t = 0`, the clue cells are one-hot and blank cells are uncertain. The model
predicts a categorical distribution for every cell. Euler probability-flow
integration moves the current distribution toward that prediction:

```text
state[t + dt] = state[t] + dt * (denoiser(state[t], t) - state[t])
```

The interval `[0, 1]` is divided into 4, 8, 16, or 32 equal intervals. A larger
step count therefore makes more model calls and usually costs more wall-clock
time without changing the trained weights. The experiment reports exact-board
accuracy, cell accuracy, total model calls, and measured wall-clock latency.
The recurrent model carries a hidden state between inference calls. The baseline
has the same task, data, optimizer, and solver, but no recurrent state.

Training samples ordered random times from `[0, 1]`. After each sampled time, the
recurrent hidden state is copied before the next time sample, which is the
explicit detached-state approximation used by this small implementation.

## Paper-to-code mapping

The experiment is a **conceptual implementation of the inference pattern** in
[*Thinking with Looped Flows*, arXiv:2609.11801](https://arxiv.org/abs/2609.11801),
not a reproduction of the paper's results:

| Paper idea | This repository |
| --- | --- |
| A learned denoising/vector-field step is repeatedly reused at inference | `CategoricalDenoiser.predict` and `euler_probability_flow` |
| Continuous-time state evolution | `interpolant_state`, `interpolant_velocity`, and the Euler update |
| Stateful looped computation | `recurrent_state` in the model and solver |
| Inference-time compute scaling | `inference_steps` in `configs/smoke.json` |
| Compare against a non-looped/state-free control | `ModelConfig(recurrent=False)` baseline |

There is no claim of full reproduction: this demo does not use the paper's model,
data, tokenizer, training scale, objective details, hardware, or benchmark
protocol. It is designed to expose the control flow and metric trade-off in a
small, inspectable program.

## Why Sudoku and not Qwen?

Qwen would turn this into a large-language-model integration exercise: it would
require a heavyweight checkpoint, tokenizer behavior, external downloads, and
hardware-dependent inference. Those details would obscure the question this demo
is meant to isolate. A 4x4 Sudoku completion task gives us:

* an exact, locally generated target for every example;
* a categorical state whose normalization can be tested directly;
* a clear exact-board metric in addition to per-cell accuracy;
* a model small enough to train from scratch and inspect in one sitting; and
* a fully offline, deterministic seed path suitable for CI and code review.

This is a teaching and systems-measurement choice, not a claim that Sudoku is a
proxy for language-model quality.

## Output artifacts

After a successful run, `results/` contains:

* `metrics.json` — one aggregate row per model and step count;
* `metrics.csv` — the same rows for spreadsheets and quick plotting;
* `summary.json` — dataset sizes, sampled training times, final losses, and best
  step count for each model;
* `metadata.json` — UTC timestamp, CPU/platform, Python/package versions, seed,
  exact command, numeric dtype, and git revision; and
* `inference_scaling.svg` — an accuracy-versus-Euler-steps chart.

The runner computes training and all evaluations before publishing artifacts.
It stages every file in a temporary directory and only replaces the named output
files after the complete run has succeeded.

`docs/architecture.svg` is the static architecture overview. The smoke config is
deliberately small and can be edited to explore other deterministic dataset sizes,
hidden sizes, clue counts, or step schedules.

## Senior engineer to intern walkthrough

1. **Start with the contract.** `dataset.py` defines a valid 4x4 Sudoku, generates
   transformed solutions and clue masks from a seed, and validates every example.
2. **Look at the state.** `initial_state` represents a clue as one-hot and a blank
   as uniform. `flow.py` proves the linear interpolant endpoints and then performs
   the solver update while preserving clue cells and categorical normalization.
3. **Read one model call.** `model.py` flattens the current state, visible clues,
   Sudoku candidate masks, position features, and time into a small CPU MLP. The
   recurrent variant also reads the previous hidden state; the baseline does not.
4. **Follow training.** `fit` walks the sorted sampled times for each example,
   applies cross-entropy SGD, and copies the hidden state between times so it is
   detached rather than backpropagated through the whole sequence.
5. **Follow inference.** `euler_probability_flow` calls the same denoiser exactly
   `steps` times. The returned call count is therefore an auditable measure of
   inference compute, not an estimate.
6. **Read the evaluation boundary.** `experiment.py` trains two independent
   configurations and aggregates exact accuracy, cell accuracy, calls, and latency.
7. **Run and inspect.** The script collects metadata and writes JSON, CSV, and SVG
   only after every model/step combination has completed.

## Limitations

This is intentionally not a production inference stack or a paper reproduction.
The dataset is tiny and synthetic, the network is small, latency is measured in a
single local Python process, and the task-specific candidate features inject
Sudoku structure that a general language model would not receive. The baseline
is separately initialized and trained rather than parameter-matched to a large
architecture. Accuracy and latency are therefore useful for understanding the
loop/solver mechanics, not for comparing model families or hardware.
