# Experiment summary

This result was generated locally from synthetic 4x4 Sudoku data.
The same trained weights are evaluated with different probability-flow
integration grids; increasing `flow_steps` is the inference-time
scaling axis.

| Variant | Steps | Exact accuracy | Model calls | Mean call latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| looped-flow | 2 | 19.2% | 480 | 0.003 |
| looped-flow | 4 | 35.0% | 960 | 0.003 |
| looped-flow | 8 | 44.6% | 1920 | 0.002 |
| looped-flow | 16 | 50.8% | 3840 | 0.002 |
| looped-flow | 32 | 55.8% | 7680 | 0.002 |
| no-state-flow | 2 | 17.9% | 480 | 0.003 |
| no-state-flow | 4 | 33.8% | 960 | 0.002 |
| no-state-flow | 8 | 43.8% | 1920 | 0.002 |
| no-state-flow | 16 | 46.7% | 3840 | 0.002 |
| no-state-flow | 32 | 51.2% | 7680 | 0.002 |

Interpretation: accuracy is the primary measure. Latency is
machine-dependent and is included to show the cost of spending
more inference compute.
