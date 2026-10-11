"""Offline CPU-only inference-time scaling demo."""

__version__ = "0.1.0"

from .dataset import SudokuExample, make_dataset
from .experiment import ExperimentConfig, ExperimentResult, run_experiment

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "SudokuExample",
    "__version__",
    "make_dataset",
    "run_experiment",
]
