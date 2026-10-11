"""Small, offline CPU experiments for looped probability flows."""

from .data import PuzzleDataset, make_dataset, is_valid_solution
from .flow import integrate_probability_flow
from .model import DenoiserConfig, StatefulDenoiser

__all__ = [
    "DenoiserConfig",
    "PuzzleDataset",
    "StatefulDenoiser",
    "integrate_probability_flow",
    "is_valid_solution",
    "make_dataset",
]
