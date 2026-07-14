"""InfoSlice core: information gain, greedy selection, posterior update."""

from src.algorithm.greedy_selector import ScoredTest, select_next_test, select_next_test_params
from src.algorithm.information_gain import (
    compute_information_gain,
    compute_information_gain_fast,
    shannon_entropy,
)
from src.algorithm.posterior_update import is_diagnosed, map_hypothesis_index, update_posterior
from src.algorithm.slice_isolator import Slice, build_slices, evaluate_slices

__all__ = [
    "ScoredTest",
    "select_next_test",
    "select_next_test_params",
    "compute_information_gain",
    "compute_information_gain_fast",
    "shannon_entropy",
    "is_diagnosed",
    "map_hypothesis_index",
    "update_posterior",
    "Slice",
    "build_slices",
    "evaluate_slices",
]
