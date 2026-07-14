"""Greedy test selection: argmax I(T; H) / duration(T)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.algorithm.information_gain import compute_information_gain_fast
from src.failure_model.hypotheses import Hypothesis
from src.test_model.catalog import TestVariant
from src.test_model.observation_matrix import ObservationModel


@dataclass(frozen=True)
class ScoredTest:
    variant: TestVariant
    information_gain: float
    score: float  # IG / duration (or IG if unweighted)


def _component_index(hypotheses: Sequence[Hypothesis]) -> dict[str, int]:
    return {h.component_id: i for i, h in enumerate(hypotheses)}


def fail_indices_for_test(
    test_id: str,
    test_params: dict[str, str],
    observation_model: ObservationModel,
    component_to_index: dict[str, int],
) -> np.ndarray:
    """Hypothesis indices that predict FAIL for this test."""
    exercised = observation_model.exercised(test_id, test_params)
    indices = [
        component_to_index[cid]
        for cid in exercised
        if cid in component_to_index
    ]
    return np.asarray(indices, dtype=int)


def score_test(
    variant: TestVariant,
    posterior: np.ndarray,
    observation_model: ObservationModel,
    component_to_index: dict[str, int],
    *,
    weight_by_duration: bool = True,
) -> ScoredTest:
    fail_idx = fail_indices_for_test(
        variant.test_id,
        variant.params,
        observation_model,
        component_to_index,
    )
    ig = compute_information_gain_fast(posterior, fail_idx)
    if weight_by_duration:
        duration = max(float(variant.duration_seconds), 1e-9)
        score = ig / duration
    else:
        score = ig
    return ScoredTest(variant=variant, information_gain=ig, score=score)


def select_next_test(
    posterior: np.ndarray,
    hypotheses: Sequence[Hypothesis],
    available_tests: Sequence[TestVariant],
    observation_model: ObservationModel,
    *,
    weight_by_duration: bool = True,
    exclude_keys: set[str] | None = None,
) -> ScoredTest:
    """Select the test with highest I(T; H) / duration (greedy).

    Returns a ``ScoredTest``. Raises ``ValueError`` if no candidates remain.
    """
    if not available_tests:
        raise ValueError("No available tests to select from")

    component_to_index = _component_index(hypotheses)
    excluded = exclude_keys or set()

    best: ScoredTest | None = None
    for variant in available_tests:
        if variant.key in excluded:
            continue
        scored = score_test(
            variant,
            posterior,
            observation_model,
            component_to_index,
            weight_by_duration=weight_by_duration,
        )
        if best is None or scored.score > best.score:
            best = scored
        elif (
            best is not None
            and scored.score == best.score
            and scored.variant.key < best.variant.key
        ):
            # Deterministic tie-break for reproducibility
            best = scored

    if best is None:
        raise ValueError("All available tests were excluded")
    return best


def select_next_test_params(
    posterior: np.ndarray,
    hypotheses: Sequence[Hypothesis],
    available_tests: Sequence[TestVariant],
    observation_model: ObservationModel,
    *,
    weight_by_duration: bool = True,
    exclude_keys: set[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Guide-compatible API: return ``(test_id, params)`` for the best test."""
    scored = select_next_test(
        posterior,
        hypotheses,
        available_tests,
        observation_model,
        weight_by_duration=weight_by_duration,
        exclude_keys=exclude_keys,
    )
    return scored.variant.test_id, dict(scored.variant.params)
