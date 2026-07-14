"""Bayesian posterior update after observing a test outcome."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.failure_model.hypotheses import Hypothesis

OutcomeFn = Callable[[str, dict[str, str], Hypothesis], float]


def update_posterior(
    posterior: np.ndarray,
    test_id: str,
    test_params: dict[str, str],
    observed_outcome: float,
    hypotheses: Sequence[Hypothesis],
    observation_matrix_fn: OutcomeFn,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """Bayesian update: P(h|outcome) ∝ P(outcome|h) * P(h).

    Under the deterministic v1 observation model, P(outcome|h) is 1 if the
    expected outcome matches ``observed_outcome``, else 0.
    """
    posterior = np.asarray(posterior, dtype=float).copy()
    if len(posterior) != len(hypotheses):
        raise ValueError("posterior length must match hypotheses")

    likelihood = np.array(
        [
            1.0
            if abs(observation_matrix_fn(test_id, test_params, hyp) - observed_outcome)
            <= 1e-9
            else 0.0
            for hyp in hypotheses
        ],
        dtype=float,
    )

    updated = posterior * likelihood
    total = updated.sum()
    if total <= eps:
        # Observation impossible under current support; return normalized prior mass
        # on hypotheses that were already zeroed... keep zeros and warn via zerosum.
        return updated

    return updated / total


def update_posterior_from_fail_indices(
    posterior: np.ndarray,
    fail_indices: np.ndarray,
    observed_outcome: float,
    *,
    eps: float = 1e-15,
) -> np.ndarray:
    """Fast update when FAIL set is known by hypothesis index."""
    posterior = np.asarray(posterior, dtype=float).copy()
    fail_mask = np.zeros(posterior.shape[0], dtype=bool)
    fail_mask[fail_indices] = True

    if abs(observed_outcome - 0.0) <= 1e-9:
        # Observed FAIL → keep only hypotheses that predict FAIL
        updated = np.where(fail_mask, posterior, 0.0)
    elif abs(observed_outcome - 1.0) <= 1e-9:
        # Observed PASS → keep only hypotheses that predict PASS
        updated = np.where(~fail_mask, posterior, 0.0)
    else:
        # DEGRADED or other: no hard constraint in v1 fast path
        updated = posterior

    total = updated.sum()
    if total <= eps:
        return updated
    return updated / total


def is_diagnosed(
    posterior: np.ndarray,
    *,
    threshold: float = 0.95,
    require_unique_support: bool = True,
) -> bool:
    """True when the fault hypothesis is identified.

    Under the deterministic v1 observation model, ``require_unique_support=True``
    (default) waits until only one hypothesis retains posterior mass. This avoids
    prior skew declaring a high-prior GPU before its incident links are ruled out.
    Set ``require_unique_support=False`` to use the MAP ``threshold`` rule instead.
    """
    posterior = np.asarray(posterior, dtype=float)
    support = np.where(posterior > 1e-15)[0]
    if support.size <= 1:
        return True
    if require_unique_support:
        return False
    return float(posterior.max()) >= threshold


def map_hypothesis_index(posterior: np.ndarray) -> int:
    return int(np.argmax(posterior))
