"""Mutual information I(T; H) for candidate diagnostic tests."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.failure_model.hypotheses import Hypothesis

OutcomeFn = Callable[[str, dict[str, str], Hypothesis], float]


def shannon_entropy(probs: np.ndarray, *, eps: float = 1e-15) -> float:
    """Shannon entropy H(p) in nats for a discrete distribution."""
    p = np.asarray(probs, dtype=float)
    p = p[p > eps]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _fail_mask_from_outcomes(outcomes: np.ndarray) -> np.ndarray:
    """Treat outcome == 0.0 as FAIL; anything else as non-fail for v1 binary MI."""
    return np.isclose(outcomes, 0.0)


def compute_information_gain_from_outcomes(
    posterior: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """I(T; H) given per-hypothesis expected outcomes under current posterior.

    Uses the deterministic PASS/FAIL partition induced by ``outcomes``.
    """
    posterior = np.asarray(posterior, dtype=float)
    if posterior.shape != outcomes.shape:
        raise ValueError("posterior and outcomes must have the same shape")
    total = posterior.sum()
    if total <= 0:
        return 0.0
    posterior = posterior / total

    prior_entropy = shannon_entropy(posterior)
    fail_mask = _fail_mask_from_outcomes(outcomes)
    p_fail = float(posterior[fail_mask].sum())
    p_pass = 1.0 - p_fail

    conditional = 0.0
    if p_fail > 1e-15:
        conditional += p_fail * shannon_entropy(posterior[fail_mask])
    if p_pass > 1e-15:
        conditional += p_pass * shannon_entropy(posterior[~fail_mask])

    ig = prior_entropy - conditional
    return float(max(0.0, ig))


def compute_information_gain(
    test_id: str,
    test_params: dict[str, str],
    posterior: np.ndarray,
    hypotheses: Sequence[Hypothesis],
    observation_matrix_fn: OutcomeFn,
) -> float:
    """Compute I(T; H) for a candidate test.

    ``observation_matrix_fn(test_id, test_params, hypothesis)`` should return the
    expected outcome in {PASS=1.0, FAIL=0.0, DEGRADED=0.5}.
    """
    outcomes = np.array(
        [
            observation_matrix_fn(test_id, test_params, hyp)
            for hyp in hypotheses
        ],
        dtype=float,
    )
    return compute_information_gain_from_outcomes(posterior, outcomes)


def compute_information_gain_fast(
    posterior: np.ndarray,
    fail_indices: np.ndarray,
) -> float:
    """I(T; H) when FAIL hypotheses are known by index (single-fault model)."""
    posterior = np.asarray(posterior, dtype=float)
    total = posterior.sum()
    if total <= 0:
        return 0.0
    posterior = posterior / total

    prior_entropy = shannon_entropy(posterior)
    if fail_indices.size == 0:
        return 0.0

    p_fail = float(posterior[fail_indices].sum())
    p_pass = 1.0 - p_fail

    conditional = 0.0
    if p_fail > 1e-15:
        conditional += p_fail * shannon_entropy(posterior[fail_indices])
    if p_pass > 1e-15:
        pass_mask = np.ones(posterior.shape[0], dtype=bool)
        pass_mask[fail_indices] = False
        conditional += p_pass * shannon_entropy(posterior[pass_mask])

    return float(max(0.0, prior_entropy - conditional))
