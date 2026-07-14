"""Sample a ground-truth fault from the prior distribution."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.failure_model.hypotheses import Hypothesis


def inject_fault(
    hypotheses: Sequence[Hypothesis],
    rng: np.random.Generator | None = None,
) -> Hypothesis:
    """Sample a random fault weighted by hypothesis priors."""
    if not hypotheses:
        raise ValueError("hypotheses must be non-empty")
    rng = rng if rng is not None else np.random.default_rng()
    priors = np.array([h.prior for h in hypotheses], dtype=float)
    total = priors.sum()
    if total <= 0:
        raise ValueError("priors must sum to a positive value")
    priors = priors / total
    index = int(rng.choice(len(hypotheses), p=priors))
    return hypotheses[index]


def inject_fault_by_id(
    hypotheses: Sequence[Hypothesis],
    hypothesis_id: str,
) -> Hypothesis:
    """Return a specific hypothesis (for debugging / unit tests)."""
    for hyp in hypotheses:
        if hyp.id == hypothesis_id:
            return hyp
    raise KeyError(f"Unknown hypothesis id: {hypothesis_id}")
