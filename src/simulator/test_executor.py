"""Simulate test execution by looking up expected outcomes."""

from __future__ import annotations

import numpy as np

from src.failure_model.hypotheses import Hypothesis
from src.test_model.observation_matrix import ObservationModel


def execute_test(
    test_id: str,
    test_params: dict[str, str],
    actual_fault: Hypothesis,
    observation_model: ObservationModel,
    *,
    detection_prob: float = 1.0,
    false_positive_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Simulate running a test given the injected ground-truth fault.

    v1 default is deterministic lookup (detection_prob=1, FPR=0). Optional noise
    flips FAIL→PASS with probability (1 - detection_prob) and PASS→FAIL with
    probability false_positive_rate.
    """
    expected = observation_model.expected_outcome(test_id, test_params, actual_fault)

    if detection_prob >= 1.0 and false_positive_rate <= 0.0:
        return expected

    rng = rng if rng is not None else np.random.default_rng()
    if abs(expected - 0.0) <= 1e-9:
        # True FAIL: report FAIL with detection_prob, else false PASS
        return 0.0 if rng.random() < detection_prob else 1.0
    # True PASS: report FAIL with FPR, else PASS
    return 0.0 if rng.random() < false_positive_rate else 1.0
