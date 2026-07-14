"""Baseline diagnosis: fixed full-suite re-test (non-adaptive)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.algorithm.greedy_selector import fail_indices_for_test
from src.algorithm.posterior_update import (
    is_diagnosed,
    map_hypothesis_index,
    update_posterior_from_fail_indices,
)
from src.failure_model.hypotheses import Hypothesis
from src.simulator.test_executor import execute_test
from src.test_model.catalog import TestVariant, expand_variants, get_test_type
from src.test_model.observation_matrix import ObservationModel
from src.topology.graph import NVL72Topology


@dataclass
class BaselineResult:
    total_time_seconds: float
    num_tests: int
    correct: bool
    diagnosed_hypothesis_id: str | None
    suite_time_seconds: float
    time_to_diagnosis_seconds: float | None


def build_baseline_suite(topology: NVL72Topology) -> list[TestVariant]:
    """Fixed diagnostic re-test suite approximating current practice.

    Order: full fabric → all intra-tray → all NVSwitch slices → all NVLink checks.
    Nominal total duration ≈ 300 + 18*30 + 18*45 + 72*5 = 2010s (~33.5 min).
    """
    suite: list[TestVariant] = []
    suite.extend(expand_variants(get_test_type("full_fabric_allreduce"), topology))
    suite.extend(expand_variants(get_test_type("intra_tray_nccl"), topology))
    suite.extend(expand_variants(get_test_type("nvswitch_slice"), topology))
    suite.extend(expand_variants(get_test_type("nvlink_error_check"), topology))
    return suite


def run_baseline(
    actual_fault: Hypothesis,
    hypotheses: Sequence[Hypothesis],
    observation_model: ObservationModel,
    suite: Sequence[TestVariant] | None = None,
    *,
    stop_when_diagnosed: bool = False,
    confidence_threshold: float = 0.95,
) -> BaselineResult:
    """Run the baseline suite sequentially.

    By default runs the **entire** suite (non-adaptive full re-test). Set
    ``stop_when_diagnosed=True`` to stop early once the posterior concentrates.
    """
    topology = observation_model.topology
    tests = list(suite) if suite is not None else build_baseline_suite(topology)
    component_to_index = {h.component_id: i for i, h in enumerate(hypotheses)}
    posterior = np.array([h.prior for h in hypotheses], dtype=float)

    total_time = 0.0
    time_to_diagnosis: float | None = None
    num_tests = 0

    for variant in tests:
        outcome = execute_test(
            variant.test_id, variant.params, actual_fault, observation_model
        )
        fail_idx = fail_indices_for_test(
            variant.test_id,
            variant.params,
            observation_model,
            component_to_index,
        )
        posterior = update_posterior_from_fail_indices(posterior, fail_idx, outcome)
        total_time += float(variant.duration_seconds)
        num_tests += 1

        if time_to_diagnosis is None and is_diagnosed(
            posterior, threshold=confidence_threshold
        ):
            time_to_diagnosis = total_time
            if stop_when_diagnosed:
                break

    suite_time = sum(float(v.duration_seconds) for v in tests)
    reported_time = total_time if stop_when_diagnosed else suite_time

    map_idx = map_hypothesis_index(posterior)
    diagnosed_id = hypotheses[map_idx].id if posterior.sum() > 0 else None
    correct = diagnosed_id == actual_fault.id and is_diagnosed(
        posterior, threshold=confidence_threshold
    )

    return BaselineResult(
        total_time_seconds=reported_time,
        num_tests=num_tests if stop_when_diagnosed else len(tests),
        correct=correct,
        diagnosed_hypothesis_id=diagnosed_id,
        suite_time_seconds=suite_time,
        time_to_diagnosis_seconds=time_to_diagnosis,
    )
