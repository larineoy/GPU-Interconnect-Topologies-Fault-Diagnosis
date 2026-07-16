"""Unit tests for the diagnosis simulator (Step 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.failure_model.hypotheses import build_hypotheses  # noqa: E402
from src.simulator.baseline import build_baseline_suite, run_baseline  # noqa: E402
from src.simulator.fault_injector import inject_fault, inject_fault_by_id  # noqa: E402
from src.simulator.runner import (  # noqa: E402
    PrecomputedCandidates,
    build_candidate_tests,
    run_experiment,
    run_infoslice,
)
from src.simulator.test_executor import execute_test  # noqa: E402
from src.test_model.observation_matrix import FAIL, PASS, ObservationModel  # noqa: E402
from src.topology.graph import load_topology  # noqa: E402
from src.utils.metrics import compare_methods, format_summary  # noqa: E402


@pytest.fixture(scope="module")
def topology():
    return load_topology()


@pytest.fixture(scope="module")
def hypotheses(topology):
    return build_hypotheses(topology)


@pytest.fixture(scope="module")
def observation_model(topology):
    return ObservationModel(topology)


def test_inject_fault_respects_support(hypotheses):
    rng = np.random.default_rng(0)
    fault = inject_fault(hypotheses, rng=rng)
    assert fault.id.endswith("_fail")
    assert fault in hypotheses or any(h.id == fault.id for h in hypotheses)


def test_execute_test_matches_observation_model(hypotheses, observation_model):
    fault = inject_fault_by_id(hypotheses, "gpu_0_fail")
    assert (
        execute_test(
            "nvlink_error_check",
            {"gpu_id": "gpu_0"},
            fault,
            observation_model,
        )
        == FAIL
    )
    assert (
        execute_test(
            "nvlink_error_check",
            {"gpu_id": "gpu_50"},
            fault,
            observation_model,
        )
        == PASS
    )


def test_baseline_suite_duration_near_half_hour(topology):
    suite = build_baseline_suite(topology)
    total = sum(v.duration_seconds for v in suite)
    # 300 + 18*30 + 18*45 + 72*5 = 2010
    assert total == pytest.approx(2010)
    assert len(suite) == 1 + 18 + 18 + 72


def test_candidate_default_count(topology):
    candidates = build_candidate_tests(topology, mode="default")
    # 18 + 18 + 1 + 72 + 72 + 17 pivot pairs → 198
    assert len(candidates) == 198


def test_infoslice_diagnoses_known_gpu_fault(
    hypotheses, observation_model, topology
):
    fault = inject_fault_by_id(hypotheses, "gpu_0_fail")
    candidates = build_candidate_tests(topology, mode="default")
    precomputed = PrecomputedCandidates(candidates, hypotheses, observation_model)
    result = run_infoslice(fault, hypotheses, observation_model, precomputed)
    assert result.correct
    assert result.diagnosed_hypothesis_id == "gpu_0_fail"
    assert result.num_tests >= 1
    assert result.total_time_seconds < 2010


def test_baseline_correct_on_known_fault(hypotheses, observation_model, topology):
    fault = inject_fault_by_id(hypotheses, "nvswitch_3_fail")
    result = run_baseline(
        fault,
        hypotheses,
        observation_model,
        suite=build_baseline_suite(topology),
        stop_when_diagnosed=False,
    )
    assert result.suite_time_seconds == pytest.approx(2010)
    assert result.total_time_seconds == pytest.approx(2010)
    assert result.correct


def test_run_experiment_small(topology):
    df = run_experiment(num_trials=5, seed=1, topology=topology)
    assert set(df["method"]) == {"infoslice", "baseline"}
    assert len(df) == 10
    comparison = compare_methods(df)
    assert comparison["infoslice"]["accuracy"] >= 0.0
    assert "Speedup" in format_summary(comparison)
    # InfoSlice should be faster than full baseline suite on these trials
    assert comparison["speedup_median_time"] > 1.0
