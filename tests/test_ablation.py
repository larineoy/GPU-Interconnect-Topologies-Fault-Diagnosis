"""Unit tests for ablation helpers (Step 9)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.failure_model.hypotheses import build_hypotheses  # noqa: E402
from src.simulator.runner import (  # noqa: E402
    PrecomputedCandidates,
    build_candidate_tests,
    run_experiment,
    run_infoslice,
)
from src.simulator.fault_injector import inject_fault_by_id  # noqa: E402
from src.test_model.observation_matrix import ObservationModel  # noqa: E402
from src.topology.factory import make_nvl36, make_nvl72  # noqa: E402


def test_nvl36_topology_counts():
    topo = make_nvl36()
    assert topo.graph.number_of_nodes() == 36 + 9
    assert topo.graph.number_of_edges() == 36 * 9
    assert len(topo.compute_nodes) == 9


def test_nvl36_hypothesis_count_scales():
    # 36 + 9 + 324 + 9 = 378
    hyps = build_hypotheses(make_nvl36())
    assert len(hyps) == 378
    assert abs(sum(h.prior for h in hyps) - 1.0) < 1e-9


def test_random_policy_runs():
    topo = make_nvl72()
    hyps = build_hypotheses(topo)
    om = ObservationModel(topo)
    pre = PrecomputedCandidates(build_candidate_tests(topo), hyps, om)
    fault = inject_fault_by_id(hyps, "gpu_0_fail")
    import numpy as np

    result = run_infoslice(
        fault,
        hyps,
        om,
        pre,
        selection_policy="random",
        rng=np.random.default_rng(0),
    )
    assert result.num_tests >= 1
    assert result.correct


def test_ablation_experiment_smoke():
    df = run_experiment(
        num_trials=2,
        seed=0,
        topology=make_nvl36(),
        method_name="scale_nvl36",
        include_baseline=False,
    )
    assert set(df["method"]) == {"scale_nvl36"}
    assert len(df) == 2
