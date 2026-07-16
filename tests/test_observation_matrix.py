"""Unit tests for on-the-fly observation outcomes (Step 5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.failure_model.hypotheses import Hypothesis, build_hypotheses  # noqa: E402
from src.test_model.observation_matrix import (  # noqa: E402
    FAIL,
    PASS,
    ObservationModel,
    exercised_components,
    get_expected_outcome,
)
from src.topology.graph import load_topology  # noqa: E402


def _hyp(component_id: str, component_type: str) -> Hypothesis:
    return Hypothesis(
        id=f"{component_id}_fail",
        component_id=component_id,
        component_type=component_type,
        prior=1.0,
    )


@pytest.fixture(scope="module")
def topology():
    return load_topology()


@pytest.fixture(scope="module")
def model(topology):
    return ObservationModel(topology)


def test_intra_tray_gpu_in_scope_fails(topology):
    # Guide sanity check
    assert (
        get_expected_outcome(
            "intra_tray_nccl",
            {"node_id": "node_0"},
            _hyp("gpu_0", "gpu"),
            topology,
        )
        == FAIL
    )


def test_intra_tray_gpu_out_of_scope_passes(topology):
    # Guide sanity check
    assert (
        get_expected_outcome(
            "intra_tray_nccl",
            {"node_id": "node_0"},
            _hyp("gpu_50", "gpu"),
            topology,
        )
        == PASS
    )


def test_intra_tray_exercises_all_nvswitches(topology):
    exercised = exercised_components(
        "intra_tray_nccl", {"node_id": "node_3"}, topology
    )
    assert {"gpu_12", "gpu_13", "gpu_14", "gpu_15", "node_3"}.issubset(exercised)
    assert set(topology.get_switches()).issubset(exercised)
    assert get_expected_outcome(
        "intra_tray_nccl",
        {"node_id": "node_3"},
        _hyp("nvswitch_5", "nvswitch"),
        topology,
    ) == FAIL


def test_cross_tray_pair_examples(topology):
    params = {"gpu_a": "gpu_0", "gpu_b": "gpu_40"}
    assert get_expected_outcome(
        "cross_tray_pair", params, _hyp("gpu_0", "gpu"), topology
    ) == FAIL
    assert get_expected_outcome(
        "cross_tray_pair", params, _hyp("gpu_40", "gpu"), topology
    ) == FAIL
    assert get_expected_outcome(
        "cross_tray_pair", params, _hyp("nvswitch_7", "nvswitch"), topology
    ) == FAIL
    assert get_expected_outcome(
        "cross_tray_pair", params, _hyp("gpu_50", "gpu"), topology
    ) == PASS


def test_nvswitch_slice_discriminates_switches(topology):
    # Data-flow example: switch_3 fault → slice(3) FAIL, slice(5) PASS
    fault = _hyp("nvswitch_3", "nvswitch")
    assert get_expected_outcome(
        "nvswitch_slice", {"switch_id": "nvswitch_3"}, fault, topology
    ) == FAIL
    assert get_expected_outcome(
        "nvswitch_slice", {"switch_id": "nvswitch_5"}, fault, topology
    ) == PASS


def test_nvlink_error_check_examples(topology):
    params = {"gpu_id": "gpu_12"}
    assert get_expected_outcome(
        "nvlink_error_check", params, _hyp("gpu_12", "gpu"), topology
    ) == FAIL
    assert get_expected_outcome(
        "nvlink_error_check", params, _hyp("link_12_0", "nvlink"), topology
    ) == FAIL
    assert get_expected_outcome(
        "nvlink_error_check", params, _hyp("link_12_17", "nvlink"), topology
    ) == FAIL
    assert get_expected_outcome(
        "nvlink_error_check", params, _hyp("gpu_50", "gpu"), topology
    ) == PASS
    assert get_expected_outcome(
        "nvlink_error_check", params, _hyp("nvswitch_0", "nvswitch"), topology
    ) == PASS


def test_full_fabric_fails_on_any_component(topology):
    assert get_expected_outcome(
        "full_fabric_allreduce", {}, _hyp("gpu_71", "gpu"), topology
    ) == FAIL
    assert get_expected_outcome(
        "full_fabric_allreduce", {}, _hyp("link_3_4", "nvlink"), topology
    ) == FAIL
    assert get_expected_outcome(
        "full_fabric_allreduce", {}, _hyp("node_10", "compute_tray"), topology
    ) == FAIL


def test_observation_model_vector_aligns_with_hypotheses(model, topology):
    hypotheses = build_hypotheses(topology)
    outcomes = model.outcomes_for_hypotheses(
        "intra_tray_nccl", {"node_id": "node_0"}, hypotheses
    )
    assert len(outcomes) == len(hypotheses)
    by_id = {h.id: o for h, o in zip(hypotheses, outcomes)}
    assert by_id["gpu_0_fail"] == FAIL
    assert by_id["gpu_50_fail"] == PASS
    assert by_id["nvswitch_0_fail"] == FAIL
