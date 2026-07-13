"""Unit tests for failure priors and hypothesis enumeration (Step 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.failure_model.hypotheses import build_hypotheses  # noqa: E402
from src.failure_model.priors import (  # noqa: E402
    expected_hypothesis_count,
    instance_prior,
    load_priors_json,
    modeled_type_priors,
)
from src.topology.graph import load_topology  # noqa: E402


@pytest.fixture(scope="module")
def priors_data():
    return load_priors_json()


@pytest.fixture(scope="module")
def hypotheses():
    return build_hypotheses(load_topology())


def test_modeled_type_priors_sum_to_one(priors_data):
    type_priors = modeled_type_priors(priors_data)
    assert set(type_priors) == {"gpu", "nvswitch", "nvlink", "compute_tray"}
    assert abs(sum(type_priors.values()) - 1.0) < 1e-12


def test_type_priors_match_renormalized_raw_shares(priors_data):
    raw = priors_data["raw_type_shares_among_all_failures"]
    modeled_mass = (
        raw["gpu"] + raw["nvswitch"] + raw["nvlink"] + raw["compute_tray"]
    )
    expected = {
        "gpu": raw["gpu"] / modeled_mass,
        "nvswitch": raw["nvswitch"] / modeled_mass,
        "nvlink": raw["nvlink"] / modeled_mass,
        "compute_tray": raw["compute_tray"] / modeled_mass,
    }
    actual = modeled_type_priors(priors_data)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value)


def test_hypothesis_count_is_1404(hypotheses, priors_data):
    assert expected_hypothesis_count(priors_data) == 1404
    assert len(hypotheses) == 1404

    by_type = {}
    for h in hypotheses:
        by_type[h.component_type] = by_type.get(h.component_type, 0) + 1
    assert by_type == {
        "gpu": 72,
        "nvswitch": 18,
        "nvlink": 1296,
        "compute_tray": 18,
    }


def test_priors_sum_to_one(hypotheses):
    total = sum(h.prior for h in hypotheses)
    assert total == pytest.approx(1.0, abs=1e-12)


def test_instance_priors_are_uniform_within_type(hypotheses, priors_data):
    for component_type in ("gpu", "nvswitch", "nvlink", "compute_tray"):
        expected = instance_prior(component_type, priors_data)
        typed = [h for h in hypotheses if h.component_type == component_type]
        assert typed
        assert all(h.prior == pytest.approx(expected) for h in typed)


def test_hypothesis_ids_are_unique(hypotheses):
    ids = [h.id for h in hypotheses]
    assert len(ids) == len(set(ids))
    assert "gpu_0_fail" in ids
    assert "nvswitch_0_fail" in ids
    assert "link_0_0_fail" in ids
    assert "node_0_fail" in ids
