"""Unit tests for diagnostic test catalog (Step 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.test_model.catalog import (  # noqa: E402
    expand_all_variants,
    expand_variants,
    get_test_type,
    load_test_types,
    validate_variant_counts,
)
from src.topology.graph import load_topology  # noqa: E402


@pytest.fixture(scope="module")
def topology():
    return load_topology()


@pytest.fixture(scope="module")
def test_types():
    return load_test_types()


def test_catalog_has_at_least_six_types(test_types):
    assert len(test_types) >= 6
    ids = {t.test_id for t in test_types}
    assert {
        "intra_tray_nccl",
        "cross_tray_pair",
        "nvswitch_slice",
        "full_fabric_allreduce",
        "dcgm_level3_gpu",
        "nvlink_error_check",
    }.issubset(ids)


def test_durations_are_positive(test_types):
    for test_type in test_types:
        assert test_type.duration_seconds > 0


def test_declared_variant_counts_match_expansion(topology, test_types):
    counts = validate_variant_counts(topology)
    for test_type in test_types:
        declared, expanded = counts[test_type.test_id]
        assert expanded == declared, (
            f"{test_type.test_id}: declared {declared}, expanded {expanded}"
        )


def test_intra_tray_variants(topology):
    test_type = get_test_type("intra_tray_nccl")
    variants = expand_variants(test_type, topology)
    assert len(variants) == 18
    assert variants[0].params == {"node_id": "node_0"}
    assert variants[0].duration_seconds == 30


def test_cross_tray_excludes_same_tray_pairs(topology):
    variants = expand_variants(get_test_type("cross_tray_pair"), topology)
    assert len(variants) == 2448
    for variant in variants:
        gpu_a = variant.params["gpu_a"]
        gpu_b = variant.params["gpu_b"]
        assert topology.get_tray_for_gpu(gpu_a) != topology.get_tray_for_gpu(gpu_b)


def test_total_variant_count(topology):
    # 18 + 2448 + 18 + 1 + 72 + 72 = 2629
    variants = expand_all_variants(topology)
    assert len(variants) == 2629
    assert len({v.key for v in variants}) == 2629
