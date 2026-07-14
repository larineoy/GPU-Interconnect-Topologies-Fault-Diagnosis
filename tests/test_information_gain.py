"""Unit tests for mutual information, selection, and posterior updates (Step 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.algorithm.greedy_selector import select_next_test  # noqa: E402
from src.algorithm.information_gain import (  # noqa: E402
    compute_information_gain,
    compute_information_gain_fast,
    shannon_entropy,
)
from src.algorithm.posterior_update import (  # noqa: E402
    is_diagnosed,
    update_posterior,
    update_posterior_from_fail_indices,
)
from src.failure_model.hypotheses import build_hypotheses  # noqa: E402
from src.test_model.catalog import expand_variants, get_test_type, load_test_types  # noqa: E402
from src.test_model.observation_matrix import ObservationModel  # noqa: E402
from src.topology.graph import load_topology  # noqa: E402


@pytest.fixture(scope="module")
def topology():
    return load_topology()


@pytest.fixture(scope="module")
def hypotheses(topology):
    return build_hypotheses(topology)


@pytest.fixture(scope="module")
def observation_model(topology):
    return ObservationModel(topology)


@pytest.fixture(scope="module")
def prior(hypotheses):
    return np.array([h.prior for h in hypotheses], dtype=float)


def test_shannon_entropy_uniform():
    p = np.ones(4) / 4
    assert shannon_entropy(p) == pytest.approx(np.log(4))


def test_information_gain_non_negative(hypotheses, prior, observation_model):
    def outcome_fn(test_id, params, hyp):
        return observation_model.expected_outcome(test_id, params, hyp)

    ig = compute_information_gain(
        "nvlink_error_check",
        {"gpu_id": "gpu_0"},
        prior,
        hypotheses,
        outcome_fn,
    )
    assert ig >= 0.0


def test_fast_ig_matches_generic(hypotheses, prior, observation_model):
    from src.algorithm.greedy_selector import fail_indices_for_test

    component_to_index = {h.component_id: i for i, h in enumerate(hypotheses)}
    params = {"gpu_id": "gpu_0"}
    fail_idx = fail_indices_for_test(
        "nvlink_error_check", params, observation_model, component_to_index
    )
    fast = compute_information_gain_fast(prior, fail_idx)

    def outcome_fn(test_id, p, hyp):
        return observation_model.expected_outcome(test_id, p, hyp)

    slow = compute_information_gain(
        "nvlink_error_check", params, prior, hypotheses, outcome_fn
    )
    assert fast == pytest.approx(slow, abs=1e-10)


def test_discriminative_test_has_higher_ig_than_redundant(
    hypotheses, prior, observation_model
):
    from src.algorithm.greedy_selector import fail_indices_for_test

    component_to_index = {h.component_id: i for i, h in enumerate(hypotheses)}

    # Slice tests discriminate among switches; full fabric does not (always FAIL).
    ig_slice = compute_information_gain_fast(
        prior,
        fail_indices_for_test(
            "nvswitch_slice",
            {"switch_id": "nvswitch_0"},
            observation_model,
            component_to_index,
        ),
    )
    ig_full = compute_information_gain_fast(
        prior,
        fail_indices_for_test(
            "full_fabric_allreduce",
            {},
            observation_model,
            component_to_index,
        ),
    )
    assert ig_slice > ig_full
    assert ig_full == pytest.approx(0.0, abs=1e-12)


def test_posterior_update_zeros_inconsistent(hypotheses, prior, observation_model):
    def outcome_fn(test_id, params, hyp):
        return observation_model.expected_outcome(test_id, params, hyp)

    # FAIL on nvlink_error_check(gpu_0) eliminates hypotheses outside that scope.
    updated = update_posterior(
        prior,
        "nvlink_error_check",
        {"gpu_id": "gpu_0"},
        0.0,  # FAIL
        hypotheses,
        outcome_fn,
    )
    assert updated.sum() == pytest.approx(1.0)
    idx = {h.id: i for i, h in enumerate(hypotheses)}
    assert updated[idx["gpu_50_fail"]] == pytest.approx(0.0)
    assert updated[idx["gpu_0_fail"]] > 0.0
    assert updated[idx["link_0_0_fail"]] > 0.0


def test_fast_posterior_matches_generic(hypotheses, prior, observation_model):
    from src.algorithm.greedy_selector import fail_indices_for_test

    component_to_index = {h.component_id: i for i, h in enumerate(hypotheses)}
    params = {"gpu_id": "gpu_0"}
    fail_idx = fail_indices_for_test(
        "nvlink_error_check", params, observation_model, component_to_index
    )

    def outcome_fn(test_id, p, hyp):
        return observation_model.expected_outcome(test_id, p, hyp)

    slow = update_posterior(
        prior, "nvlink_error_check", params, 0.0, hypotheses, outcome_fn
    )
    fast = update_posterior_from_fail_indices(prior, fail_idx, 0.0)
    assert np.allclose(slow, fast)


def test_greedy_prefers_high_ig_per_time(hypotheses, prior, observation_model, topology):
    # Restrict to a small candidate set where the answer is obvious:
    # nvlink_error_check is cheap (5s) and informative vs full fabric (300s, IG≈0).
    candidates = []
    candidates.extend(expand_variants(get_test_type("nvlink_error_check"), topology)[:3])
    candidates.extend(expand_variants(get_test_type("full_fabric_allreduce"), topology))
    candidates.extend(expand_variants(get_test_type("dcgm_level3_gpu"), topology)[:1])

    scored = select_next_test(
        prior, hypotheses, candidates, observation_model, weight_by_duration=True
    )
    assert scored.variant.test_id == "nvlink_error_check"
    assert scored.information_gain >= 0.0
    assert scored.score == pytest.approx(
        scored.information_gain / scored.variant.duration_seconds
    )


def test_select_over_full_catalog_runs_quickly(
    hypotheses, prior, observation_model, topology
):
    import time

    from src.test_model.catalog import expand_all_variants

    candidates = expand_all_variants(topology)
    assert len(candidates) == 2629

    t0 = time.perf_counter()
    scored = select_next_test(prior, hypotheses, candidates, observation_model)
    elapsed = time.perf_counter() - t0

    assert scored.information_gain >= 0.0
    assert elapsed < 5.0  # should be well under this on a laptop


def test_is_diagnosed_threshold():
    posterior = np.zeros(10)
    posterior[3] = 0.96
    posterior[4] = 0.04
    # Deterministic v1: require unique support by default
    assert not is_diagnosed(posterior, threshold=0.95)
    assert is_diagnosed(posterior, threshold=0.95, require_unique_support=False)
    posterior[4] = 0.0
    posterior[3] = 1.0
    assert is_diagnosed(posterior, threshold=0.95)


def test_entropy_decreases_after_informative_update(
    hypotheses, prior, observation_model
):
    from src.algorithm.greedy_selector import fail_indices_for_test

    component_to_index = {h.component_id: i for i, h in enumerate(hypotheses)}
    fail_idx = fail_indices_for_test(
        "nvlink_error_check",
        {"gpu_id": "gpu_0"},
        observation_model,
        component_to_index,
    )
    before = shannon_entropy(prior)
    after_post = update_posterior_from_fail_indices(prior, fail_idx, 0.0)
    after = shannon_entropy(after_post)
    assert after < before
