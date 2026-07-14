"""Main simulation loop: InfoSlice vs baseline over N trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.algorithm.greedy_selector import fail_indices_for_test
from src.algorithm.information_gain import shannon_entropy
from src.algorithm.posterior_update import (
    is_diagnosed,
    map_hypothesis_index,
    update_posterior_from_fail_indices,
)
from src.failure_model.hypotheses import Hypothesis, build_hypotheses
from src.simulator.baseline import BaselineResult, build_baseline_suite, run_baseline
from src.simulator.fault_injector import inject_fault
from src.simulator.test_executor import execute_test
from src.test_model.catalog import (
    TestVariant,
    expand_all_variants,
    expand_variants,
    get_test_type,
)
from src.test_model.observation_matrix import ObservationModel
from src.topology.graph import NVL72Topology, load_topology
from src.utils.metrics import TrialResult, compare_methods, results_to_dataframe


CandidateMode = Literal["default", "full"]
SelectionPolicy = Literal["greedy", "random"]
FINE_GRAINED_TESTS = frozenset({"nvlink_error_check", "dcgm_level3_gpu"})


@dataclass
class InfoSliceResult:
    total_time_seconds: float
    num_tests: int
    correct: bool
    diagnosed_hypothesis_id: str | None
    test_keys: list[str]


class PrecomputedCandidates:
    """Cache FAIL-index sets for fast greedy selection."""

    def __init__(
        self,
        candidates: Sequence[TestVariant],
        hypotheses: Sequence[Hypothesis],
        observation_model: ObservationModel,
    ):
        self.candidates = list(candidates)
        self.hypotheses = list(hypotheses)
        component_to_index = {h.component_id: i for i, h in enumerate(hypotheses)}
        self.fail_indices = [
            fail_indices_for_test(
                v.test_id, v.params, observation_model, component_to_index
            )
            for v in self.candidates
        ]
        self.keys = [v.key for v in self.candidates]
        self.durations = np.array(
            [float(v.duration_seconds) for v in self.candidates], dtype=float
        )


def build_candidate_tests(
    topology: NVL72Topology,
    mode: CandidateMode = "default",
) -> list[TestVariant]:
    """Build the adaptive candidate pool.

    ``default`` uses tray tests, switch slices, per-GPU checks, and pivot
    cross-tray pairs. ``full`` expands every catalog variant (2,629).
    """
    if mode == "full":
        return expand_all_variants(topology)

    candidates: list[TestVariant] = []
    for test_id in (
        "intra_tray_nccl",
        "nvswitch_slice",
        "full_fabric_allreduce",
        "dcgm_level3_gpu",
        "nvlink_error_check",
    ):
        candidates.extend(expand_variants(get_test_type(test_id), topology))

    # Pivot cross-tray pairs: first GPU of tray 0 vs first GPU of every other tray.
    pair_type = get_test_type("cross_tray_pair")
    pivot = topology.get_gpus_in_tray(topology.compute_nodes[0])[0]
    for node_id in topology.compute_nodes[1:]:
        other = topology.get_gpus_in_tray(node_id)[0]
        candidates.append(
            TestVariant(
                test_id=pair_type.test_id,
                params={"gpu_a": pivot, "gpu_b": other},
                duration_seconds=pair_type.duration_seconds,
                name=pair_type.name,
            )
        )
    return candidates


def _gpu_support_count(
    posterior: np.ndarray, hypotheses: Sequence[Hypothesis]
) -> int:
    return int(
        sum(
            1
            for h, p in zip(hypotheses, posterior)
            if p > 1e-15 and h.component_type == "gpu"
        )
    )


def select_best_precomputed(
    posterior: np.ndarray,
    precomputed: PrecomputedCandidates,
    *,
    exclude_keys: set[str],
    weight_by_duration: bool = True,
    min_information_gain: float = 1e-12,
    fine_grained_support_threshold: int = 48,
    fine_grained_gpu_threshold: int = 4,
) -> tuple[int, float, float] | None:
    """Return (index, information_gain, score), or None if no informative test remains."""
    best_idx = -1
    best_score = -1.0
    best_ig = -1.0
    best_key = ""
    best_balance = -1.0

    support_size = int(np.sum(posterior > 1e-15))
    gpu_support = _gpu_support_count(posterior, precomputed.hypotheses)
    allow_fine = (
        support_size <= fine_grained_support_threshold
        or gpu_support <= fine_grained_gpu_threshold
    )

    prior_entropy = shannon_entropy(posterior)

    for i, (variant, fail_idx, key, duration) in enumerate(
        zip(
            precomputed.candidates,
            precomputed.fail_indices,
            precomputed.keys,
            precomputed.durations,
        )
    ):
        if key in exclude_keys:
            continue
        if not allow_fine and variant.test_id in FINE_GRAINED_TESTS:
            continue

        total = float(posterior.sum())
        if total <= 0:
            ig = 0.0
            p_fail = 0.0
        else:
            p = posterior / total
            p_fail = float(p[fail_idx].sum()) if fail_idx.size else 0.0
            p_pass = 1.0 - p_fail
            conditional = 0.0
            if p_fail > 1e-15:
                conditional += p_fail * shannon_entropy(p[fail_idx])
            if p_pass > 1e-15:
                pass_mask = np.ones(p.shape[0], dtype=bool)
                pass_mask[fail_idx] = False
                conditional += p_pass * shannon_entropy(p[pass_mask])
            ig = max(0.0, prior_entropy - conditional)

        if ig <= min_information_gain:
            continue

        score = ig / max(duration, 1e-9) if weight_by_duration else ig
        balance = 4.0 * p_fail * (1.0 - p_fail)

        better = False
        if score > best_score + 1e-15:
            better = True
        elif abs(score - best_score) <= 1e-15:
            if balance > best_balance + 1e-15 or (
                abs(balance - best_balance) <= 1e-15 and key < best_key
            ):
                better = True

        if better:
            best_idx = i
            best_score = score
            best_ig = ig
            best_key = key
            best_balance = balance

    if best_idx < 0:
        return None
    return best_idx, best_ig, best_score


def select_random_informative(
    posterior: np.ndarray,
    precomputed: PrecomputedCandidates,
    *,
    exclude_keys: set[str],
    rng: np.random.Generator,
    min_information_gain: float = 1e-12,
) -> tuple[int, float, float] | None:
    """Pick uniformly among unused candidates with positive information gain."""
    prior_entropy = shannon_entropy(posterior)
    informative: list[tuple[int, float]] = []

    for i, (fail_idx, key) in enumerate(
        zip(precomputed.fail_indices, precomputed.keys)
    ):
        if key in exclude_keys:
            continue
        total = float(posterior.sum())
        if total <= 0:
            continue
        p = posterior / total
        p_fail = float(p[fail_idx].sum()) if fail_idx.size else 0.0
        p_pass = 1.0 - p_fail
        conditional = 0.0
        if p_fail > 1e-15:
            conditional += p_fail * shannon_entropy(p[fail_idx])
        if p_pass > 1e-15:
            pass_mask = np.ones(p.shape[0], dtype=bool)
            pass_mask[fail_idx] = False
            conditional += p_pass * shannon_entropy(p[pass_mask])
        ig = max(0.0, prior_entropy - conditional)
        if ig > min_information_gain:
            informative.append((i, ig))

    if not informative:
        return None
    choice = int(rng.integers(0, len(informative)))
    idx, ig = informative[choice]
    duration = float(precomputed.durations[idx])
    return idx, ig, ig / max(duration, 1e-9)


def run_infoslice(
    actual_fault: Hypothesis,
    hypotheses: Sequence[Hypothesis],
    observation_model: ObservationModel,
    precomputed: PrecomputedCandidates,
    *,
    confidence_threshold: float = 0.95,
    max_tests: int = 80,
    weight_by_duration: bool = True,
    use_uniform_prior: bool = False,
    selection_policy: SelectionPolicy = "greedy",
    rng: np.random.Generator | None = None,
) -> InfoSliceResult:
    """Run adaptive diagnosis for one injected fault."""
    if use_uniform_prior:
        posterior = np.ones(len(hypotheses), dtype=float) / len(hypotheses)
    else:
        posterior = np.array([h.prior for h in hypotheses], dtype=float)

    total_time = 0.0
    num_tests = 0
    used: set[str] = set()
    test_keys: list[str] = []
    local_rng = rng if rng is not None else np.random.default_rng()

    while not is_diagnosed(posterior, threshold=confidence_threshold):
        if num_tests >= max_tests:
            break
        if selection_policy == "random":
            selected = select_random_informative(
                posterior, precomputed, exclude_keys=used, rng=local_rng
            )
        else:
            selected = select_best_precomputed(
                posterior,
                precomputed,
                exclude_keys=used,
                weight_by_duration=weight_by_duration,
            )
        if selected is None:
            break
        idx, _ig, _score = selected
        variant = precomputed.candidates[idx]
        outcome = execute_test(
            variant.test_id, variant.params, actual_fault, observation_model
        )
        posterior = update_posterior_from_fail_indices(
            posterior, precomputed.fail_indices[idx], outcome
        )
        total_time += float(variant.duration_seconds)
        num_tests += 1
        used.add(variant.key)
        test_keys.append(variant.key)

    map_idx = map_hypothesis_index(posterior)
    diagnosed_id = hypotheses[map_idx].id if posterior.sum() > 0 else None
    correct = diagnosed_id == actual_fault.id and is_diagnosed(
        posterior, threshold=confidence_threshold
    )
    return InfoSliceResult(
        total_time_seconds=total_time,
        num_tests=num_tests,
        correct=correct,
        diagnosed_hypothesis_id=diagnosed_id,
        test_keys=test_keys,
    )


def run_experiment(
    num_trials: int = 1000,
    *,
    seed: int = 42,
    candidate_mode: CandidateMode = "default",
    confidence_threshold: float = 0.95,
    weight_by_duration: bool = True,
    use_uniform_prior: bool = False,
    stop_baseline_when_diagnosed: bool = False,
    topology: NVL72Topology | None = None,
    selection_policy: SelectionPolicy = "greedy",
    method_name: str = "infoslice",
    include_baseline: bool = True,
) -> pd.DataFrame:
    """Run N trials of a diagnosis policy (optionally vs baseline)."""
    topo = topology if topology is not None else load_topology()
    hypotheses = build_hypotheses(topo)
    observation_model = ObservationModel(topo)
    candidates = build_candidate_tests(topo, mode=candidate_mode)
    precomputed = PrecomputedCandidates(candidates, hypotheses, observation_model)
    baseline_suite = build_baseline_suite(topo) if include_baseline else []
    rng = np.random.default_rng(seed)

    results: list[TrialResult] = []
    for trial_id in range(num_trials):
        fault = inject_fault(hypotheses, rng=rng)
        trial_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

        info = run_infoslice(
            fault,
            hypotheses,
            observation_model,
            precomputed,
            confidence_threshold=confidence_threshold,
            weight_by_duration=weight_by_duration,
            use_uniform_prior=use_uniform_prior,
            selection_policy=selection_policy,
            rng=trial_rng,
        )
        results.append(
            TrialResult(
                trial_id=trial_id,
                method=method_name,
                fault_id=fault.id,
                fault_type=fault.component_type,
                total_time_seconds=info.total_time_seconds,
                num_tests=info.num_tests,
                correct=info.correct,
                diagnosed_hypothesis_id=info.diagnosed_hypothesis_id,
            )
        )

        if include_baseline:
            base: BaselineResult = run_baseline(
                fault,
                hypotheses,
                observation_model,
                suite=baseline_suite,
                stop_when_diagnosed=stop_baseline_when_diagnosed,
                confidence_threshold=confidence_threshold,
            )
            results.append(
                TrialResult(
                    trial_id=trial_id,
                    method="baseline",
                    fault_id=fault.id,
                    fault_type=fault.component_type,
                    total_time_seconds=base.total_time_seconds,
                    num_tests=base.num_tests,
                    correct=base.correct,
                    diagnosed_hypothesis_id=base.diagnosed_hypothesis_id,
                )
            )

    return results_to_dataframe(results)


def run_and_summarize(num_trials: int = 100, **kwargs) -> tuple[pd.DataFrame, dict]:
    df = run_experiment(num_trials=num_trials, **kwargs)
    comparison = compare_methods(df)
    return df, comparison
