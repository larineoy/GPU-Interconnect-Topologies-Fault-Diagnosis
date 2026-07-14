"""On-the-fly (test, hypothesis) → expected outcome mapping.

Outcomes:
  PASS     = 1.0  — test does not exercise the faulty component
  FAIL     = 0.0  — test exercises the faulty component
  DEGRADED = 0.5  — reserved for grey/partial effects (unused in v1 hard model)

The full matrix is never materialized (2,629 tests × 1,404 hypotheses). Callers
query get_expected_outcome / outcome_likelihood on demand.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx

from src.failure_model.hypotheses import Hypothesis
from src.topology.graph import NVL72Topology, load_topology

PASS = 1.0
FAIL = 0.0
DEGRADED = 0.5


def _require_param(params: dict[str, str], key: str) -> str:
    if key not in params:
        raise KeyError(f"Test params missing required key '{key}': {params}")
    return params[key]


def exercised_components(
    test_id: str,
    test_params: dict[str, str],
    topology: NVL72Topology,
) -> frozenset[str]:
    """Return component IDs exercised by a concrete test variant."""
    if test_id == "intra_tray_nccl":
        node_id = _require_param(test_params, "node_id")
        gpus = topology.get_gpus_in_tray(node_id)
        components: set[str] = set(gpus)
        components.add(node_id)
        # NVL72: even intra-tray traffic is switch-mediated.
        components.update(topology.get_switches())
        for gpu_id in gpus:
            for switch_id in topology.get_switches():
                components.add(topology.get_link(gpu_id, switch_id))
        return frozenset(components)

    if test_id == "cross_tray_pair":
        gpu_a = _require_param(test_params, "gpu_a")
        gpu_b = _require_param(test_params, "gpu_b")
        components = {gpu_a, gpu_b}
        components.add(topology.get_tray_for_gpu(gpu_a))
        components.add(topology.get_tray_for_gpu(gpu_b))
        # Fully connected bipartite fabric: pair traffic can use any switch.
        components.update(topology.get_path_between_gpus(gpu_a, gpu_b))
        for gpu_id in (gpu_a, gpu_b):
            for switch_id in topology.get_switches():
                components.add(topology.get_link(gpu_id, switch_id))
        return frozenset(components)

    if test_id == "nvswitch_slice":
        switch_id = _require_param(test_params, "switch_id")
        # Targeted switch-domain probe: the switch and its incident links only.
        # GPU/tray faults are diagnosed by per-GPU tests; including all GPUs here
        # makes every slice predict FAIL for any GPU fault (IG ≈ 0).
        components = {switch_id}
        components.update(topology.get_links_through_switch(switch_id))
        return frozenset(components)

    if test_id == "full_fabric_allreduce":
        components = set(topology.get_gpus())
        components.update(topology.get_switches())
        components.update(topology.links)
        components.update(topology.compute_nodes)
        return frozenset(components)

    if test_id in {"dcgm_level3_gpu", "nvlink_error_check"}:
        gpu_id = _require_param(test_params, "gpu_id")
        components = {gpu_id, topology.get_tray_for_gpu(gpu_id)}
        for switch_id in topology.get_switches():
            components.add(topology.get_link(gpu_id, switch_id))
        return frozenset(components)

    raise ValueError(f"Unknown test_id: {test_id}")


def get_expected_outcome(
    test_id: str,
    test_params: dict[str, str],
    hypothesis: Hypothesis,
    topology: NVL72Topology | nx.Graph,
) -> float:
    """Return PASS/FAIL/DEGRADED for (test, hypothesis) under the single-fault model.

    If ``topology`` is a raw NetworkX graph, the default NVL72 JSON is loaded
    (graph object alone lacks tray/link indexes). Prefer passing ``NVL72Topology``.
    """
    topo = topology if isinstance(topology, NVL72Topology) else load_topology()
    exercised = exercised_components(test_id, test_params, topo)
    if hypothesis.component_id in exercised:
        return FAIL
    return PASS


def outcome_likelihood(
    test_id: str,
    test_params: dict[str, str],
    hypothesis: Hypothesis,
    topology: NVL72Topology,
    observed_outcome: float,
    *,
    match_tolerance: float = 1e-9,
) -> float:
    """P(observed_outcome | hypothesis, test) for the deterministic v1 model.

    Returns 1.0 if the expected outcome matches the observation, else 0.0.
    """
    expected = get_expected_outcome(test_id, test_params, hypothesis, topology)
    return 1.0 if abs(expected - observed_outcome) <= match_tolerance else 0.0


class ObservationModel:
    """Queryable observation model with per-variant exercise-set caching."""

    def __init__(self, topology: NVL72Topology | None = None):
        self.topology = topology if topology is not None else load_topology()
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], frozenset[str]] = {}

    def _params_key(self, test_params: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(test_params.items()))

    def exercised(
        self, test_id: str, test_params: dict[str, str]
    ) -> frozenset[str]:
        key = (test_id, self._params_key(test_params))
        if key not in self._cache:
            self._cache[key] = exercised_components(
                test_id, test_params, self.topology
            )
        return self._cache[key]

    def expected_outcome(
        self,
        test_id: str,
        test_params: dict[str, str],
        hypothesis: Hypothesis,
    ) -> float:
        exercised = self.exercised(test_id, test_params)
        return FAIL if hypothesis.component_id in exercised else PASS

    def likelihood(
        self,
        test_id: str,
        test_params: dict[str, str],
        hypothesis: Hypothesis,
        observed_outcome: float,
    ) -> float:
        expected = self.expected_outcome(test_id, test_params, hypothesis)
        return 1.0 if abs(expected - observed_outcome) < 1e-9 else 0.0

    def outcomes_for_hypotheses(
        self,
        test_id: str,
        test_params: dict[str, str],
        hypotheses: Iterable[Hypothesis],
    ) -> list[float]:
        """Vector of expected outcomes aligned with ``hypotheses``."""
        exercised = self.exercised(test_id, test_params)
        return [
            FAIL if h.component_id in exercised else PASS for h in hypotheses
        ]
