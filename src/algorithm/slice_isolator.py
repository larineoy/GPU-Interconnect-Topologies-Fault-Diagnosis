"""Isolate a diagnosed faulty slice and continue validation on healthy slices.

A diagnostic slice is a failure-correlation domain (compute tray, NVSwitch
domain, or full fabric). Early isolation releases healthy hardware while
repair proceeds on the suspect slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.failure_model.hypotheses import Hypothesis
from src.topology.graph import NVL72Topology


@dataclass(frozen=True)
class Slice:
    slice_id: str
    slice_type: str  # "compute_tray" | "nvswitch" | "fabric"
    component_ids: frozenset[str]


def build_slices(topology: NVL72Topology) -> list[Slice]:
    slices: list[Slice] = []
    for node_id in topology.compute_nodes:
        comps = set(topology.get_gpus_in_tray(node_id))
        comps.add(node_id)
        slices.append(
            Slice(
                slice_id=f"tray:{node_id}",
                slice_type="compute_tray",
                component_ids=frozenset(comps),
            )
        )
    for switch_id in topology.get_switches():
        comps = {switch_id}
        comps.update(topology.get_links_through_switch(switch_id))
        slices.append(
            Slice(
                slice_id=f"switch:{switch_id}",
                slice_type="nvswitch",
                component_ids=frozenset(comps),
            )
        )
    fabric_comps = set(topology.get_gpus())
    fabric_comps.update(topology.get_switches())
    fabric_comps.update(topology.links)
    fabric_comps.update(topology.compute_nodes)
    slices.append(
        Slice(
            slice_id="fabric:nvl72",
            slice_type="fabric",
            component_ids=frozenset(fabric_comps),
        )
    )
    return slices


def slice_fault_probability(
    slice_: Slice,
    posterior: np.ndarray,
    hypotheses: Sequence[Hypothesis],
) -> float:
    """Posterior mass on hypotheses whose faulty component lies in the slice."""
    total = 0.0
    for hyp, p in zip(hypotheses, posterior):
        if hyp.component_id in slice_.component_ids:
            total += float(p)
    return total


@dataclass
class IsolationDecision:
    isolated: list[Slice]
    cleared: list[Slice]


def evaluate_slices(
    slices: Sequence[Slice],
    posterior: np.ndarray,
    hypotheses: Sequence[Hypothesis],
    *,
    isolation_threshold: float = 0.9,
    clear_threshold: float = 0.05,
) -> IsolationDecision:
    isolated: list[Slice] = []
    cleared: list[Slice] = []
    for slice_ in slices:
        p = slice_fault_probability(slice_, posterior, hypotheses)
        if p >= isolation_threshold:
            isolated.append(slice_)
        elif p <= clear_threshold:
            cleared.append(slice_)
    return IsolationDecision(isolated=isolated, cleared=cleared)
