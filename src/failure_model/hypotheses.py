"""Enumerate single-component failure hypotheses for NVL72."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.failure_model.priors import (
    DEFAULT_PRIORS_PATH,
    expected_hypothesis_count,
    instance_prior,
    load_priors_json,
)
from src.topology.graph import NVL72Topology, load_topology


@dataclass(frozen=True)
class Hypothesis:
    """One single-component failure hypothesis."""

    id: str
    component_id: str
    component_type: str
    prior: float

    def __post_init__(self) -> None:
        if self.prior < 0:
            raise ValueError(f"Prior must be non-negative, got {self.prior}")


def _hypotheses_for_components(
    component_type: str,
    component_ids: Iterable[str],
    prior: float,
) -> list[Hypothesis]:
    return [
        Hypothesis(
            id=f"{component_id}_fail",
            component_id=component_id,
            component_type=component_type,
            prior=prior,
        )
        for component_id in component_ids
    ]


def build_hypotheses(
    topology: NVL72Topology | None = None,
    priors_path: Path | str | None = None,
) -> list[Hypothesis]:
    """Enumerate all 1,404 single-fault hypotheses with normalized priors."""
    topo = topology if topology is not None else load_topology()
    priors_data = load_priors_json(priors_path)

    gpu_prior = instance_prior("gpu", priors_data)
    switch_prior = instance_prior("nvswitch", priors_data)
    link_prior = instance_prior("nvlink", priors_data)
    tray_prior = instance_prior("compute_tray", priors_data)

    hypotheses: list[Hypothesis] = []
    hypotheses.extend(_hypotheses_for_components("gpu", topo.get_gpus(), gpu_prior))
    hypotheses.extend(
        _hypotheses_for_components("nvswitch", topo.get_switches(), switch_prior)
    )
    hypotheses.extend(_hypotheses_for_components("nvlink", topo.links, link_prior))
    hypotheses.extend(
        _hypotheses_for_components("compute_tray", topo.compute_nodes, tray_prior)
    )

    expected = expected_hypothesis_count(priors_data)
    if len(hypotheses) != expected:
        raise ValueError(
            f"Expected {expected} hypotheses, built {len(hypotheses)}"
        )

    total_prior = sum(h.prior for h in hypotheses)
    if abs(total_prior - 1.0) > 1e-9:
        # Numerical guard: renormalize if floating-point drift appears.
        hypotheses = [
            Hypothesis(
                id=h.id,
                component_id=h.component_id,
                component_type=h.component_type,
                prior=h.prior / total_prior,
            )
            for h in hypotheses
        ]

    return hypotheses


def prior_vector(hypotheses: list[Hypothesis]) -> list[float]:
    return [h.prior for h in hypotheses]


def hypothesis_index(hypotheses: list[Hypothesis]) -> dict[str, int]:
    return {h.id: i for i, h in enumerate(hypotheses)}


def load_default_hypotheses() -> list[Hypothesis]:
    """Convenience: topology + default priors path."""
    return build_hypotheses(priors_path=DEFAULT_PRIORS_PATH)
