"""Load diagnostic test catalog and expand parameterized variants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

from src.topology.graph import NVL72Topology, load_topology

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "tests" / "test_catalog.json"


@dataclass(frozen=True)
class TestType:
    """A parameterized diagnostic test type from the catalog."""

    test_id: str
    name: str
    duration_seconds: float
    components_exercised: tuple[str, ...]
    parameterized_by: tuple[str, ...]
    num_variants: int
    scope_notes: str = ""
    source: str = ""


@dataclass(frozen=True)
class TestVariant:
    """A concrete runnable test: type + parameter binding."""

    test_id: str
    params: dict[str, str]
    duration_seconds: float
    name: str

    @property
    def key(self) -> str:
        if not self.params:
            return self.test_id
        param_str = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.test_id}({param_str})"


def load_catalog_json(path: Path | str | None = None) -> dict[str, Any]:
    catalog_path = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    with catalog_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_test_types(path: Path | str | None = None) -> list[TestType]:
    data = load_catalog_json(path)
    return [
        TestType(
            test_id=entry["test_id"],
            name=entry["name"],
            duration_seconds=float(entry["duration_seconds"]),
            components_exercised=tuple(entry["components_exercised"]),
            parameterized_by=tuple(entry.get("parameterized_by", [])),
            num_variants=int(entry["num_variants"]),
            scope_notes=entry.get("scope_notes", ""),
            source=entry.get("source", ""),
        )
        for entry in data["tests"]
    ]


def get_test_type(test_id: str, path: Path | str | None = None) -> TestType:
    for test_type in load_test_types(path):
        if test_type.test_id == test_id:
            return test_type
    raise KeyError(f"Unknown test_id: {test_id}")


def _cross_tray_gpu_pairs(topology: NVL72Topology) -> Iterator[tuple[str, str]]:
    for gpu_a, gpu_b in combinations(topology.get_gpus(), 2):
        if topology.get_tray_for_gpu(gpu_a) != topology.get_tray_for_gpu(gpu_b):
            yield gpu_a, gpu_b


def expand_variants(
    test_type: TestType,
    topology: NVL72Topology | None = None,
) -> list[TestVariant]:
    """Expand one catalog test type into concrete variants on the topology."""
    topo = topology if topology is not None else load_topology()
    duration = test_type.duration_seconds
    name = test_type.name
    test_id = test_type.test_id

    if test_id == "intra_tray_nccl":
        return [
            TestVariant(
                test_id=test_id,
                params={"node_id": node_id},
                duration_seconds=duration,
                name=name,
            )
            for node_id in topo.compute_nodes
        ]

    if test_id == "cross_tray_pair":
        return [
            TestVariant(
                test_id=test_id,
                params={"gpu_a": gpu_a, "gpu_b": gpu_b},
                duration_seconds=duration,
                name=name,
            )
            for gpu_a, gpu_b in _cross_tray_gpu_pairs(topo)
        ]

    if test_id == "nvswitch_slice":
        return [
            TestVariant(
                test_id=test_id,
                params={"switch_id": switch_id},
                duration_seconds=duration,
                name=name,
            )
            for switch_id in topo.get_switches()
        ]

    if test_id == "full_fabric_allreduce":
        return [
            TestVariant(
                test_id=test_id,
                params={},
                duration_seconds=duration,
                name=name,
            )
        ]

    if test_id in {"dcgm_level3_gpu", "nvlink_error_check"}:
        return [
            TestVariant(
                test_id=test_id,
                params={"gpu_id": gpu_id},
                duration_seconds=duration,
                name=name,
            )
            for gpu_id in topo.get_gpus()
        ]

    raise ValueError(f"No variant expansion rule for test_id={test_id}")


def expand_all_variants(
    topology: NVL72Topology | None = None,
    catalog_path: Path | str | None = None,
) -> list[TestVariant]:
    """Expand every catalog test type into concrete variants."""
    topo = topology if topology is not None else load_topology()
    variants: list[TestVariant] = []
    for test_type in load_test_types(catalog_path):
        variants.extend(expand_variants(test_type, topo))
    return variants


def validate_variant_counts(
    topology: NVL72Topology | None = None,
    catalog_path: Path | str | None = None,
) -> dict[str, tuple[int, int]]:
    """Return {test_id: (declared_num_variants, expanded_count)}."""
    topo = topology if topology is not None else load_topology()
    result: dict[str, tuple[int, int]] = {}
    for test_type in load_test_types(catalog_path):
        expanded = expand_variants(test_type, topo)
        result[test_type.test_id] = (test_type.num_variants, len(expanded))
    return result
