"""Load nvl72_topology.json and build a NetworkX bipartite graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOPOLOGY_PATH = REPO_ROOT / "data" / "topology" / "nvl72_topology.json"


class NVL72Topology:
    """In-memory NVL72 fabric: GPUs ↔ NVSwitches, plus compute-tray grouping."""

    def __init__(self, data: dict[str, Any], graph: nx.Graph):
        self.data = data
        self.graph = graph
        self.metadata: dict[str, Any] = data["metadata"]

        self._gpus_by_node: dict[str, list[str]] = {
            node["id"]: list(node["gpus"]) for node in data["compute_nodes"]
        }
        self._node_by_gpu: dict[str, str] = {
            gpu_id: node_id
            for node_id, gpus in self._gpus_by_node.items()
            for gpu_id in gpus
        }
        self._links_by_switch: dict[str, list[str]] = {}
        self._link_by_endpoints: dict[tuple[str, str], str] = {}
        for link in data["links"]:
            switch_id = link["nvswitch"]
            self._links_by_switch.setdefault(switch_id, []).append(link["id"])
            self._link_by_endpoints[(link["gpu"], switch_id)] = link["id"]

    @property
    def gpus(self) -> list[str]:
        return sorted(self._node_by_gpu.keys(), key=_numeric_suffix)

    @property
    def nvswitches(self) -> list[str]:
        return [s["id"] for s in self.data["nvswitches"]]

    @property
    def compute_nodes(self) -> list[str]:
        return [n["id"] for n in self.data["compute_nodes"]]

    @property
    def links(self) -> list[str]:
        return [link["id"] for link in self.data["links"]]

    def get_gpus(self) -> list[str]:
        return self.gpus

    def get_switches(self) -> list[str]:
        return self.nvswitches

    def get_gpus_in_tray(self, node_id: str) -> list[str]:
        """Return GPU IDs belonging to a compute tray/node."""
        if node_id not in self._gpus_by_node:
            raise KeyError(f"Unknown compute node: {node_id}")
        return list(self._gpus_by_node[node_id])

    def get_tray_for_gpu(self, gpu_id: str) -> str:
        if gpu_id not in self._node_by_gpu:
            raise KeyError(f"Unknown GPU: {gpu_id}")
        return self._node_by_gpu[gpu_id]

    def get_links_through_switch(self, switch_id: str) -> list[str]:
        """Return all NVLink IDs incident on an NVSwitch."""
        if switch_id not in self._links_by_switch:
            raise KeyError(f"Unknown NVSwitch: {switch_id}")
        return list(self._links_by_switch[switch_id])

    def get_link(self, gpu_id: str, switch_id: str) -> str:
        key = (gpu_id, switch_id)
        if key not in self._link_by_endpoints:
            raise KeyError(f"No link between {gpu_id} and {switch_id}")
        return self._link_by_endpoints[key]

    def get_path_between_gpus(self, gpu_a: str, gpu_b: str) -> list[str]:
        """Return NVSwitches that both GPUs connect to (shared fabric paths).

        In the fully connected NVL72 bipartite model, every GPU connects to
        every NVSwitch, so any pair shares all switches.
        """
        if gpu_a not in self._node_by_gpu:
            raise KeyError(f"Unknown GPU: {gpu_a}")
        if gpu_b not in self._node_by_gpu:
            raise KeyError(f"Unknown GPU: {gpu_b}")
        if gpu_a == gpu_b:
            raise ValueError("gpu_a and gpu_b must be distinct")

        neighbors_a = set(self.graph.neighbors(gpu_a))
        neighbors_b = set(self.graph.neighbors(gpu_b))
        shared = neighbors_a & neighbors_b
        return sorted(shared, key=_numeric_suffix)


def _numeric_suffix(component_id: str) -> tuple[str, int]:
    prefix, _, suffix = component_id.rpartition("_")
    try:
        return prefix, int(suffix)
    except ValueError:
        return component_id, -1


def load_topology_json(path: Path | str | None = None) -> dict[str, Any]:
    topology_path = Path(path) if path is not None else DEFAULT_TOPOLOGY_PATH
    with topology_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_graph(data: dict[str, Any]) -> nx.Graph:
    """Build an undirected bipartite graph: GPU nodes ↔ NVSwitch nodes."""
    graph = nx.Graph()

    for node in data["compute_nodes"]:
        for gpu_id in node["gpus"]:
            graph.add_node(
                gpu_id,
                bipartite=0,
                component_type="gpu",
                compute_node=node["id"],
            )

    for switch in data["nvswitches"]:
        graph.add_node(
            switch["id"],
            bipartite=1,
            component_type="nvswitch",
        )

    for link in data["links"]:
        graph.add_edge(
            link["gpu"],
            link["nvswitch"],
            link_id=link["id"],
            component_type="nvlink",
        )

    return graph


def load_topology(path: Path | str | None = None) -> NVL72Topology:
    """Load topology JSON and return a queryable NVL72Topology object."""
    data = load_topology_json(path)
    graph = build_graph(data)
    return NVL72Topology(data, graph)
