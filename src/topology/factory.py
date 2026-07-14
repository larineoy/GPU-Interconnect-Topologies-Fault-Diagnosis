"""Build bipartite NVLink topologies of configurable size (NVL36 / NVL72)."""

from __future__ import annotations

from typing import Any

from src.topology.graph import NVL72Topology, build_graph


def build_topology_dict(
    *,
    num_gpus: int,
    num_nvswitches: int,
    gpus_per_node: int = 4,
    name: str | None = None,
) -> dict[str, Any]:
    """Construct a fully connected bipartite GPU↔NVSwitch topology dict."""
    if num_gpus <= 0 or num_nvswitches <= 0:
        raise ValueError("num_gpus and num_nvswitches must be positive")
    if num_gpus % gpus_per_node != 0:
        raise ValueError("num_gpus must be divisible by gpus_per_node")

    num_compute_nodes = num_gpus // gpus_per_node
    compute_nodes = []
    for node_idx in range(num_compute_nodes):
        gpu_start = node_idx * gpus_per_node
        gpu_ids = [f"gpu_{gpu_start + offset}" for offset in range(gpus_per_node)]
        compute_nodes.append({"id": f"node_{node_idx}", "gpus": gpu_ids})

    nvswitches = [{"id": f"nvswitch_{i}"} for i in range(num_nvswitches)]
    links = [
        {
            "id": f"link_{gpu_idx}_{switch_idx}",
            "gpu": f"gpu_{gpu_idx}",
            "nvswitch": f"nvswitch_{switch_idx}",
        }
        for gpu_idx in range(num_gpus)
        for switch_idx in range(num_nvswitches)
    ]

    label = name or f"NVL{num_gpus}"
    return {
        "metadata": {
            "name": label,
            "source": "Synthetic fully-connected bipartite model scaled from NVL72 ERA",
            "url": (
                "https://docs.nvidia.com/enterprise-reference-architectures/"
                "nvl72-ai-factory/latest/components.html"
            ),
            "num_gpus": num_gpus,
            "num_nvswitches": num_nvswitches,
            "num_compute_nodes": num_compute_nodes,
            "gpus_per_node": gpus_per_node,
            "total_nvlinks": num_gpus * num_nvswitches,
        },
        "compute_nodes": compute_nodes,
        "nvswitches": nvswitches,
        "links": links,
    }


def make_topology(
    *,
    num_gpus: int,
    num_nvswitches: int,
    gpus_per_node: int = 4,
    name: str | None = None,
) -> NVL72Topology:
    data = build_topology_dict(
        num_gpus=num_gpus,
        num_nvswitches=num_nvswitches,
        gpus_per_node=gpus_per_node,
        name=name,
    )
    return NVL72Topology(data, build_graph(data))


def make_nvl36() -> NVL72Topology:
    """36 GPUs, 9 trays, 9 NVSwitches (half-scale of NVL72 model)."""
    return make_topology(
        num_gpus=36,
        num_nvswitches=9,
        gpus_per_node=4,
        name="NVL36 (scaled bipartite model)",
    )


def make_nvl72() -> NVL72Topology:
    return make_topology(
        num_gpus=72,
        num_nvswitches=18,
        gpus_per_node=4,
        name="NVL72 GB200 UltraServer",
    )
