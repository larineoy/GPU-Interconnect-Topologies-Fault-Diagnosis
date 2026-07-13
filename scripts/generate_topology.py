#!/usr/bin/env python3
"""Generate NVL72 topology JSON from public NVIDIA reference architecture rules.

NVL72 is modeled as a fully connected bipartite graph:
  - Left: 72 GPUs in 18 compute trays (4 GPUs each)
  - Right: 18 NVSwitches
  - Edges: every GPU connects to every NVSwitch → 72 × 18 = 1,296 NVLinks

Sources:
  https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html
  https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.0.0/high-speed-fabric-management.html
"""

from __future__ import annotations

import json
from pathlib import Path

NUM_GPUS = 72
NUM_NVSWITCHES = 18
NUM_COMPUTE_NODES = 18
GPUS_PER_NODE = 4

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "topology" / "nvl72_topology.json"


def build_topology() -> dict:
    compute_nodes = []
    for node_idx in range(NUM_COMPUTE_NODES):
        gpu_start = node_idx * GPUS_PER_NODE
        gpu_ids = [f"gpu_{gpu_start + offset}" for offset in range(GPUS_PER_NODE)]
        compute_nodes.append(
            {
                "id": f"node_{node_idx}",
                "gpus": gpu_ids,
            }
        )

    nvswitches = [{"id": f"nvswitch_{i}"} for i in range(NUM_NVSWITCHES)]

    links = []
    for gpu_idx in range(NUM_GPUS):
        for switch_idx in range(NUM_NVSWITCHES):
            links.append(
                {
                    "id": f"link_{gpu_idx}_{switch_idx}",
                    "gpu": f"gpu_{gpu_idx}",
                    "nvswitch": f"nvswitch_{switch_idx}",
                }
            )

    return {
        "metadata": {
            "name": "NVL72 GB200 UltraServer",
            "source": "NVIDIA NVL72 AI Factory Reference Architecture",
            "url": (
                "https://docs.nvidia.com/enterprise-reference-architectures/"
                "nvl72-ai-factory/latest/components.html"
            ),
            "num_gpus": NUM_GPUS,
            "num_nvswitches": NUM_NVSWITCHES,
            "num_compute_nodes": NUM_COMPUTE_NODES,
            "gpus_per_node": GPUS_PER_NODE,
            "total_nvlinks": NUM_GPUS * NUM_NVSWITCHES,
        },
        "compute_nodes": compute_nodes,
        "nvswitches": nvswitches,
        "links": links,
    }


def main() -> None:
    topology = build_topology()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(topology, f, indent=2)
        f.write("\n")

    meta = topology["metadata"]
    print(f"Wrote {OUTPUT_PATH}")
    print(
        f"  GPUs={meta['num_gpus']}, NVSwitches={meta['num_nvswitches']}, "
        f"nodes={meta['num_compute_nodes']}, links={meta['total_nvlinks']}"
    )


if __name__ == "__main__":
    main()
