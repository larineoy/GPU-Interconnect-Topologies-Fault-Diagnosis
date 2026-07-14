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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.topology.factory import build_topology_dict  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "topology" / "nvl72_topology.json"


def build_topology() -> dict:
    return build_topology_dict(
        num_gpus=72,
        num_nvswitches=18,
        gpus_per_node=4,
        name="NVL72 GB200 UltraServer",
    )


def main() -> None:
    topology = build_topology()
    # Keep the NVL72 ERA citation on the on-disk artifact.
    topology["metadata"]["source"] = "NVIDIA NVL72 AI Factory Reference Architecture"
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
