"""Unit tests for topology graph construction (Step 2)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_PATH = REPO_ROOT / "data" / "topology" / "nvl72_topology.json"
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_topology.py"

sys.path.insert(0, str(REPO_ROOT))

from src.topology.graph import load_topology  # noqa: E402


@pytest.fixture(scope="module")
def topology():
    if not TOPOLOGY_PATH.exists():
        subprocess.run([sys.executable, str(GENERATE_SCRIPT)], check=True)
    return load_topology(TOPOLOGY_PATH)


def test_topology_json_metadata_counts():
    if not TOPOLOGY_PATH.exists():
        subprocess.run([sys.executable, str(GENERATE_SCRIPT)], check=True)

    with TOPOLOGY_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["num_gpus"] == 72
    assert meta["num_nvswitches"] == 18
    assert meta["num_compute_nodes"] == 18
    assert meta["gpus_per_node"] == 4
    assert meta["total_nvlinks"] == 1296
    assert len(data["compute_nodes"]) == 18
    assert len(data["nvswitches"]) == 18
    assert len(data["links"]) == 1296


def test_graph_node_and_edge_counts(topology):
    # Sanity check from Step 2 guide: 72 GPUs + 18 switches = 90 nodes, 1296 edges
    assert topology.graph.number_of_nodes() == 90
    assert topology.graph.number_of_edges() == 1296


def test_get_gpus_in_tray(topology):
    gpus = topology.get_gpus_in_tray("node_0")
    assert gpus == ["gpu_0", "gpu_1", "gpu_2", "gpu_3"]

    gpus_last = topology.get_gpus_in_tray("node_17")
    assert gpus_last == ["gpu_68", "gpu_69", "gpu_70", "gpu_71"]


def test_get_links_through_switch(topology):
    links = topology.get_links_through_switch("nvswitch_0")
    assert len(links) == 72
    assert links[0] == "link_0_0"
    assert links[-1] == "link_71_0"


def test_get_path_between_gpus_shares_all_switches(topology):
    shared = topology.get_path_between_gpus("gpu_0", "gpu_40")
    assert len(shared) == 18
    assert shared[0] == "nvswitch_0"
    assert shared[-1] == "nvswitch_17"


def test_fully_connected_bipartite(topology):
    for gpu_id in topology.get_gpus():
        neighbors = set(topology.graph.neighbors(gpu_id))
        assert neighbors == set(topology.get_switches())
