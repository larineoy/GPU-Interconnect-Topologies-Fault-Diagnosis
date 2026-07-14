#!/usr/bin/env python3
"""Generate paper figures from experiment results (Step 10).

Figures:
  1. NVL72 topology (bipartite schematic)
  2. Entropy vs. test number (InfoSlice convergence traces)
  3. Diagnosis time comparison (box plot)
  4. Number of tests needed (histogram)
  5. Accuracy vs. max tests allowed (diminishing returns)
  6. Ablation bar chart

Usage:
  python experiments/generate_figures.py
  python experiments/generate_figures.py --main results/main_trials.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless-safe matplotlib before pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")
_mpl_cache = Path(__file__).resolve().parents[1] / ".matplotlib_cache"
_mpl_cache.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithm.information_gain import shannon_entropy
from src.algorithm.posterior_update import (
    is_diagnosed,
    update_posterior_from_fail_indices,
)
from src.failure_model.hypotheses import build_hypotheses
from src.simulator.fault_injector import inject_fault_by_id
from src.simulator.runner import (
    PrecomputedCandidates,
    build_candidate_tests,
    select_best_precomputed,
)
from src.simulator.test_executor import execute_test
from src.test_model.observation_matrix import ObservationModel
from src.topology.factory import make_nvl72
from src.topology.graph import load_topology

RESULTS_DIR = REPO_ROOT / "results"
PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
RESULTS_FIG_DIR = REPO_ROOT / "results" / "figures"

# Clean, print-friendly style (avoid generic purple AI look)
COLORS = {
    "infoslice": "#0B6E4F",
    "baseline": "#8B4513",
    "random": "#A23B72",
    "no_duration_weight": "#2E86AB",
    "uniform_prior": "#F18F01",
    "scale_nvl36": "#3D5A80",
    "scale_nvl72": "#0B6E4F",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> list[Path]:
    paths = []
    for out_dir in (PAPER_FIG_DIR, RESULTS_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / name
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def fig1_topology() -> list[Path]:
    """Bipartite NVL72 schematic (sampled for readability)."""
    topo = load_topology()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Show a readable subset: 12 GPUs + 6 switches, note full scale in caption text
    gpu_ids = [f"gpu_{i}" for i in range(0, 72, 6)]  # 12 GPUs
    switch_ids = [f"nvswitch_{i}" for i in range(0, 18, 3)]  # 6 switches

    y_gpu = np.linspace(0.05, 0.95, len(gpu_ids))
    y_sw = np.linspace(0.15, 0.85, len(switch_ids))
    x_gpu, x_sw = 0.18, 0.82

    # Draw a sparse sample of edges for texture (every GPU to 2 switches)
    for i, gpu in enumerate(gpu_ids):
        for j, sw in enumerate(switch_ids):
            if (i + j) % 2 == 0:
                ax.plot(
                    [x_gpu, x_sw],
                    [y_gpu[i], y_sw[j]],
                    color="#B8C4CE",
                    lw=0.6,
                    zorder=1,
                )

    ax.scatter(
        [x_gpu] * len(gpu_ids),
        y_gpu,
        s=90,
        c="#1B4965",
        zorder=3,
        label="GPUs (sample)",
    )
    ax.scatter(
        [x_sw] * len(switch_ids),
        y_sw,
        s=140,
        c="#E36414",
        marker="s",
        zorder=3,
        label="NVSwitches (sample)",
    )

    for i, gid in enumerate(gpu_ids):
        ax.text(x_gpu - 0.03, y_gpu[i], gid, ha="right", va="center", fontsize=8)
    for j, sid in enumerate(switch_ids):
        ax.text(x_sw + 0.03, y_sw[j], sid, ha="left", va="center", fontsize=8)

    ax.text(
        0.5,
        0.02,
        "Edges shown sparsely for readability; model is fully connected bipartite.",
        ha="center",
        fontsize=8,
        color="#5C677D",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "NVL72 interconnect (schematic)\n"
        f"Full model: {topo.metadata['num_gpus']} GPUs × "
        f"{topo.metadata['num_nvswitches']} NVSwitches = "
        f"{topo.metadata['total_nvlinks']} NVLinks (fully connected bipartite)"
    )
    ax.legend(loc="lower center", ncol=2, frameon=False)
    return _save(fig, "fig1_nvl72_topology.png")


def _entropy_trace(fault_id: str) -> list[float]:
    topo = make_nvl72()
    hyps = build_hypotheses(topo)
    om = ObservationModel(topo)
    pre = PrecomputedCandidates(build_candidate_tests(topo), hyps, om)
    fault = inject_fault_by_id(hyps, fault_id)
    posterior = np.array([h.prior for h in hyps], dtype=float)
    entropies = [shannon_entropy(posterior)]
    used: set[str] = set()
    while not is_diagnosed(posterior):
        selected = select_best_precomputed(
            posterior, pre, exclude_keys=used, weight_by_duration=True
        )
        if selected is None:
            break
        idx, _, _ = selected
        variant = pre.candidates[idx]
        outcome = execute_test(variant.test_id, variant.params, fault, om)
        posterior = update_posterior_from_fail_indices(
            posterior, pre.fail_indices[idx], outcome
        )
        used.add(variant.key)
        entropies.append(shannon_entropy(posterior))
        if len(entropies) > 80:
            break
    return entropies


def fig2_entropy_convergence() -> list[Path]:
    faults = {
        "GPU (gpu_67)": "gpu_67_fail",
        "NVSwitch (nvswitch_3)": "nvswitch_3_fail",
        "NVLink (link_10_5)": "link_10_5_fail",
        "Tray (node_2)": "node_2_fail",
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    palette = ["#0B6E4F", "#E36414", "#2E86AB", "#A23B72"]
    for (label, fault_id), color in zip(faults.items(), palette):
        ent = _entropy_trace(fault_id)
        ax.plot(range(len(ent)), ent, marker="o", ms=3, lw=1.8, label=label, color=color)

    ax.set_xlabel("Number of tests executed")
    ax.set_ylabel("Posterior entropy H(H) [nats]")
    ax.set_title("Information-gain convergence under InfoSlice")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return _save(fig, "fig2_entropy_convergence.png")


def fig3_diagnosis_time(main_df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    data = [
        main_df.loc[main_df["method"] == "infoslice", "total_time_seconds"] / 60.0,
        main_df.loc[main_df["method"] == "baseline", "total_time_seconds"] / 60.0,
    ]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["InfoSlice", "Baseline\n(full suite)"])
    for patch, color in zip(bp["boxes"], [COLORS["infoslice"], COLORS["baseline"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel("Diagnosis time (minutes)")
    ax.set_title("Diagnosis time: InfoSlice vs full-suite baseline")
    ax.grid(True, axis="y", alpha=0.25)
    return _save(fig, "fig3_diagnosis_time.png")


def fig4_num_tests(main_df: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    info = main_df.loc[main_df["method"] == "infoslice", "num_tests"]
    base = main_df.loc[main_df["method"] == "baseline", "num_tests"]
    bins = np.arange(0, max(info.max(), base.max()) + 5, 5)
    ax.hist(
        info,
        bins=bins,
        alpha=0.75,
        color=COLORS["infoslice"],
        label=f"InfoSlice (median={info.median():.0f})",
        edgecolor="white",
    )
    ax.hist(
        base,
        bins=bins,
        alpha=0.45,
        color=COLORS["baseline"],
        label=f"Baseline (median={base.median():.0f})",
        edgecolor="white",
    )
    ax.set_xlabel("Number of diagnostic tests")
    ax.set_ylabel("Number of trials")
    ax.set_title("Tests required for diagnosis")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    return _save(fig, "fig4_num_tests_hist.png")


def fig5_accuracy_vs_tests(main_df: pd.DataFrame) -> list[Path]:
    info = main_df[main_df["method"] == "infoslice"].copy()
    max_k = int(info["num_tests"].max())
    ks = np.arange(1, max_k + 1)
    # Fraction correctly diagnosed within ≤ k tests
    acc = [
        float(((info["num_tests"] <= k) & info["correct"]).mean()) for k in ks
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ks, acc, color=COLORS["infoslice"], lw=2.2)
    ax.axhline(0.95, color="#888888", ls="--", lw=1, label="95% accuracy")
    # Mark median tests
    med = float(info["num_tests"].median())
    ax.axvline(med, color="#E36414", ls=":", lw=1.5, label=f"median tests ({med:.0f})")
    ax.set_xlabel("Max tests allowed (k)")
    ax.set_ylabel("Fraction correctly diagnosed within ≤ k tests")
    ax.set_title("Accuracy vs. test budget (diminishing returns)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    return _save(fig, "fig5_accuracy_vs_tests.png")


def fig6_ablation(ablation_csv: Path, ablation_summary: Path) -> list[Path]:
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)
        methods = [
            m
            for m in [
                "infoslice",
                "random",
                "no_duration_weight",
                "uniform_prior",
                "scale_nvl36",
                "scale_nvl72",
            ]
            if m in set(df["method"])
        ]
        med_time = [
            df.loc[df["method"] == m, "total_time_seconds"].median() / 60.0
            for m in methods
        ]
        med_tests = [df.loc[df["method"] == m, "num_tests"].median() for m in methods]
    else:
        import json

        with ablation_summary.open() as f:
            summaries = json.load(f)["summaries"]
        methods = list(summaries.keys())
        med_time = [summaries[m]["median_time_seconds"] / 60.0 for m in methods]
        med_tests = [summaries[m]["median_num_tests"] for m in methods]

    labels = {
        "infoslice": "InfoSlice",
        "random": "Random",
        "no_duration_weight": "No duration\nweight",
        "uniform_prior": "Uniform\nprior",
        "scale_nvl36": "NVL36",
        "scale_nvl72": "NVL72",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(methods))
    colors = [COLORS.get(m, "#555555") for m in methods]

    axes[0].bar(x, med_time, color=colors, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels.get(m, m) for m in methods], fontsize=9)
    axes[0].set_ylabel("Median diagnosis time (minutes)")
    axes[0].set_title("Ablation: diagnosis time")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, med_tests, color=colors, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([labels.get(m, m) for m in methods], fontsize=9)
    axes[1].set_ylabel("Median number of tests")
    axes[1].set_title("Ablation: tests required")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Ablation study summary", y=1.02, fontsize=13)
    fig.tight_layout()
    return _save(fig, "fig6_ablation.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate InfoSlice paper figures.")
    p.add_argument(
        "--main",
        type=Path,
        default=RESULTS_DIR / "main_trials.csv",
        help="Path to main experiment trials CSV",
    )
    p.add_argument(
        "--ablation",
        type=Path,
        default=RESULTS_DIR / "ablation_trials.csv",
        help="Path to ablation trials CSV",
    )
    p.add_argument(
        "--ablation-summary",
        type=Path,
        default=RESULTS_DIR / "ablation_summary.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_style()

    if not args.main.exists():
        raise SystemExit(
            f"Missing {args.main}. Run: python experiments/run_main.py --trials 1000"
        )

    main_df = pd.read_csv(args.main)
    print("Generating figures from", args.main)

    all_paths: list[Path] = []
    all_paths += fig1_topology()
    print("  fig1 topology")
    all_paths += fig2_entropy_convergence()
    print("  fig2 entropy")
    all_paths += fig3_diagnosis_time(main_df)
    print("  fig3 time boxplot")
    all_paths += fig4_num_tests(main_df)
    print("  fig4 tests histogram")
    all_paths += fig5_accuracy_vs_tests(main_df)
    print("  fig5 accuracy curve")
    all_paths += fig6_ablation(args.ablation, args.ablation_summary)
    print("  fig6 ablation")

    print("\nWrote figures to:")
    for path in sorted(set(p.parent for p in all_paths)):
        print(f"  {path}/")
    for path in all_paths:
        if "paper/figures" in str(path):
            print(f"  - {path.name}")


if __name__ == "__main__":
    main()
