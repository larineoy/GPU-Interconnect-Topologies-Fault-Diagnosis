#!/usr/bin/env python3
"""Ablation studies for InfoSlice.

Ablations:
  1. random          — random informative test selection vs greedy InfoSlice
  2. no_duration     — maximize I(T;H) without dividing by duration
  3. uniform_prior   — uniform hypothesis prior vs curated priors
  4. scale_nvl36     — half-scale topology (36 GPU / 9 switch)
  5. scale_nvl72     — full NVL72 (reference)

Usage:
  python experiments/run_ablation.py
  python experiments/run_ablation.py --trials 200 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulator.runner import run_experiment
from src.topology.factory import make_nvl36, make_nvl72
from src.utils.metrics import format_ablation_table, summarize_all_methods

RESULTS_DIR = REPO_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InfoSlice ablation studies.")
    parser.add_argument("--trials", type=int, default=500, help="Trials per ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ablation",
        help="Prefix for files under results/",
    )
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="Skip NVL36/NVL72 scaling ablations",
    )
    return parser.parse_args()


def run_variant(
    *,
    label: str,
    trials: int,
    seed: int,
    topology=None,
    selection_policy: str = "greedy",
    weight_by_duration: bool = True,
    use_uniform_prior: bool = False,
) -> pd.DataFrame:
    print(f"\n--- {label} ---")
    t0 = time.perf_counter()
    df = run_experiment(
        num_trials=trials,
        seed=seed,
        topology=topology,
        selection_policy=selection_policy,  # type: ignore[arg-type]
        weight_by_duration=weight_by_duration,
        use_uniform_prior=use_uniform_prior,
        method_name=label,
        include_baseline=False,
    )
    elapsed = time.perf_counter() - t0
    stats = summarize_all_methods(df)[label]
    print(
        f"  median {stats['median_time_seconds']:.1f}s, "
        f"{stats['median_num_tests']:.1f} tests, "
        f"acc {stats['accuracy']*100:.1f}%  ({elapsed:.1f}s wall)"
    )
    return df


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("InfoSlice ablation studies")
    print("=" * 60)
    print(f"trials/ablation={args.trials}  seed={args.seed}")

    frames: list[pd.DataFrame] = []

    # 1) Full InfoSlice (reference)
    frames.append(
        run_variant(
            label="infoslice",
            trials=args.trials,
            seed=args.seed,
            topology=make_nvl72(),
        )
    )

    # 2) Random selection
    frames.append(
        run_variant(
            label="random",
            trials=args.trials,
            seed=args.seed,
            topology=make_nvl72(),
            selection_policy="random",
        )
    )

    # 3) No duration weighting
    frames.append(
        run_variant(
            label="no_duration_weight",
            trials=args.trials,
            seed=args.seed,
            topology=make_nvl72(),
            weight_by_duration=False,
        )
    )

    # 4) Uniform priors
    frames.append(
        run_variant(
            label="uniform_prior",
            trials=args.trials,
            seed=args.seed,
            topology=make_nvl72(),
            use_uniform_prior=True,
        )
    )

    if not args.skip_scale:
        # 5) Topology scale
        frames.append(
            run_variant(
                label="scale_nvl36",
                trials=args.trials,
                seed=args.seed,
                topology=make_nvl36(),
            )
        )
        frames.append(
            run_variant(
                label="scale_nvl72",
                trials=args.trials,
                seed=args.seed,
                topology=make_nvl72(),
            )
        )

    df = pd.concat(frames, ignore_index=True)
    summaries = summarize_all_methods(df)

    print("\n" + "=" * 60)
    print("Ablation summary")
    print("=" * 60)
    print(format_ablation_table(summaries))

    # Relative to infoslice reference
    if "infoslice" in summaries and "random" in summaries:
        ref_t = summaries["infoslice"]["median_time_seconds"]
        rand_t = summaries["random"]["median_time_seconds"]
        print(
            f"\nInfoSlice vs random speedup: "
            f"{rand_t / ref_t:.2f}× on median time"
            if ref_t > 0
            else ""
        )

    csv_path = RESULTS_DIR / f"{args.output_prefix}_trials.csv"
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.json"
    df.to_csv(csv_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "trials": args.trials,
                    "seed": args.seed,
                    "skip_scale": args.skip_scale,
                },
                "summaries": summaries,
            },
            f,
            indent=2,
        )
        f.write("\n")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
