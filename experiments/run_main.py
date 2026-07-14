#!/usr/bin/env python3
"""Main experiment: InfoSlice vs baseline over N simulated faults.

Usage:
  python experiments/run_main.py
  python experiments/run_main.py --trials 100 --seed 0
  python experiments/run_main.py --trials 1000 --early-stop-baseline
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
from src.utils.metrics import compare_methods, format_summary, summarize_method

RESULTS_DIR = REPO_ROOT / "results"

# Paper outline aspirational targets (binary v1 model does not yet hit these).
PAPER_TARGETS = {
    "infoslice_median_tests": 3.2,
    "infoslice_median_time_seconds": 240.0,  # ~4 minutes
    "baseline_median_time_seconds": 1860.0,  # ~31 minutes
    "speedup": 7.7,
    "accuracy": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InfoSlice vs baseline diagnosis experiment."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of simulated fault scenarios (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--candidate-mode",
        choices=("default", "full"),
        default="default",
        help="Adaptive candidate pool size",
    )
    parser.add_argument(
        "--early-stop-baseline",
        action="store_true",
        help="Stop baseline once diagnosed (vs always running the full suite)",
    )
    parser.add_argument(
        "--uniform-prior",
        action="store_true",
        help="Ignore curated priors; use a uniform prior over hypotheses",
    )
    parser.add_argument(
        "--no-duration-weight",
        action="store_true",
        help="Maximize raw I(T;H) instead of I(T;H)/duration",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="main",
        help="Prefix for files written under results/",
    )
    return parser.parse_args()


def breakdown_by_fault_type(df: pd.DataFrame) -> dict[str, dict]:
    info = df[df["method"] == "infoslice"]
    out: dict[str, dict] = {}
    for fault_type, group in info.groupby("fault_type"):
        out[str(fault_type)] = {
            "n": int(len(group)),
            "median_time_seconds": float(group["total_time_seconds"].median()),
            "median_num_tests": float(group["num_tests"].median()),
            "accuracy": float(group["correct"].mean()),
        }
    return out


def format_targets_gap(comparison: dict) -> str:
    info = comparison["infoslice"]
    lines = [
        "vs paper outline targets (aspirational for binary v1):",
        (
            f"  tests:   {info['median_num_tests']:.1f}  "
            f"(target {PAPER_TARGETS['infoslice_median_tests']})"
        ),
        (
            f"  time:    {info['median_time_seconds']:.1f}s "
            f"(target {PAPER_TARGETS['infoslice_median_time_seconds']:.0f}s / ~4 min)"
        ),
        (
            f"  speedup: {comparison['speedup_median_time']:.2f}× "
            f"(target {PAPER_TARGETS['speedup']}×)"
        ),
        (
            f"  accuracy:{info['accuracy']*100:.1f}% "
            f"(target {PAPER_TARGETS['accuracy']*100:.0f}%+)"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("InfoSlice main experiment")
    print("=" * 60)
    print(f"trials={args.trials}  seed={args.seed}  candidates={args.candidate_mode}")
    print(
        f"baseline={'early-stop' if args.early_stop_baseline else 'full-suite'}  "
        f"prior={'uniform' if args.uniform_prior else 'curated'}  "
        f"duration_weight={not args.no_duration_weight}"
    )
    print()

    t0 = time.perf_counter()
    df = run_experiment(
        num_trials=args.trials,
        seed=args.seed,
        candidate_mode=args.candidate_mode,
        weight_by_duration=not args.no_duration_weight,
        use_uniform_prior=args.uniform_prior,
        stop_baseline_when_diagnosed=args.early_stop_baseline,
    )
    elapsed = time.perf_counter() - t0

    comparison = compare_methods(df)
    by_type = breakdown_by_fault_type(df)

    print(format_summary(comparison))
    print()
    print(format_targets_gap(comparison))
    print()
    print("InfoSlice by fault type:")
    for fault_type, stats in sorted(by_type.items()):
        print(
            f"  {fault_type:13s}  n={stats['n']:4d}  "
            f"median_tests={stats['median_num_tests']:.1f}  "
            f"median_time={stats['median_time_seconds']:.1f}s  "
            f"acc={stats['accuracy']*100:.1f}%"
        )
    print()
    print(f"Wall-clock: {elapsed:.2f}s ({elapsed / max(args.trials, 1) * 1000:.1f} ms/trial)")

    stem = args.output_prefix
    csv_path = RESULTS_DIR / f"{stem}_trials.csv"
    summary_path = RESULTS_DIR / f"{stem}_summary.json"

    df.to_csv(csv_path, index=False)
    payload = {
        "config": {
            "trials": args.trials,
            "seed": args.seed,
            "candidate_mode": args.candidate_mode,
            "early_stop_baseline": args.early_stop_baseline,
            "uniform_prior": args.uniform_prior,
            "weight_by_duration": not args.no_duration_weight,
        },
        "wall_clock_seconds": elapsed,
        "comparison": comparison,
        "by_fault_type": by_type,
        "paper_targets": PAPER_TARGETS,
        "infoslice": summarize_method(df, "infoslice"),
        "baseline": summarize_method(df, "baseline"),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
