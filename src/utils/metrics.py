"""Diagnosis metrics: time, test count, accuracy, speedup."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrialResult:
    trial_id: int
    method: str
    fault_id: str
    fault_type: str
    total_time_seconds: float
    num_tests: int
    correct: bool
    diagnosed_hypothesis_id: str | None


def results_to_dataframe(results: list[TrialResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def summarize_method(df: pd.DataFrame, method: str) -> dict[str, Any]:
    subset = df[df["method"] == method]
    if subset.empty:
        raise ValueError(f"No rows for method={method}")
    return {
        "method": method,
        "n_trials": int(len(subset)),
        "median_time_seconds": float(subset["total_time_seconds"].median()),
        "mean_time_seconds": float(subset["total_time_seconds"].mean()),
        "p90_time_seconds": float(subset["total_time_seconds"].quantile(0.9)),
        "median_num_tests": float(subset["num_tests"].median()),
        "mean_num_tests": float(subset["num_tests"].mean()),
        "accuracy": float(subset["correct"].mean()),
    }


def compare_methods(
    df: pd.DataFrame,
    *,
    method: str = "infoslice",
    baseline: str = "baseline",
) -> dict[str, Any]:
    info = summarize_method(df, method)
    base = summarize_method(df, baseline)
    speedup = (
        base["median_time_seconds"] / info["median_time_seconds"]
        if info["median_time_seconds"] > 0
        else float("inf")
    )
    result: dict[str, Any] = {
        method: info,
        baseline: base,
        "method": method,
        "baseline_method": baseline,
        "speedup_median_time": float(speedup),
        "accuracy_delta": info["accuracy"] - base["accuracy"],
    }
    # Back-compat keys used by experiments/run_main.py
    if method == "infoslice":
        result["infoslice"] = info
    if baseline == "baseline":
        result["baseline"] = base
    return result


def summarize_all_methods(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {method: summarize_method(df, method) for method in sorted(df["method"].unique())}


def format_summary(comparison: dict[str, Any]) -> str:
    method = comparison.get("method", "infoslice")
    baseline = comparison.get("baseline_method", "baseline")
    info = comparison.get(method) or comparison.get("infoslice")
    base = comparison.get(baseline) or comparison.get("baseline")
    if info is None or base is None:
        raise KeyError("comparison missing method summaries")
    lines = [
        f"{method} vs {baseline}",
        (
            f"  {method}: median {info['median_time_seconds']:.1f}s, "
            f"{info['median_num_tests']:.1f} tests, "
            f"accuracy {info['accuracy']*100:.1f}%"
        ),
        (
            f"  {baseline}:  median {base['median_time_seconds']:.1f}s, "
            f"{base['median_num_tests']:.1f} tests, "
            f"accuracy {base['accuracy']*100:.1f}%"
        ),
        f"  Speedup (median time): {comparison['speedup_median_time']:.2f}×",
    ]
    return "\n".join(lines)


def format_ablation_table(summaries: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"{'method':28s}  {'median_t':>10s}  {'median_n':>8s}  {'acc':>7s}  {'n':>5s}"
    ]
    for method, stats in summaries.items():
        lines.append(
            f"{method:28s}  {stats['median_time_seconds']:10.1f}  "
            f"{stats['median_num_tests']:8.1f}  {stats['accuracy']*100:6.1f}%  "
            f"{stats['n_trials']:5d}"
        )
    return "\n".join(lines)
