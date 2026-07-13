"""Load and normalize failure prior probabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORS_PATH = REPO_ROOT / "data" / "failures" / "failure_priors.json"

MODELED_TYPES = ("gpu", "nvswitch", "nvlink", "compute_tray")


def load_priors_json(path: Path | str | None = None) -> dict[str, Any]:
    priors_path = Path(path) if path is not None else DEFAULT_PRIORS_PATH
    with priors_path.open(encoding="utf-8") as f:
        return json.load(f)


def modeled_type_priors(priors_data: dict[str, Any] | None = None) -> dict[str, float]:
    """Return P(fault type | modeled), summing to 1.0 over modeled types."""
    data = priors_data if priors_data is not None else load_priors_json()

    if "type_priors_modeled" in data:
        type_priors = {t: float(data["type_priors_modeled"][t]) for t in MODELED_TYPES}
    else:
        raw = data["raw_type_shares_among_all_failures"]
        modeled_mass = sum(float(raw[t]) for t in MODELED_TYPES)
        if modeled_mass <= 0:
            raise ValueError("Modeled type shares must sum to a positive value")
        type_priors = {t: float(raw[t]) / modeled_mass for t in MODELED_TYPES}

    total = sum(type_priors.values())
    if abs(total - 1.0) > 1e-9:
        type_priors = {t: p / total for t, p in type_priors.items()}
    return type_priors


def instance_prior(component_type: str, priors_data: dict[str, Any] | None = None) -> float:
    """Uniform prior for one instance of a component type."""
    data = priors_data if priors_data is not None else load_priors_json()
    type_priors = modeled_type_priors(data)
    if component_type not in type_priors:
        raise KeyError(f"Unknown component type: {component_type}")

    counts = data["component_counts"]
    count = int(counts[component_type])
    if count <= 0:
        raise ValueError(f"Invalid component count for {component_type}: {count}")
    return type_priors[component_type] / count


def expected_hypothesis_count(priors_data: dict[str, Any] | None = None) -> int:
    data = priors_data if priors_data is not None else load_priors_json()
    return sum(int(data["component_counts"][t]) for t in MODELED_TYPES)
