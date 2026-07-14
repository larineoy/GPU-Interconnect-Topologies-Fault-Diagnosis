"""Simulation of fault injection, test execution, and experiment runs."""

from src.simulator.baseline import build_baseline_suite, run_baseline
from src.simulator.fault_injector import inject_fault
from src.simulator.runner import build_candidate_tests, run_experiment, run_infoslice
from src.simulator.test_executor import execute_test

__all__ = [
    "build_baseline_suite",
    "run_baseline",
    "inject_fault",
    "build_candidate_tests",
    "run_experiment",
    "run_infoslice",
    "execute_test",
]
