import numpy as np
from typing import Dict, Any, Optional, List
from .config import ExperimentConfig

def simulate_density(
    config: ExperimentConfig,
    circuit_fn,  # callable that returns unitary list or applies gates; MVP can be simple
) -> Dict[str, Any]:
    """
    Deterministic simulation using density matrices + noise after each step.
    Returns dict with rho_final, probs, histogram (via sampling).
    """

def simulate_mc(
    config: ExperimentConfig,
    circuit_fn,
    n_trajectories: int = 200,
) -> Dict[str, Any]:
    """
    Monte Carlo trajectories to estimate uncertainty bands.
    Returns mean histogram + variance + CIs.
    """

def sweep_param(
    base_config: ExperimentConfig,
    param_name: str,          # e.g. "noise[0].strength"
    values: List[float],
    runner,                   # simulate_density or simulate_mc
    circuit_fn,
) -> Dict[str, Any]:
    """Run a parameter sweep, return results + summary table."""
