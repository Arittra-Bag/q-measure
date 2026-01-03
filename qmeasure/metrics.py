import numpy as np
from typing import Dict, Tuple

def hist_to_probs(hist: Dict[str, int]) -> np.ndarray:
    """Return probability vector aligned to basis ordering."""

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p||q)."""

def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """0.5 * Σ |p - q|."""

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Confidence interval for Bernoulli proportion (extendable per outcome)."""

def multinomial_ci(hist: Dict[str, int], alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
    """Per-outcome CI for multinomial (MVP: Wilson per outcome)."""
