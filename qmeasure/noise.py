import numpy as np
from typing import Callable, List, Tuple

Kraus = List[np.ndarray]

def kraus_depolarizing(p: float) -> Kraus:
    """Single-qubit depolarizing channel kraus operators."""

def kraus_dephasing(p: float) -> Kraus:
    """Single-qubit dephasing channel kraus operators."""

def kraus_amplitude_damping(gamma: float) -> Kraus:
    """Single-qubit amplitude damping."""

def apply_kraus(rho: np.ndarray, K: Kraus) -> np.ndarray:
    """rho' = Σ_k K_k rho K_k†"""

def compose_channels(channels: List[Callable[[np.ndarray], np.ndarray]]) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function applying channels sequentially."""
