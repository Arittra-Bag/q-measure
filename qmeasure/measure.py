import numpy as np
from typing import Dict, Optional, Tuple

def measure_projective(
    rho: np.ndarray,
    shots: int,
    seed: int = 42,
    readout_confusion: Optional[np.ndarray] = None
) -> Dict[str, int]:
    """
    Sample bitstrings from rho's diagonal (computational basis).
    Optionally apply readout confusion matrix (2x2 for 1 qubit; for n qubits we use per-qubit confusion).
    Returns histogram: {"00": 123, "01": 456, ...}
    """

def apply_readout_error(
    histogram: Dict[str, int],
    p01: float,
    p10: float,
    seed: int = 42
) -> Dict[str, int]:
    """Flip bits probabilistically to simulate readout noise (MVP: independent per bit)."""
