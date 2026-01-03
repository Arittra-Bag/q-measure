import numpy as np
from typing import Dict, Any, Tuple

def fit_readout_error_1q(
    observed: Dict[str, int],
    true_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Fit p01 and p10 for 1-qubit readout model via MLE/least-squares.
    Returns {"p01": ..., "p10": ...}
    """

def fit_noise_strength_grid(
    observed: Dict[str, int],
    grid: np.ndarray,
    simulate_fn,   # closure that returns histogram for a given noise strength
    metric: str = "tvd",
) -> Dict[str, Any]:
    """
    Grid search for best noise strength; return best value + curve + metric scores.
    """
