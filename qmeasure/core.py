import numpy as np
from typing import Literal

StateKind = Literal["ket", "rho"]

def ket0(n_qubits: int) -> np.ndarray:
    """|0...0> state vector, shape (2^n,)."""

def density_from_ket(ket: np.ndarray) -> np.ndarray:
    """rho = |psi><psi|."""

def apply_unitary(state: np.ndarray, U: np.ndarray, kind: StateKind) -> np.ndarray:
    """Apply U to ket or rho."""
