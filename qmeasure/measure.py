import numpy as np
from typing import Dict, Optional

def measure_projective(
    rho: np.ndarray,
    shots: int,
    seed: int = 42,
    readout_confusion: Optional[np.ndarray] = None
) -> Dict[str, int]:
    """
    Sample bitstrings from rho's diagonal (computational basis).
    If readout_confusion is provided, it must be a 2x2 confusion matrix for independent per-qubit readout:
        [[P(0|0), P(1|0)],
         [P(0|1), P(1|1)]]
    Returns histogram like {"00": 123, "01": 456, ...}
    """
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square density matrix")

    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("rho dimension must be a power of 2")

    probs = np.real(np.diag(rho)).clip(0, 1)
    s = probs.sum()
    if s <= 0:
        raise ValueError("Invalid density matrix diagonal (sum <= 0)")
    probs = probs / s

    rng = np.random.default_rng(seed)
    idx = rng.choice(dim, size=shots, p=probs)

    hist: Dict[str, int] = {}
    for i in idx:
        b = format(int(i), f"0{n_qubits}b")
        hist[b] = hist.get(b, 0) + 1

    # Optional: apply independent readout confusion to the sampled histogram
    if readout_confusion is not None:
        if readout_confusion.shape != (2, 2):
            raise ValueError("readout_confusion must be 2x2 for MVP")
        # Convert confusion matrix into p01/p10 (independent model)
        # Confusion: rows = true bit (0/1), cols = observed bit (0/1)
        #   P(obs=1|true=0) = p01
        #   P(obs=0|true=1) = p10
        p01 = float(readout_confusion[0, 1])
        p10 = float(readout_confusion[1, 0])
        hist = apply_readout_error(hist, p01=p01, p10=p10, seed=seed + 999)

    return hist


def apply_readout_error(
    histogram: Dict[str, int],
    p01: float,
    p10: float,
    seed: int = 42
) -> Dict[str, int]:
    """
    Flip bits probabilistically to simulate readout noise (independent per bit):
      0 -> 1 with prob p01
      1 -> 0 with prob p10
    """
    p01 = float(p01)
    p10 = float(p10)
    if not (0.0 <= p01 <= 1.0 and 0.0 <= p10 <= 1.0):
        raise ValueError("p01 and p10 must be in [0, 1]")

    rng = np.random.default_rng(seed)
    out: Dict[str, int] = {}

    for bitstr, c in histogram.items():
        if c <= 0:
            continue
        for _ in range(int(c)):
            bits = list(bitstr)
            for j, bj in enumerate(bits):
                r = rng.random()
                if bj == "0":
                    if r < p01:
                        bits[j] = "1"
                else:  # bj == "1"
                    if r < p10:
                        bits[j] = "0"
            bs2 = "".join(bits)
            out[bs2] = out.get(bs2, 0) + 1

    return out
