import numpy as np
from typing import Dict, Any, List
from .config import ExperimentConfig
from .noise import (
    kraus_dephasing,
    kraus_depolarizing,
    kraus_amplitude_damping,
    apply_kraus,
)
from .measure import measure_projective, apply_readout_error

def _I():
    return np.eye(2, dtype=np.complex128)

def _kron(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def _ket0(n: int) -> np.ndarray:
    v = np.zeros((2**n,), dtype=np.complex128)
    v[0] = 1.0 + 0j
    return v

def _rho_from_ket(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())

def _apply_unitary_rho(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    return U @ rho @ U.conj().T

def _apply_single_qubit_kraus(rho: np.ndarray, Ks: List[np.ndarray], n: int, target: int) -> np.ndarray:
    """
    Apply 1-qubit kraus set Ks on 'target' qubit of an n-qubit density matrix.
    """
    out = np.zeros_like(rho, dtype=np.complex128)
    for K in Ks:
        ops = []
        for i in range(n):
            ops.append(K if i == target else _I())
        K_full = _kron(*ops)
        out += K_full @ rho @ K_full.conj().T
    return out

def _resolve_targets(n_qubits: int, targets):
    if targets is None:
        return list(range(n_qubits))
    return [int(t) for t in targets]

def simulate_density(
    config: ExperimentConfig,
    circuit_fn,
) -> Dict[str, Any]:
    """
    Deterministic simulation using density matrices + noise (applied once after circuit for MVP).
    Contract for circuit_fn (MVP):
      - circuit_fn(n_qubits) can return:
          (A) a unitary U of shape (2^n, 2^n), OR
          (B) a list of unitaries [U1, U2, ...], OR
          (C) a prepared density matrix rho of shape (2^n, 2^n)
    We then apply noise specs (if any), then sample measurement histogram.
    """
    n = int(config.n_qubits)
    shots = int(config.shots)
    seed = int(config.seed)

    obj = circuit_fn(n)

    # Build initial rho
    if isinstance(obj, list):
        rho = _rho_from_ket(_ket0(n))
        for U in obj:
            rho = _apply_unitary_rho(rho, U)
    else:
        arr = np.asarray(obj)
        if arr.ndim == 2 and arr.shape == (2**n, 2**n):
            # Could be either a unitary or rho; detect by trace ~ 1 and PSD-ish is hard.
            # We'll assume: if it's (approximately) unitary -> treat as unitary
            # else treat as rho.
            # Unitary check: U U† ≈ I
            UU = arr @ arr.conj().T
            if np.allclose(UU, np.eye(2**n), atol=1e-6):
                rho = _rho_from_ket(_ket0(n))
                rho = _apply_unitary_rho(rho, arr)
            else:
                rho = arr.astype(np.complex128)
        else:
            raise ValueError("circuit_fn must return rho (2^n x 2^n), unitary (2^n x 2^n) or list of unitaries.")

    # Apply noise specs (MVP: apply after circuit)
    if config.noise:
        for ns in config.noise:
            kind = ns.kind
            strength = float(ns.strength)
            targets = _resolve_targets(n, ns.targets)

            if kind == "dephasing":
                Ks = kraus_dephasing(strength)
                for t in targets:
                    rho = _apply_single_qubit_kraus(rho, Ks, n=n, target=t)

            elif kind == "depolarizing":
                Ks = kraus_depolarizing(strength)
                for t in targets:
                    rho = _apply_single_qubit_kraus(rho, Ks, n=n, target=t)

            elif kind == "amplitude_damping":
                Ks = kraus_amplitude_damping(strength)
                for t in targets:
                    rho = _apply_single_qubit_kraus(rho, Ks, n=n, target=t)

            elif kind == "readout":
                # handled at measurement time via apply_readout_error
                pass
            else:
                raise ValueError(f"Unknown noise kind: {kind}")

    # Measurement sampling
    hist = measure_projective(rho, shots=shots, seed=seed)

    # Apply readout noise if specified
    if config.noise:
        for ns in config.noise:
            if ns.kind == "readout":
                # For MVP: interpret strength as "p01", and ns.meta isn't present in dataclass;
                # so use p10 = strength too if you want symmetric.
                # Better: pass readout as NoiseSpec(kind="readout", strength=..., targets=None) and
                # encode p10 in config.meta. For now: symmetric default.
                p01 = float(ns.strength)
                p10 = float(ns.strength)
                # If you stored p10 in config.meta, use it:
                if config.meta and "p10" in config.meta:
                    p10 = float(config.meta["p10"])
                hist = apply_readout_error(hist, p01=p01, p10=p10, seed=seed + 7)

    probs = np.real(np.diag(rho)).clip(0, 1)
    probs = probs / probs.sum()

    return {
        "rho_final": rho,
        "probs": probs,
        "histogram": hist,
        "config": {
            "name": config.name,
            "n_qubits": n,
            "shots": shots,
            "seed": seed,
        },
    }

def simulate_mc(
    config: ExperimentConfig,
    circuit_fn,
    n_trajectories: int = 200,
) -> Dict[str, Any]:
    """
    Placeholder for MC trajectories (we'll implement next).
    For now, return density sim output so your code doesn't break.
    """
    out = simulate_density(config, circuit_fn)
    out["note"] = "simulate_mc not implemented yet; returned simulate_density output"
    out["n_trajectories"] = int(n_trajectories)
    return out

def sweep_param(
    base_config: ExperimentConfig,
    param_name: str,
    values: List[float],
    runner,
    circuit_fn,
) -> Dict[str, Any]:
    """
    Minimal param sweep. MVP supports:
      param_name = "noise[0].strength"
    """
    results = []
    for v in values:
        cfg = base_config
        # shallow clone (good enough for MVP)
        # If noise exists, copy list and replace strength
        if cfg.noise:
            noise_copy = []
            for ns in cfg.noise:
                noise_copy.append(type(ns)(kind=ns.kind, strength=ns.strength, targets=ns.targets))
            cfg = type(base_config)(
                name=base_config.name,
                n_qubits=base_config.n_qubits,
                shots=base_config.shots,
                seed=base_config.seed,
                noise=noise_copy,
                measurement=base_config.measurement,
                meta=base_config.meta,
            )

        if param_name == "noise[0].strength":
            if not cfg.noise:
                raise ValueError("No noise in config to sweep.")
            cfg.noise[0].strength = float(v)
        else:
            raise ValueError("MVP sweep supports only param_name='noise[0].strength'")

        out = runner(cfg, circuit_fn)
        results.append({"value": float(v), "out": out})

    return {"param_name": param_name, "values": values, "results": results}
