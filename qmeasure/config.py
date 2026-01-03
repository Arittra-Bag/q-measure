from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class NoiseSpec:
    kind: str                 # "depolarizing" | "dephasing" | "amplitude_damping" | "readout"
    strength: float           # e.g. p or gamma
    targets: Optional[List[int]] = None  # qubits affected

@dataclass
class MeasurementSpec:
    kind: str                 # "projective"
    basis: str = "computational"
    targets: Optional[List[int]] = None  # None => all qubits

@dataclass
class ExperimentConfig:
    name: str
    n_qubits: int
    shots: int = 10_000
    seed: int = 42
    noise: Optional[List[NoiseSpec]] = None
    measurement: MeasurementSpec = MeasurementSpec(kind="projective")
    meta: Optional[Dict[str, Any]] = None
