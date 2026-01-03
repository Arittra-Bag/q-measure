import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to find qmeasure package
sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path("outputs/bell_readout_calibration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

from qmeasure.config import ExperimentConfig, MeasurementSpec
from qmeasure.simulate import simulate_density
from qmeasure.measure import apply_readout_error

# ----------------------------
# Readout error fitting utilities
# ----------------------------
def _tvd(p_dict, q_dict):
    # total variation distance over union support
    keys = set(p_dict) | set(q_dict)
    return 0.5 * sum(abs(p_dict.get(k, 0) - q_dict.get(k, 0)) for k in keys)

def _fit_readout_error_from_bell(observed_hist, shots, init=(0.02, 0.02)):
    """
    Simple grid-fit for p01,p10 using Bell state's ideal distribution:
    ideal approx: P(00)=0.5, P(11)=0.5, P(01)=P(10)=0
    We'll grid-search p01,p10 to minimize TVD between observed and predicted.
    """
    ideal = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
    obs = {k: v / shots for k, v in observed_hist.items()}
    # Ensure all keys exist
    for k in ["00", "01", "10", "11"]:
        obs.setdefault(k, 0.0)

    grid = np.linspace(0.0, 0.15, 61)  # 0..0.15
    best = (None, None, 1e9)
    # Predict distribution by sampling readout noise on ideal counts (approx via expectation)
    # For small n we can compute expectation exactly assuming independent bit flips:
    def predict(p01, p10):
        # Start from ideal probabilities over 2-qubit strings
        pred = {"00": 0.0, "01": 0.0, "10": 0.0, "11": 0.0}
        for s, ps in ideal.items():
            # probability of flipping each bit independently:
            for t in ["00", "01", "10", "11"]:
                pt = 1.0
                for i in range(2):
                    a = s[i]
                    b = t[i]
                    if a == "0" and b == "0":
                        pt *= (1 - p01)
                    elif a == "0" and b == "1":
                        pt *= p01
                    elif a == "1" and b == "1":
                        pt *= (1 - p10)
                    elif a == "1" and b == "0":
                        pt *= p10
                pred[t] += ps * pt
        return pred

    for p01 in grid:
        for p10 in grid:
            pred = predict(p01, p10)
            score = _tvd(obs, pred)
            if score < best[2]:
                best = (float(p01), float(p10), float(score))
    return {"p01": best[0], "p10": best[1], "tvd": best[2]}

# ----------------------------
# Main experiment
# ----------------------------
def main():
    shots = 20000
    seed = 42

    # Ground truth readout noise we will inject (unknown to fitter)
    true_p01 = 0.06
    true_p10 = 0.03

    # --- Circuit: Bell state via unitaries ---
    def _H():
        return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

    def _I():
        return np.eye(2, dtype=np.complex128)

    def _CNOT_2q():
        return np.array(
            [[1,0,0,0],
             [0,1,0,0],
             [0,0,0,1],
             [0,0,1,0]],
            dtype=np.complex128
        )

    def _kron(*mats):
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def bell_circuit(n_qubits: int):
        assert n_qubits == 2
        U1 = _kron(_H(), _I())   # H on qubit 0
        U2 = _CNOT_2q()          # CNOT 0->1
        return [U1, U2]

    cfg = ExperimentConfig(
        name="bell_readout_calibration",
        n_qubits=2,
        shots=shots,
        seed=seed,
        noise=None,
        measurement=MeasurementSpec(kind="projective"),
    )

    out = simulate_density(cfg, circuit_fn=bell_circuit)
    ideal_hist = out["histogram"]

    # inject readout error (ground-truth)
    noisy_hist = apply_readout_error(ideal_hist, p01=true_p01, p10=true_p10, seed=seed + 1)

    # Fit p01/p10 from observed histogram using the same grid-search logic (inline)
    # (Keep the same fitter you already wrote, but rename it locally if you want.)
    fit = _fit_readout_error_from_bell(noisy_hist, shots=shots)

    payload = {
        "experiment": "bell_readout_calibration",
        "shots": shots,
        "true_readout": {"p01": true_p01, "p10": true_p10},
        "fit_readout": fit,
        "ideal_hist": ideal_hist,
        "noisy_hist": noisy_hist,
    }

    # Save outputs
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    # Tiny markdown report
    report = f"""# Bell Readout Calibration

- Shots: {shots}
- True readout: p01={true_p01:.4f}, p10={true_p10:.4f}
- Fit readout:  p01={fit['p01']:.4f}, p10={fit['p10']:.4f}
- TVD (fit objective): {fit['tvd']:.6f}

## Histograms
- reedout-noisy: {noisy_hist}
"""
    (OUT_DIR / "report.md").write_text(report, encoding='utf-8')

    print("✅ Done.")
    print(f"Saved: {OUT_DIR / 'results.json'}")
    print(f"Saved: {OUT_DIR / 'report.md'}")
    print("\nKey result:")
    print(f"True p01={true_p01:.3f}, p10={true_p10:.3f} | Fit p01={fit['p01']:.3f}, p10={fit['p10']:.3f} | TVD={fit['tvd']:.5f}")

if __name__ == "__main__":
    main()
