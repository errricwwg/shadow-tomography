"""
Shadow Tomography Estimator: (1/K) * sum_k Tr(M^{-1}(sigma_k) H)
for a single-qubit system with fixed state rho = |0><0|.

Reads measurement data from measurement_data.txt (produced by measurement_encoder.py).
M^{-1}(sigma_k) = 3 * sigma_k  (classical shadow channel inverse)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pauli matrices and identity
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
Z  = np.array([[1,  0 ], [0, -1 ]], dtype=complex)

PAULIS = {'X': X, 'Y': Y, 'Z': Z}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def load_measurements(filename: str = "measurement_data.txt") -> list[tuple[str, int]]:
    """
    Load measurement results from a file produced by measurement_encoder.py.
    Each line must contain: basis outcome  (e.g. "X 1" or "Z -1")

    Returns
    -------
    results : list of (basis, outcome) tuples
    """
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            basis, outcome = line.split()
            results.append((basis, int(outcome)))
    return results


def build_hamiltonian(a: float, b: float, c: float) -> np.ndarray:
    """Return H = aX + bY + cZ."""
    return a * X + b * Y + c * Z


def build_snapshot(basis: str, outcome: int) -> np.ndarray:
    """
    Construct the classical shadow snapshot:
        sigma_k = (I + outcome * Q_k) / 2
    """
    return (I2 + outcome * PAULIS[basis]) / 2


def shadow_inverse(sigma_k: np.ndarray) -> np.ndarray:
    """
    Apply the inverse shadow channel:
        M^{-1}(sigma_k) = 3 * sigma_k
    """
    return 3 * sigma_k


def estimator_value(sigma_k: np.ndarray, H: np.ndarray) -> float:
    """
    Compute the per-shot estimator value:
        value_k = Tr(M^{-1}(sigma_k) H) = Tr(3 * sigma_k * H)
    """
    return float(np.real(np.trace(shadow_inverse(sigma_k) @ H)))


def run_estimator(measurements: list[tuple[str, int]], H: np.ndarray):
    """
    Run the shadow estimator over the loaded measurement data.

    Parameters
    ----------
    measurements : list of (basis, outcome) tuples from measurement_data.txt
    H            : Hamiltonian matrix

    Returns
    -------
    sigmas : list of (2,2) complex arrays  -- snapshot matrices
    values : list of floats                -- per-shot estimator values
    mean   : float                         -- (1/K) sum_k value_k
    std    : float                         -- standard deviation of values
    """
    sigmas = []
    values = []

    for basis, outcome in measurements:
        sigma_k = build_snapshot(basis, outcome)
        val     = estimator_value(sigma_k, H)
        sigmas.append(sigma_k)
        values.append(val)

    values_arr = np.array(values)
    return sigmas, values, float(np.mean(values_arr)), float(np.std(values_arr, ddof=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Shadow Estimator: (1/K) sum_k Tr(3*sigma_k * H)")
    print("Reads data from: measurement_data.txt")
    print("=" * 60)

    # --- Hamiltonian input ---
    a = float(input("Enter coefficient a (for X): "))
    b = float(input("Enter coefficient b (for Y): "))
    c = float(input("Enter coefficient c (for Z): "))

    H = build_hamiltonian(a, b, c)
    print("\nHamiltonian H = aX + bY + cZ:")
    print(H)

    # --- Load measurement data ---
    measurements = load_measurements("measurement_data.txt")
    K = len(measurements)
    print(f"\nLoaded {K} measurements from measurement_data.txt.")

    # --- Run estimator ---
    sigmas, values, mean_val, std_val = run_estimator(measurements, H)

    # --- Output results ---
    print(f"\n{'='*60}")
    print(f"Results for K = {K} shots")
    print(f"{'='*60}")

    print("\n--- Snapshot matrices sigma_k ---")
    for k, sigma in enumerate(sigmas):
        print(f"  sigma_{k+1}:\n{sigma}\n")

    print("--- Per-shot estimator values Tr(3*sigma_k*H) ---")
    for k, val in enumerate(values):
        print(f"  value_{k+1:>4d} = {val:.6f}")

    print(f"\n{'='*60}")
    print(f"  Estimated <H>  = (1/K) sum_k value_k = {mean_val:.6f}")
    print(f"  Std deviation  = {std_val:.6f}")
    print(f"  Exact <H|0>    = Tr(rho H) = c = {c:.6f}  (for verification)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
