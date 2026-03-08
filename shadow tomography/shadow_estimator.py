"""
Shadow Tomography Estimator: (1/K) * sum_k Tr(M^{-1}(sigma_k) H)
for a single-qubit system with fixed state rho = |0><0|.

M^{-1}(sigma_k) = 3 * sigma_k  (classical shadow channel inverse)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pauli matrices and identity
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1],  [1, 0]],  dtype=complex)
Y  = np.array([[0, -1j],[1j, 0]], dtype=complex)
Z  = np.array([[1, 0],  [0, -1]], dtype=complex)

PAULIS     = [X, Y, Z]
PAULI_NAMES = ['X', 'Y', 'Z']

# Fixed density matrix rho = |0><0| = (I + Z)/2
RHO = (I2 + Z) / 2


# ---------------------------------------------------------------------------
# Born rule probabilities for rho = |0><0|
# ---------------------------------------------------------------------------
BORN_PROBS = {
    'X': {+1: 0.5, -1: 0.5},
    'Y': {+1: 0.5, -1: 0.5},
    'Z': {+1: 1.0, -1: 0.0},
}


def build_hamiltonian(a: float, b: float, c: float) -> np.ndarray:
    """Return H = aX + bY + cZ."""
    return a * X + b * Y + c * Z


def sample_outcome(basis_name: str, rng: np.random.Generator) -> int:
    """
    Sample a measurement outcome (+1 or -1) for the fixed state rho = |0><0|
    using the precomputed Born rule probabilities.
    """
    probs = BORN_PROBS[basis_name]
    outcomes = [+1, -1]
    weights  = [probs[+1], probs[-1]]
    return rng.choice(outcomes, p=weights)


def build_snapshot(b_k: int, Q_k: np.ndarray) -> np.ndarray:
    """
    Construct the measurement snapshot:
        sigma_k = (I + b_k * Q_k) / 2
    """
    return (I2 + b_k * Q_k) / 2


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
    return np.real(np.trace(shadow_inverse(sigma_k) @ H))


def run_estimator(a: float, b: float, c: float, K: int, seed: int = None):
    """
    Run the shadow estimator for K measurement shots.

    Parameters
    ----------
    a, b, c : float  -- Hamiltonian coefficients (H = aX + bY + cZ)
    K       : int    -- number of measurement shots
    seed    : int    -- optional RNG seed for reproducibility

    Returns
    -------
    sigmas : list of (2,2) complex arrays  -- snapshot matrices
    values : list of floats                -- per-shot estimator values
    mean   : float                         -- (1/K) sum_k value_k
    std    : float                         -- standard deviation of values
    """
    rng = np.random.default_rng(seed)
    H   = build_hamiltonian(a, b, c)

    sigmas = []
    values = []

    for _ in range(K):
        # Step 1: choose a random Pauli basis uniformly
        idx        = rng.integers(0, 3)           # 0, 1, or 2
        basis_name = PAULI_NAMES[idx]
        Q_k        = PAULIS[idx]

        # Step 2: sample measurement outcome via Born rule
        b_k = sample_outcome(basis_name, rng)

        # Step 3: build snapshot sigma_k = (I + b_k Q_k) / 2
        sigma_k = build_snapshot(b_k, Q_k)

        # Step 4: compute per-shot estimator value
        val = estimator_value(sigma_k, H)

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
    print("Fixed state: rho = |0><0|")
    print("=" * 60)

    # --- User inputs ---
    a = float(input("Enter coefficient a (for X): "))
    b = float(input("Enter coefficient b (for Y): "))
    c = float(input("Enter coefficient c (for Z): "))
    K = int(input("Enter number of measurements K: "))

    # --- Build and display Hamiltonian ---
    H = build_hamiltonian(a, b, c)
    print("\nHamiltonian H = aX + bY + cZ:")
    print(H)

    # --- Run estimator ---
    sigmas, values, mean_val, std_val = run_estimator(a, b, c, K)

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
