"""
Shadow Tomography – Measurement Encoder
Generates raw random Pauli measurement data for rho = |0><0|.
Does NOT compute any estimator or apply inverse channel.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Pauli matrices and fixed state
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
Z  = np.array([[1,  0 ], [0, -1 ]], dtype=complex)

PAULIS      = {'X': X, 'Y': Y, 'Z': Z}
BASIS_NAMES = ['X', 'Y', 'Z']

# rho = |0><0| = (I + Z) / 2
RHO = (I2 + Z) / 2

# Born rule probabilities for rho = |0><0|
# P(+1 | basis) = Tr(Pi_+ rho), where Pi_+ = (I + basis) / 2
BORN_PROB_PLUS = {
    'X': 0.5,
    'Y': 0.5,
    'Z': 1.0,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def choose_basis(rng: np.random.Generator) -> str:
    """Randomly select a Pauli basis from {X, Y, Z} with equal probability."""
    return rng.choice(BASIS_NAMES)


def measure(basis: str, rng: np.random.Generator) -> int:
    """
    Perform a projective measurement in the given basis on rho = |0><0|.
    Returns +1 or -1 according to the Born rule.
    """
    p_plus = BORN_PROB_PLUS[basis]
    return int(rng.choice([+1, -1], p=[p_plus, 1 - p_plus]))


def run_measurements(K: int, seed: int = None) -> list[tuple[str, int]]:
    """
    Perform K random Pauli measurements on rho = |0><0|.

    Returns
    -------
    results : list of (basis, outcome) tuples
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(K):
        basis   = choose_basis(rng)
        outcome = measure(basis, rng)
        results.append((basis, outcome))
    return results


def print_results(results: list[tuple[str, int]]) -> None:
    """Print all measurement results."""
    print(f"\nMeasurement results ({len(results)} shots):")
    print("-" * 20)
    for k, (basis, outcome) in enumerate(results, start=1):
        sign = "+" if outcome == 1 else "-"
        print(f"  Shot {k:>4d}: {basis}  {sign}1")
    print("-" * 20)


def save_results(results: list[tuple[str, int]], filename: str = "measurement_data.txt") -> None:
    """Save measurement results to a text file (one 'basis outcome' per line)."""
    with open(filename, 'w') as f:
        for basis, outcome in results:
            f.write(f"{basis} {outcome}\n")
    print(f"\nResults saved to '{filename}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 40)
    print("Random Pauli Measurement Encoder")
    print("Fixed state: rho = |0><0|")
    print("=" * 40)

    K = int(input("\nEnter number of measurements: "))

    results = run_measurements(K)

    print_results(results)
    save_results(results)


if __name__ == "__main__":
    main()