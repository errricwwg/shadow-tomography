# Shadow Tomography

A single-qubit shadow tomography implementation using NumPy.

## Overview

This project simulates classical shadow tomography for a fixed quantum state `rho = |0><0|`.
It is split into two parts:

- **Measurement Encoder** (`measurement_encoder.py`): Performs random Pauli measurements and saves raw data.
- **Shadow Estimator** (`shadow_estimator.py`): Reads the measurement data and estimates the expectation value of a Hamiltonian.

## Files

| File | Description |
|------|-------------|
| `measurement_encoder.py` | Randomly measures the qubit in X, Y, or Z basis using Born rule probabilities and saves results to `measurement_data.txt` |
| `shadow_estimator.py` | Reads `measurement_data.txt`, takes Hamiltonian coefficients a, b, c as input, and computes the shadow estimate of `<H>` |
| `measurement_data.txt` | Raw measurement output: one `basis outcome` pair per line (local only, not tracked by git) |

## Usage

### Step 1 — Generate measurement data

```bash
python measurement_encoder.py
```

Enter the number of measurements `K` when prompted.

### Step 2 — Run the estimator

```bash
python shadow_estimator.py
```

Enter coefficients `a`, `b`, `c` for the Hamiltonian `H = aX + bY + cZ`.

## Quantum Details

- Fixed state: `rho = |0><0|`
- Pauli bases: X, Y, Z (chosen uniformly at random)
- Inverse shadow channel: `M^{-1}(sigma_k) = 3 * sigma_k`
- Estimator: `<H> ≈ (1/K) * sum_k Tr(3 * sigma_k * H)`
