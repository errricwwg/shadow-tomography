"""
train.py — ShadowGPT: generative classical-shadow training and evaluation.

Learning objective
------------------
Train a decoder-only GPT to model the conditional distribution

    p_model(b_0, …, b_{n-1} | P_0, …, P_{n-1}, g)

where:
    g       = Hamiltonian conditioning tokens (model family + parameters)
    P_i     = Pauli basis choice for qubit i  (X, Y, or Z)
    b_i     = binary measurement outcome for qubit i  (0 or 1)

Training sequence format (one measurement per sequence):

    [BOS] [MODEL_FAM, params…, SEP]  [GX|GY|GZ  GO0|GO1] × n_qubits  [EOS]
          └─── conditioning g ───┘   └───── measurement (P, b) ──────┘

At inference, the model generates synthetic shadow measurements (P, b) for a
target Hamiltonian g.  These are fed to ShadowProcessor for classical-shadow
estimation of physical observables (magnetization, correlations, etc.).

Pipeline
--------
    quantum state ψ(g)
        → ShadowCollector  (random Pauli bases + outcomes)
        → tokenize as [BOS, g, GX, GO0, GY, GO1, …, EOS]
        → ShadowGPT training (cross-entropy on GO0/GO1 tokens)
        → generate synthetic shadows autoregressively
        → ShadowProcessor  (estimate magnetization, correlations, …)

Usage
-----
    cd "shadow GPT/code/shadows"
    python train.py

    # Quick smoke test:
    python train.py --n-states 20 --n-shadows 20 --n-epochs 5 --n-qubits 4

    # TFIM ground states, modest scale:
    python train.py --n-states 200 --n-shadows 50 --n-epochs 30

    # XXZ model:
    python train.py --hamiltonian-family xxz --n-states 200 --n-shadows 50

    # Multi-seed evaluation:
    python train.py --n-states 100 --n-shadows 50 --n-epochs 30 --multi-seed 3
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Pauli matrices — used locally for exact observable computation
_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0,   1 ], [1,   0 ]], dtype=complex)
_Y  = np.array([[0, -1j ], [1j,  0 ]], dtype=complex)
_Z  = np.array([[1,   0 ], [0,  -1 ]], dtype=complex)

# ── path setup ────────────────────────────────────────────────────────────────
# This file lives inside shadows/, so we go up one level to reach code/ where
# the shadows package itself lives.
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shadows.config import create_default_config
from shadows.collector import ShadowCollector
from shadows.processor import ShadowProcessor
from shadows.hamiltonians import (
    build_hamiltonian_spec,
    build_tfim_dense_matrix         as _build_tfim_matrix,
    build_ising_general_dense_matrix as _build_ising_general_matrix,
    build_xxz_dense_matrix          as _build_xxz_matrix,
    build_heisenberg_dense_matrix   as _build_heisenberg_matrix,
)
from shadows.tokenization import (
    create_default_tokenizer,
    add_hamiltonian_conditioning,
    encode_hamiltonian_prefix,
    add_multi_hamiltonian_conditioning,
    encode_multi_hamiltonian_prefix,
    add_generative_tokens,
    build_generative_sequence,
)
from shadows.datasets import GenerativeShadowDataset
from shadows.model import ShadowGPT, GPTConfig, create_gpt_from_tokenizer


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def set_seed(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _exact_pair_averaged_zz(
    state_vector: np.ndarray,
    n_qubits: int,
    max_distance: int,
) -> float:
    """
    Exact average of <Z_i Z_j> over all pairs with 1 <= j-i <= max_distance.

    This mirrors ShadowProcessor.estimate_correlations(), which averages over
    pairs (i, j) satisfying 1 <= j-i <= config.correlation_length.  The
    benchmark must use the same pair set as the processor so that the exact
    reference and the shadow estimate measure the same quantity.

    Args:
        state_vector: Normalised 2^n-dimensional complex state vector.
        n_qubits:     System size.
        max_distance: Maximum allowed separation j-i (inclusive).
                      Pass eval_correlation_length from the evaluation section.

    Returns:
        Scalar float — pair-averaged <Z_i Z_j>.

    Raises:
        ValueError: if no valid pairs exist (e.g. n_qubits < 2 or max_distance < 1).
    """
    pairs = [
        (i, j)
        for i in range(n_qubits)
        for j in range(i + 1, n_qubits)
        if 1 <= j - i <= max_distance
    ]
    if not pairs:
        raise ValueError(
            f"No valid pairs for n_qubits={n_qubits}, max_distance={max_distance}."
        )
    total = 0.0
    for i, j in pairs:
        ZZ = _kron_op(_Z, i, n_qubits) @ _kron_op(_Z, j, n_qubits)
        total += float(np.real(state_vector.conj() @ ZZ @ state_vector))
    return total / len(pairs)


def _exact_renyi2(
    state_vector: np.ndarray,
    n_qubits: int,
    n_subsystem: int,
) -> float:
    """
    Exact Rényi-2 entropy S2(rho_A) = -log Tr(rho_A^2) for subsystem A.

    Subsystem A is taken as the first n_subsystem qubits (indices 0..n_subsystem-1).
    The state vector is reshaped to (dim_A, dim_B) so that the first index
    runs over A and the second over B.  The reduced density matrix is
    rho_A = psi_matrix @ psi_matrix†, and the purity is Tr(rho_A^2).

    Args:
        state_vector: Normalised 2^n-dimensional complex state vector.
        n_qubits:     Total number of qubits.
        n_subsystem:  Number of qubits in subsystem A (1 <= n_subsystem < n_qubits).

    Returns:
        S2(rho_A) as a float.  Returns 0.0 (pure state) if n_subsystem == 0
        or n_subsystem >= n_qubits.
    """
    k = int(np.clip(n_subsystem, 0, n_qubits))
    if k == 0 or k >= n_qubits:
        return 0.0
    # Reshape: first index = A (dim 2^k), second index = B (dim 2^(n-k))
    psi = state_vector.reshape(2 ** k, 2 ** (n_qubits - k))
    rho_A   = psi @ psi.conj().T          # (2^k, 2^k) reduced density matrix
    purity  = float(np.real(np.trace(rho_A @ rho_A)))
    purity  = min(max(purity, 1e-10), 1.0)   # clamp before log
    return float(-np.log(purity))


def random_state_vector(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a normalised random pure state in the 2^n-dimensional Hilbert space.

    Both real and imaginary parts are drawn i.i.d. from N(0,1), then
    normalised.  The resulting distribution is Haar-uniform over pure states.
    """
    dim = 2 ** n_qubits
    sv = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return sv / np.linalg.norm(sv)


def _kron_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed single-qubit `op` on `qubit` into the full 2^n x 2^n space."""
    mats = [_I2] * n_qubits
    mats[qubit] = op
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _tfim_energy(state_vector: np.ndarray, H_matrix: np.ndarray) -> float:
    """
    Return the exact TFIM energy <psi|H|psi> for a normalised state vector.

    Parameters
    ----------
    state_vector : np.ndarray, shape (2**n,), complex.
    H_matrix     : np.ndarray, shape (2**n, 2**n), complex  — dense Hamiltonian matrix.
    """
    return float(np.real(state_vector.conj() @ H_matrix @ state_vector))


def _linspace_grid(lo: float, hi: float, step: float, decimals: int = 2) -> np.ndarray:
    """Evenly-spaced grid from lo to hi (inclusive) at the given step, rounded."""
    n = round((hi - lo) / step) + 1
    return np.round(np.linspace(lo, hi, n), decimals)



def generate_dataset(
    n_states: int,
    n_qubits: int,
    n_shadows_per_state: int,
    seed: int,
    tfim_J: float,
    tfim_h: float,
    state_family: str = "tfim_ground",
    tfim_h_min: float = 0.1,
    tfim_h_max: float = 2.0,
    excited_max_level: int = 4,
    # ── Multi-family args ────────────────────────────────────────────────────
    hamiltonian_family: str = "tfim",
    ising_J: float = 1.0,
    ising_hx_min: float = 0.1,
    ising_hx_max: float = 2.0,
    ising_hz_grid: Optional[List[float]] = None,
    xxz_J: float = 1.0,
    xxz_delta_min: float = 0.0,
    xxz_delta_max: float = 2.0,
    heis_J_min: float = 0.5,
    heis_J_max: float = 2.0,
) -> tuple:
    """
    Generate quantum states and classical shadow measurements.

    For TFIM families, h is always sampled from a discrete 0.1-step grid so
    conditioning token names match exactly.

    state_family (TFIM only)
    ------------------------
    "tfim_ground"  Ground states of H(J, h) for h ~ discrete grid.
    "tfim_excited" Like tfim_ground but eigenstate index k ~ Uniform{0..K-1}.
    "haar"         Haar-random states (no meaningful Hamiltonian conditioning).

    hamiltonian_family
    ------------------
    "tfim"         Delegates to state_family logic above.
    "ising_general" Ground states of H = -J ΣZZ - hx ΣX - hz ΣZ (OBC).
    "xxz"          Ground states of H = J Σ(XX+YY+delta ZZ) (OBC).
    "heisenberg"   Ground states of H = J Σ(XX+YY+ZZ) (OBC).

    Returns
    -------
    merged_collector : ShadowCollector
    targets          : np.ndarray, shape (n_states * n_shadows_per_state, 3)
                       columns: [magnetization_est, correlations_est, exact_energy]
    h_per_state      : np.ndarray (n_states,) for TFIM families, else None.
    multi_params     : dict {param_name: np.ndarray} for non-TFIM families, else None.
    """
    rng = np.random.default_rng(seed)
    total = n_states * n_shadows_per_state

    # ── Hamiltonian setup ─────────────────────────────────────────────────────
    _multi_params = None   # populated for non-TFIM families

    if hamiltonian_family == "tfim":
        if state_family in ("tfim_ground", "tfim_excited"):
            # Always sample from the discrete grid so conditioning token names match.
            h_grid = np.round(
                np.linspace(tfim_h_min, tfim_h_max,
                            round((tfim_h_max - tfim_h_min) / 0.1) + 1), 1
            )
            h_values = rng.choice(h_grid, size=n_states)
            h_desc = f"grid {h_grid[0]:.1f}…{h_grid[-1]:.1f} (step 0.1)"

            if state_family == "tfim_ground":
                print(f"\nState family : TFIM ground states")
                print(f"  J = {tfim_J} (fixed),  h ~ {h_desc}")
                print(f"  Energy target = ground-state energy of each state's own H(J,h)")
                level_choices = None
            else:
                print(f"\nState family : TFIM excited states  (max level = {excited_max_level - 1})")
                print(f"  J = {tfim_J} (fixed),  h ~ {h_desc}")
                print(f"  Energy target = k-th eigenvalue of H(J,h), k ~ Uniform{{0..{excited_max_level-1}}}")
                level_choices = rng.integers(0, excited_max_level, size=n_states)
            H_matrix = None
        else:  # haar
            print(f"\nState family : Haar-random pure states")
            print(f"Building TFIM Hamiltonian  (n={n_qubits}, J={tfim_J}, h={tfim_h}, OBC) ...")
            H_matrix = _build_tfim_matrix(n_qubits, J=tfim_J, h=tfim_h)
            print(f"  H shape: {H_matrix.shape}  "
                  f"(Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)})")
            h_values = None
            level_choices = None

    elif hamiltonian_family == "ising_general":
        hx_grid  = _linspace_grid(ising_hx_min, ising_hx_max, 0.1)
        _hz_grid = np.array(ising_hz_grid if ising_hz_grid is not None
                            else [0.00, 0.20, 0.50, 1.00])
        hx_values = rng.choice(hx_grid, size=n_states)
        hz_values = rng.choice(_hz_grid, size=n_states)
        print(f"\nHamiltonian  : General Ising  (OBC)")
        print(f"  J = {ising_J} (fixed)")
        print(f"  hx ~ grid {hx_grid[0]:.2f}…{hx_grid[-1]:.2f} (step 0.1)")
        print(f"  hz ~ {{{', '.join(f'{v:.2f}' for v in _hz_grid)}}}")
        print(f"  State family : ground states")
        _multi_params = {
            "J":  np.full(n_states, ising_J),
            "HX": hx_values,
            "HZ": hz_values,
        }
        H_matrix = None; h_values = None; level_choices = None

    elif hamiltonian_family == "xxz":
        delta_grid   = _linspace_grid(xxz_delta_min, xxz_delta_max, 0.25)
        delta_values = rng.choice(delta_grid, size=n_states)
        print(f"\nHamiltonian  : XXZ  (OBC)")
        print(f"  J = {xxz_J} (fixed)")
        print(f"  delta ~ grid {delta_grid[0]:.2f}…{delta_grid[-1]:.2f} (step 0.25)")
        print(f"  State family : ground states")
        _multi_params = {
            "J":     np.full(n_states, xxz_J),
            "DELTA": delta_values,
        }
        H_matrix = None; h_values = None; level_choices = None

    elif hamiltonian_family == "heisenberg":
        J_grid   = _linspace_grid(heis_J_min, heis_J_max, 0.25)
        J_values = rng.choice(J_grid, size=n_states)
        print(f"\nHamiltonian  : Heisenberg  (OBC, delta=1)")
        print(f"  J ~ grid {J_grid[0]:.2f}…{J_grid[-1]:.2f} (step 0.25)")
        print(f"  State family : ground states")
        _multi_params = {"J": J_values}
        H_matrix = None; h_values = None; level_choices = None

    else:
        raise ValueError(f"Unknown hamiltonian_family {hamiltonian_family!r}.")

    all_measurements = []
    all_targets      = []

    print(f"\nGenerating {n_states} states x {n_shadows_per_state} shadows "
          f"= {total} measurements ...")
    t0 = time.time()

    for state_idx in range(n_states):
        # ── 1. State vector ───────────────────────────────────────────────────
        if hamiltonian_family == "tfim":
            if state_family in ("tfim_ground", "tfim_excited"):
                h_s = float(h_values[state_idx])
                H_s = _build_tfim_matrix(n_qubits, J=tfim_J, h=h_s)
                eigvals, eigvecs = np.linalg.eigh(H_s)
                k = int(level_choices[state_idx]) if state_family == "tfim_excited" else 0
                sv = eigvecs[:, k]
                state_energy = float(eigvals[k])
            else:  # haar
                sv = random_state_vector(n_qubits, rng)
                state_energy = _tfim_energy(sv, H_matrix)
        else:
            # Non-TFIM: always the ground state of the per-state Hamiltonian.
            si = state_idx
            if hamiltonian_family == "ising_general":
                H_s = _build_ising_general_matrix(
                    n_qubits,
                    J=float(_multi_params["J"][si]),
                    hx=float(_multi_params["HX"][si]),
                    hz=float(_multi_params["HZ"][si]),
                )
            elif hamiltonian_family == "xxz":
                H_s = _build_xxz_matrix(
                    n_qubits,
                    J=float(_multi_params["J"][si]),
                    delta=float(_multi_params["DELTA"][si]),
                )
            elif hamiltonian_family == "heisenberg":
                H_s = _build_heisenberg_matrix(
                    n_qubits,
                    J=float(_multi_params["J"][si]),
                )
            eigvals, eigvecs = np.linalg.eigh(H_s)
            sv = eigvecs[:, 0]
            state_energy = float(eigvals[0])

        # ── 2. Collect shadows ────────────────────────────────────────────────
        cfg = create_default_config(
            n_qubits=n_qubits,
            n_shadows=n_shadows_per_state,
            measurement_basis="random",
        )
        cfg.seed = int(rng.integers(0, 2**31))
        collector = ShadowCollector(cfg)
        collector.sample_dense(sv)

        # ── 3. Build target vector: [magnetization, correlations, energy] ───────
        proc_cfg = create_default_config(
            n_qubits=n_qubits,
            n_shadows=n_shadows_per_state,
        )
        proc_cfg.median_of_means = False
        processor = ShadowProcessor(proc_cfg)
        estimates = processor.process_shadows(collector)
        target = np.array([
            estimates["magnetization"].estimate,   # col 0
            estimates["correlations"].estimate,    # col 1
            state_energy,                          # col 2
        ], dtype=np.float32)

        # ── 4. Broadcast target across all measurements from this state ───────
        all_measurements.extend(collector.measurements)
        all_targets.append(np.tile(target, (n_shadows_per_state, 1)))

        if (state_idx + 1) % max(1, n_states // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {state_idx + 1}/{n_states} states  "
                  f"({elapsed:.1f}s elapsed)")

    print(f"Dataset generation complete ({time.time() - t0:.1f}s total).")

    merged_cfg = create_default_config(n_qubits=n_qubits, n_shadows=total)
    merged_collector = ShadowCollector(merged_cfg)
    merged_collector.measurements = all_measurements

    targets = np.vstack(all_targets).astype(np.float32)

    assert len(merged_collector.measurements) == total
    h_per_state = (
        h_values
        if (hamiltonian_family == "tfim"
            and state_family in ("tfim_ground", "tfim_excited"))
        else None
    )
    return merged_collector, targets, h_per_state, _multi_params


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Training entry point
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _build_family_param_grids(args: argparse.Namespace) -> dict:
    """
    Return the {param_name: [grid_values]} dict for the non-TFIM family in args.

    Used both to build the tokenizer vocabulary (via add_multi_hamiltonian_conditioning)
    and, implicitly, to determine the sampling grids inside generate_dataset.
    Both must use identical step sizes and rounding so that every sampled value
    has a corresponding token in the vocabulary.
    """
    fam = args.hamiltonian_family
    if fam == "ising_general":
        hx_grid  = list(_linspace_grid(args.tfim_h_min, args.tfim_h_max, 0.1))
        hz_grid  = [0.00, 0.20, 0.50, 1.00]
        return {"J": [round(args.ising_J, 2)], "HX": hx_grid, "HZ": hz_grid}
    elif fam == "xxz":
        delta_grid = list(_linspace_grid(args.xxz_delta_min, args.xxz_delta_max, 0.25))
        return {"J": [round(args.xxz_J, 2)], "DELTA": delta_grid}
    elif fam == "heisenberg":
        J_grid = list(_linspace_grid(args.heis_J_min, args.heis_J_max, 0.25))
        return {"J": J_grid}
    else:
        raise ValueError(f"_build_family_param_grids: unknown family {fam!r}.")


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# ShadowGPT pipeline
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

@torch.no_grad()
def generate_shadows_from_gpt(
    model: ShadowGPT,
    tokenizer,
    h_prefix: list,
    n_shadows: int,
    n_qubits: int,
    device: str,
    temperature: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> ShadowCollector:
    """
    Autoregressively generate synthetic shadow measurements from a ShadowGPT.

    For each of the n_shadows measurements:
        1. Sample Pauli basis P = (p_0, …, p_{n-1}) uniformly from {X,Y,Z}^n.
        2. Build prefix [BOS, g…, GX|GY|GZ for p_0].
        3. Sample outcome b_0 from p(b_0 | prefix), restricted to {GO0, GO1}.
        4. Append [b_0, GX|GY|GZ for p_1] and repeat until all n_qubits done.
        5. Store (P, b) as one shadow measurement.

    Args:
        model:       ShadowGPT in eval mode, moved to device.
        tokenizer:   ShadowTokenizer with GX/GY/GZ/GO0/GO1 in vocab.
        h_prefix:    Conditioning token IDs (from encode_hamiltonian_prefix or
                     encode_multi_hamiltonian_prefix).
        n_shadows:   Number of synthetic measurements to generate.
        n_qubits:    System size.
        device:      Torch device string.
        temperature: Sampling temperature (1.0 = unmodified softmax).
        rng:         numpy Generator for basis sampling.  None = default rng.

    Returns:
        ShadowCollector containing n_shadows measurements.
    """
    from shadows.config import create_default_config

    if rng is None:
        rng = np.random.default_rng()

    model.eval()
    bos = tokenizer.special_tokens["BOS"]

    # Precompute allowed token ID lists.
    basis_ids   = [tokenizer.vocab[f"G{'XYZ'[b]}"] for b in range(3)]
    outcome_ids = [tokenizer.vocab["GO0"], tokenizer.vocab["GO1"]]

    all_bases:    List[List[int]] = []
    all_outcomes: List[List[int]] = []

    for _ in range(n_shadows):
        # 1. Sample random Pauli basis for every qubit.
        P = rng.integers(0, 3, size=n_qubits).tolist()   # 0=X,1=Y,2=Z

        # 2. Start with [BOS, g_prefix..., basis_token_for_qubit_0]
        current = [bos] + h_prefix + [basis_ids[P[0]]]
        outcomes: List[int] = []

        # 3. Autoregressively generate outcomes for each qubit.
        for q in range(n_qubits):
            ids_t = torch.tensor([current], dtype=torch.long, device=device)
            o_tok = model.generate_next_token(
                ids_t, temperature=temperature, allowed_ids=outcome_ids
            )
            # Decode outcome value: GO0 → 0, GO1 → 1
            o_val = outcome_ids.index(o_tok)
            outcomes.append(o_val)
            current.append(o_tok)
            # Append next basis token (if not last qubit).
            if q < n_qubits - 1:
                current.append(basis_ids[P[q + 1]])

        all_bases.append(P)
        all_outcomes.append(outcomes)

    # Pack into a ShadowCollector.
    cfg = create_default_config(n_qubits=n_qubits, n_shadows=n_shadows)
    collector = ShadowCollector(cfg)
    from shadows.collector import ShadowMeasurement
    collector.measurements = [
        ShadowMeasurement(
            basis=np.array(b, dtype=np.int32),
            outcome=np.array(o, dtype=np.int32),
        )
        for b, o in zip(all_bases, all_outcomes)
    ]
    return collector


def _generative_tokenizer_setup(args: argparse.Namespace, tokenizer):
    """
    Add Hamiltonian conditioning tokens + generative (GX/GY/GZ/GO0/GO1) tokens
    to the tokenizer, then set max_sequence_length for the generative sequence format:

        [BOS] [h_prefix…] [GX|GY|GZ  GO0|GO1] × n_qubits [EOS]

    Returns h_prefix_len (= len(h_prefix), not counting BOS).
    """
    if args.hamiltonian_family == "tfim":
        h_grid = np.round(
            np.linspace(args.tfim_h_min, args.tfim_h_max,
                        round((args.tfim_h_max - args.tfim_h_min) / 0.1) + 1), 1
        )
        add_hamiltonian_conditioning(tokenizer, tfim_J=args.tfim_J, tfim_h_values=h_grid)
        h_prefix_len = 4   # MODEL_TFIM, J_xxx, H_xxx, SEP
    else:
        param_grids  = _build_family_param_grids(args)
        add_multi_hamiltonian_conditioning(tokenizer, args.hamiltonian_family, param_grids)
        n_params     = len(param_grids)
        h_prefix_len = 1 + n_params + 1   # MODEL_FAM + params + SEP

    add_generative_tokens(tokenizer)

    # Sequence length: BOS + h_prefix + 2*n_qubits*(GX/GY/GZ + GO0/GO1) + EOS
    tokenizer.config.max_sequence_length = 1 + h_prefix_len + 2 * args.n_qubits + 1
    return h_prefix_len


def train(args: argparse.Namespace) -> dict:
    """
    ShadowGPT training and evaluation pipeline.

    Trains a decoder-only GPT to model the joint distribution:
        p(b_0, …, b_{n-1} | P_0, …, P_{n-1}, g)
    autoregressively — measurement outcomes b given Pauli bases P and
    Hamiltonian conditioning tokens g.

    Evaluation generates synthetic shadows from the trained model and feeds
    them to ShadowProcessor to estimate physical observables; estimates are
    compared to exact values (from state vectors) as a measure of shadow quality.

    Returns a metrics dict: test_ce, test_acc, real/gen MAEs for magnetization
    and correlations, and the model/real MAE ratio.
    """
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"ShadowGPT — generative shadow model")
    print(f"{'='*60}")
    print(f"Device      : {device}")
    print(f"Ham. family : {args.hamiltonian_family}")
    print(f"n_qubits    : {args.n_qubits}")
    print(f"n_states    : {args.n_states}")
    print(f"n_shadows   : {args.n_shadows_per_state}  per state")
    print(f"n_epochs    : {args.n_epochs}")
    print(f"d_model     : {args.d_model}")

    # ── 1. Generate dataset ───────────────────────────────────────────────────
    # For the generative model, --state-family must be tfim_ground (default) or
    # another conditioned family; Haar-random states are not useful here because
    # they have no meaningful conditioning token.
    gen_state_family = args.state_family
    if args.hamiltonian_family == "tfim" and gen_state_family == "haar":
        gen_state_family = "tfim_ground"
        print(f"  (auto-switching state-family to 'tfim_ground' for generative mode)")

    collector, targets, h_per_state, multi_params = generate_dataset(
        n_states=args.n_states,
        n_qubits=args.n_qubits,
        n_shadows_per_state=args.n_shadows_per_state,
        seed=args.seed,
        tfim_J=args.tfim_J,
        tfim_h=args.tfim_h,
        state_family=gen_state_family,
        tfim_h_min=args.tfim_h_min,
        tfim_h_max=args.tfim_h_max,
        excited_max_level=args.excited_max_level,
        hamiltonian_family=args.hamiltonian_family,
        ising_J=args.ising_J,
        ising_hx_min=args.tfim_h_min,
        ising_hx_max=args.tfim_h_max,
        xxz_J=args.xxz_J,
        xxz_delta_min=args.xxz_delta_min,
        xxz_delta_max=args.xxz_delta_max,
        heis_J_min=args.heis_J_min,
        heis_J_max=args.heis_J_max,
    )
    S         = args.n_shadows_per_state
    n_states  = args.n_states
    n_qubits  = args.n_qubits
    total_meas = len(collector.measurements)

    # Per-state exact values: targets are broadcast (same value repeated S times).
    # Take the first measurement per state for exact targets.
    state_targets = targets[::S]           # (n_states, 3): [mag, corr, energy]
    state_exact_energy = state_targets[:, 2]

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = create_default_tokenizer(n_qubits=n_qubits)
    h_prefix_len = _generative_tokenizer_setup(args, tokenizer)
    print(f"\nTokenizer   : {tokenizer}")
    print(f"Sequence len: {tokenizer.config.max_sequence_length} tokens  "
          f"(1 BOS + {h_prefix_len} prefix + {2*n_qubits} meas + 1 EOS)")

    # Precompute per-state conditioning prefixes.
    state_prefixes: List[list] = []
    for s in range(n_states):
        if args.hamiltonian_family == "tfim":
            pfx = encode_hamiltonian_prefix(tokenizer, J=args.tfim_J,
                                            h=float(h_per_state[s]))
        else:
            p = {k: float(v[s]) for k, v in multi_params.items()}
            pfx = encode_multi_hamiltonian_prefix(tokenizer, args.hamiltonian_family, p)
        state_prefixes.append(pfx)

    # ── 3. Build generative sequences ─────────────────────────────────────────
    # One sequence per shadow measurement: [BOS, g…, GX/Y/Z, GO0/1, …, EOS].
    all_seqs: List[List[int]] = []
    for state_idx in range(n_states):
        pfx = state_prefixes[state_idx]
        for meas_idx in range(S):
            meas = collector.measurements[state_idx * S + meas_idx]
            seq  = build_generative_sequence(
                tokenizer, pfx,
                np.array(meas.basis, dtype=np.int32),
                np.array(meas.outcome, dtype=np.int32),
            )
            all_seqs.append(seq)

    # ── 4. State-level train / val / test split ───────────────────────────────
    split_rng    = np.random.default_rng(args.seed)
    state_order  = np.arange(n_states)
    split_rng.shuffle(state_order)
    n_train_s    = int(n_states * 0.8)
    n_val_s      = int(n_states * 0.1)
    train_states = set(state_order[:n_train_s])
    val_states   = set(state_order[n_train_s:n_train_s + n_val_s])
    test_states  = [s for s in range(n_states) if s not in train_states and s not in val_states]

    state_of_seq = np.repeat(np.arange(n_states), S)
    train_seqs = [all_seqs[i] for i, s in enumerate(state_of_seq) if s in train_states]
    val_seqs   = [all_seqs[i] for i, s in enumerate(state_of_seq) if s in val_states]
    test_seqs  = [all_seqs[i] for i, s in enumerate(state_of_seq)
                  if s not in train_states and s not in val_states]

    outcome_ids = {tokenizer.vocab["GO0"], tokenizer.vocab["GO1"]}
    train_ds = GenerativeShadowDataset(train_seqs, tokenizer, outcome_ids)
    val_ds   = GenerativeShadowDataset(val_seqs,   tokenizer, outcome_ids)
    test_ds  = GenerativeShadowDataset(test_seqs,  tokenizer, outcome_ids)

    print(f"\nDataset split : {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test"
          f"  (state-level: {n_train_s} / {n_val_s} / {len(test_states)} states)")

    pin = (device == "cuda")
    train_loader = train_ds.get_dataloader(batch_size=args.batch_size, shuffle=True,  pin_memory=pin)
    val_loader   = val_ds.get_dataloader(  batch_size=args.batch_size, shuffle=False, pin_memory=pin)
    test_loader  = test_ds.get_dataloader( batch_size=args.batch_size, shuffle=False, pin_memory=pin)

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model = create_gpt_from_tokenizer(
        tokenizer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)
    print(f"\nModel       : ShadowGPT (decoder-only, causal attention)")
    print(f"Parameters  : {model.count_parameters():,}")
    print(f"Architecture: d_model={args.d_model}  n_heads={args.n_heads}  "
          f"n_layers={args.n_layers}  d_ff={args.d_ff}")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6
    )
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_gpt.pt")

    best_val_loss  = float("inf")
    best_state     = None
    history: dict  = {"train_loss": [], "val_loss": []}

    def _run_epoch(loader, train: bool) -> float:
        model.train(train)
        total_loss = 0.0
        total_tok  = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                ids    = batch["input_ids"].to(device)   # (B, L)
                labels = batch["labels"].to(device)      # (B, L)
                lmask  = batch["loss_mask"].to(device)   # (B, L)

                logits = model(ids)                              # (B, L, V)
                B, L, V = logits.shape
                loss_flat = ce_loss(
                    logits.reshape(B * L, V),
                    labels.reshape(B * L),
                )                                               # (B*L,)
                # Only average over outcome positions in loss_mask.
                mask_flat = lmask.reshape(B * L).float()
                n_active  = mask_flat.sum().clamp(min=1)
                loss      = (loss_flat * mask_flat).sum() / n_active

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                total_loss += loss.item() * n_active.item()
                total_tok  += n_active.item()
        return total_loss / max(total_tok, 1)

    print(f"\n{'─'*60}")
    print(f"Training for {args.n_epochs} epochs  (best checkpoint → {checkpoint_path})")
    print(f"{'─'*60}")
    t_train = time.time()

    print_every = max(1, args.n_epochs // 10)
    for epoch in range(1, args.n_epochs + 1):
        tr_loss = _run_epoch(train_loader, train=True)
        va_loss = _run_epoch(val_loader,   train=False)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "config": model.config}, checkpoint_path)

        if epoch % print_every == 0:
            marker = "  *best" if va_loss == best_val_loss else ""
            print(f"Epoch {epoch:4d}/{args.n_epochs}  "
                  f"train={tr_loss:.4f}  val={va_loss:.4f}{marker}")

    print(f"\nTraining finished in {time.time() - t_train:.1f}s")
    print(f"Best val loss : {best_val_loss:.4f}")

    # ── 7. Restore best and compute token accuracy on test set ───────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    n_correct = 0
    n_outcome = 0
    te_loss   = 0.0
    te_tok    = 0
    with torch.no_grad():
        for batch in test_loader:
            ids    = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lmask  = batch["loss_mask"].to(device)
            logits = model(ids)
            B, L, V = logits.shape
            loss_flat = ce_loss(logits.reshape(B*L, V), labels.reshape(B*L))
            mask_flat = lmask.reshape(B*L).float()
            n_active  = mask_flat.sum().clamp(min=1)
            te_loss  += (loss_flat * mask_flat).sum().item()
            te_tok   += n_active.item()
            # Outcome token accuracy
            preds    = logits.argmax(dim=-1)     # (B, L)
            active   = lmask.bool()              # (B, L)
            n_correct += (preds[active] == labels[active]).sum().item()
            n_outcome += active.sum().item()

    test_ce   = te_loss / max(te_tok, 1)
    test_acc  = n_correct / max(n_outcome, 1)
    print(f"\n{'─'*60}")
    print(f"Test set  (n_sequences={len(test_ds)})")
    print(f"{'─'*60}")
    print(f"  CE loss (outcome tokens) : {test_ce:.4f}")
    print(f"  Outcome token accuracy   : {test_acc*100:.1f}%")

    # ── 8. Shadow-quality evaluation ─────────────────────────────────────────
    # Evaluate magnetization, correlations, and (for TFIM) energy from
    # (a) real shadows and (b) model-generated shadows; compare both to exact
    # state-vector values computed by diagonalisation.
    #
    # Hamiltonian abstraction:
    # build_hamiltonian_spec() returns a HamiltonianSpec with two fields:
    #   .dense_matrix      — used for np.linalg.eigh() (exact state vectors)
    #   .pauli_hamiltonian — used for ShadowProcessor.estimate_energy();
    #                        None for families without a Pauli builder yet
    #
    # Correlation alignment:
    # ShadowProcessor.estimate_correlations() averages over pairs (i, j)
    # with 1 <= j-i <= config.correlation_length.  eval_correlation_length
    # controls both the processor config AND _exact_pair_averaged_zz(),
    # so both sides always measure the same pair set.
    eval_correlation_length = 1   # nearest-neighbour only; matches processor

    n_eval_shadows = args.n_shadows_per_state
    gen_rng        = np.random.default_rng(args.seed + 9999)

    real_mag_maes:     List[float] = []
    gen_mag_maes:      List[float] = []
    real_corr_maes:    List[float] = []
    gen_corr_maes:     List[float] = []
    real_energy_maes:  List[float] = []
    gen_energy_maes:   List[float] = []
    real_renyi2_maes:  List[float] = []
    gen_renyi2_maes:   List[float] = []

    # Energy estimation is available whenever build_hamiltonian_spec() returns a
    # non-None pauli_hamiltonian.  We probe with a minimal 2-qubit spec (no
    # diagonalisation, cheap) purely to check availability before the loop.
    _probe = build_hamiltonian_spec(args.hamiltonian_family, 2)
    energy_eval_enabled = _probe.pauli_hamiltonian is not None

    print(f"\n{'─'*60}")
    print(f"Shadow quality evaluation  (n_eval_shadows={n_eval_shadows}, "
          f"n_test_states={len(test_states)})")
    print(f"  Metric            = MAE vs exact observables (state vectors)")
    print(f"  Correlation pairs = distance 1..{eval_correlation_length} "
          f"(processor correlation_length={eval_correlation_length})")
    if energy_eval_enabled:
        print(f"  Energy estimation : enabled  ({args.hamiltonian_family} Pauli Hamiltonian)")
    else:
        print(f"  Energy estimation : disabled "
              f"(Pauli builder not yet implemented for '{args.hamiltonian_family}')")
    print(f"{'─'*60}")

    from shadows.config import create_default_config as _cfgfn
    from shadows.collector import ShadowCollector as _SC
    from shadows.processor import ShadowProcessor

    # Conservative subsystem for Rényi-2: use 2 qubits regardless of system
    # size, keeping estimator variance manageable (variance grows exponentially
    # with n_subsystem for the paired-shot purity estimator).
    eval_renyi_n_subsystem = 2

    _proc_cfg = _cfgfn(n_qubits=n_qubits, n_shadows=n_eval_shadows)
    _proc_cfg.median_of_means    = False
    # Must match eval_correlation_length so the processor and the exact
    # benchmark average over the same set of pairs.
    _proc_cfg.correlation_length = eval_correlation_length
    # Suppress automatic Rényi-2 computation inside process_shadows(); we call
    # estimate_renyi_entropy() directly below with an explicit n_subsystem so
    # both real and generated shadows use the same subsystem size.
    _proc_cfg.renyi_entropy      = False

    for s in test_states:
        pfx = state_prefixes[s]

        # ── Recompute exact state vector via build_hamiltonian_spec() ──────────
        # spec.dense_matrix is the 2^n × 2^n Hamiltonian matrix;
        # spec.pauli_hamiltonian is the PauliPolynomial for energy estimation
        # (None for non-TFIM families until their builders are implemented).
        if args.hamiltonian_family == "tfim":
            h_s  = float(h_per_state[s])
            spec = build_hamiltonian_spec("tfim", n_qubits,
                                          J=args.tfim_J, h=h_s)
        elif args.hamiltonian_family == "ising_general":
            spec = build_hamiltonian_spec("ising_general", n_qubits,
                                          J=float(multi_params["J"][s]),
                                          hx=float(multi_params["HX"][s]),
                                          hz=float(multi_params["HZ"][s]))
        elif args.hamiltonian_family == "xxz":
            spec = build_hamiltonian_spec("xxz", n_qubits,
                                          J=float(multi_params["J"][s]),
                                          delta=float(multi_params["DELTA"][s]))
        elif args.hamiltonian_family == "heisenberg":
            spec = build_hamiltonian_spec("heisenberg", n_qubits,
                                          J=float(multi_params["J"][s]))
        else:
            spec = build_hamiltonian_spec("tfim", n_qubits,
                                          J=args.tfim_J, h=args.tfim_h)

        _, eigvecs = np.linalg.eigh(spec.dense_matrix)
        sv_s = eigvecs[:, 0]   # ground state

        # Exact magnetization: (1/n) Σ_i <Z_i>
        exact_mag = float(sum(
            np.real(sv_s.conj() @ _kron_op(_Z, i, n_qubits) @ sv_s)
            for i in range(n_qubits)
        ) / n_qubits)

        # Exact correlations: pair-averaged <Z_i Z_j> for 1 <= j-i <= eval_correlation_length.
        # _exact_pair_averaged_zz uses the same pair set as ShadowProcessor.
        exact_corr = _exact_pair_averaged_zz(sv_s, n_qubits, eval_correlation_length)

        # Exact energy from state_targets (already computed during dataset generation).
        exact_energy = float(state_exact_energy[s])

        # Real shadows for this state (up to n_eval_shadows)
        real_meas = collector.measurements[s * S: s * S + min(S, n_eval_shadows)]
        real_coll = _SC(_cfgfn(n_qubits=n_qubits, n_shadows=len(real_meas)))
        real_coll.measurements = real_meas

        # Generated shadows
        gen_coll = generate_shadows_from_gpt(
            model, tokenizer, pfx,
            n_shadows=n_eval_shadows,
            n_qubits=n_qubits,
            device=device,
            temperature=1.0,
            rng=gen_rng,
        )

        # Process both collectors — pass spec.pauli_hamiltonian so that energy
        # is estimated automatically when available (non-None for TFIM).
        real_est = ShadowProcessor(_proc_cfg).process_shadows(
            real_coll, hamiltonian=spec.pauli_hamiltonian
        )
        gen_est  = ShadowProcessor(_proc_cfg).process_shadows(
            gen_coll,  hamiltonian=spec.pauli_hamiltonian
        )

        real_mag_maes.append(abs(real_est["magnetization"].estimate - exact_mag))
        gen_mag_maes.append( abs(gen_est["magnetization"].estimate  - exact_mag))

        if "correlations" in real_est:
            real_corr_maes.append(abs(real_est["correlations"].estimate - exact_corr))
        if "correlations" in gen_est:
            gen_corr_maes.append( abs(gen_est["correlations"].estimate  - exact_corr))

        if "energy" in real_est:
            real_energy_maes.append(abs(real_est["energy"].estimate - exact_energy))
        if "energy" in gen_est:
            gen_energy_maes.append( abs(gen_est["energy"].estimate  - exact_energy))

        # Rényi-2 entropy: exact reference vs shadow estimates.
        # Both collectors use the same n_subsystem=eval_renyi_n_subsystem so the
        # comparison is apples-to-apples.  We call estimate_renyi_entropy()
        # directly (auto-computation was suppressed via _proc_cfg.renyi_entropy=False).
        exact_renyi2 = _exact_renyi2(sv_s, n_qubits, eval_renyi_n_subsystem)
        real_renyi2  = ShadowProcessor(_proc_cfg).estimate_renyi_entropy(
            real_coll, n_subsystem=eval_renyi_n_subsystem
        ).estimate
        gen_renyi2   = ShadowProcessor(_proc_cfg).estimate_renyi_entropy(
            gen_coll,  n_subsystem=eval_renyi_n_subsystem
        ).estimate
        real_renyi2_maes.append(abs(real_renyi2 - exact_renyi2))
        gen_renyi2_maes.append( abs(gen_renyi2  - exact_renyi2))

    real_mag_mae = float(np.mean(real_mag_maes))
    gen_mag_mae  = float(np.mean(gen_mag_maes))

    if real_corr_maes:
        real_c_mae = float(np.mean(real_corr_maes))
        gen_c_mae  = float(np.mean(gen_corr_maes))
    else:
        real_c_mae = gen_c_mae = float("nan")

    if real_energy_maes:
        real_energy_mae = float(np.mean(real_energy_maes))
        gen_energy_mae  = float(np.mean(gen_energy_maes))
    else:
        real_energy_mae = gen_energy_mae = float("nan")

    if real_renyi2_maes:
        real_renyi2_mae = float(np.mean(real_renyi2_maes))
        gen_renyi2_mae  = float(np.mean(gen_renyi2_maes))
    else:
        real_renyi2_mae = gen_renyi2_mae = float("nan")

    mag_ratio     = gen_mag_mae / max(real_mag_mae, 1e-12)
    corr_ratio    = (gen_c_mae / max(real_c_mae, 1e-12)
                     if not (np.isnan(real_c_mae) or real_c_mae < 1e-12)
                     else float("nan"))
    energy_ratio  = (gen_energy_mae / max(real_energy_mae, 1e-12)
                     if not (np.isnan(real_energy_mae) or real_energy_mae < 1e-12)
                     else float("nan"))
    renyi2_ratio  = (gen_renyi2_mae / max(real_renyi2_mae, 1e-12)
                     if not (np.isnan(real_renyi2_mae) or real_renyi2_mae < 1e-12)
                     else float("nan"))

    print(f"{'Metric':<28} {'Real shadows':>14} {'Model shadows':>14}")
    print(f"{'─'*58}")
    print(f"  {'Magnetization MAE':<26} {real_mag_mae:>14.5f} {gen_mag_mae:>14.5f}")
    if not np.isnan(real_c_mae):
        print(f"  {'Correlations MAE':<26} {real_c_mae:>14.5f} {gen_c_mae:>14.5f}")
    else:
        print(f"  {'Correlations MAE':<26} {'n/a':>14} {'n/a':>14}")
    if not np.isnan(real_energy_mae):
        print(f"  {'Energy MAE':<26} {real_energy_mae:>14.5f} {gen_energy_mae:>14.5f}")
    else:
        print(f"  {'Energy MAE':<26} {'n/a':>14} {'n/a':>14}")
    if not np.isnan(real_renyi2_mae):
        print(f"  {'Rényi-2 entropy MAE':<26} {real_renyi2_mae:>14.5f} {gen_renyi2_mae:>14.5f}")
    else:
        print(f"  {'Rényi-2 entropy MAE':<26} {'n/a':>14} {'n/a':>14}")

    print(f"\n  Model/Real magnetization MAE ratio : {mag_ratio:.3f}  "
          f"(~1 = model matches real shadows, < 2 = acceptable)")
    if not np.isnan(corr_ratio):
        print(f"  Model/Real correlations  MAE ratio : {corr_ratio:.3f}")
    else:
        print(f"  Model/Real correlations  MAE ratio : n/a")
    if not np.isnan(energy_ratio):
        print(f"  Model/Real energy        MAE ratio : {energy_ratio:.3f}")
    else:
        print(f"  Model/Real energy        MAE ratio : n/a")
    if not np.isnan(renyi2_ratio):
        print(f"  Model/Real Rényi-2       MAE ratio : {renyi2_ratio:.3f}")
    else:
        print(f"  Model/Real Rényi-2       MAE ratio : n/a")
    print(f"{'─'*58}")

    # ── 9. Save tokenizer ─────────────────────────────────────────────────────
    tokenizer.save_tokenizer(os.path.join(args.output_dir, "tokenizer_gpt.json"))
    print(f"\nSaved tokenizer → {args.output_dir}/tokenizer_gpt.json")
    print(f"Saved best GPT  → {checkpoint_path}")
    print(f"\nDone.")

    return {
        "test_ce":                test_ce,
        "test_acc":               test_acc,
        "real_mag_mae":           real_mag_mae,
        "gen_mag_mae":            gen_mag_mae,
        "real_corr_mae":          real_c_mae,
        "gen_corr_mae":           gen_c_mae,
        "real_energy_mae":        real_energy_mae,
        "gen_energy_mae":         gen_energy_mae,
        "mag_model_real_ratio":    mag_ratio,
        "corr_model_real_ratio":   corr_ratio,
        "energy_model_real_ratio": energy_ratio,
        "real_renyi2_mae":         real_renyi2_mae,
        "gen_renyi2_mae":          gen_renyi2_mae,
        "renyi2_model_real_ratio": renyi2_ratio,
    }


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# CLI
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ShadowGPT: autoregressive p(b|P,g) over classical shadow measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment ────────────────────────────────────────────────────────────
    g = p.add_argument_group("Experiment")
    g.add_argument("--hamiltonian-family", type=str, default="tfim",
                   dest="hamiltonian_family",
                   choices=["tfim", "ising_general", "xxz", "heisenberg"],
                   help="Spin-chain Hamiltonian family for dataset generation and "
                        "Hamiltonian conditioning tokens. "
                        "'tfim': H=-J ΣZZ -h ΣX (controlled by --state-family). "
                        "'ising_general': H=-J ΣZZ -hx ΣX -hz ΣZ. "
                        "'xxz': H=J Σ(XX+YY+delta ZZ). "
                        "'heisenberg': H=J Σ(XX+YY+ZZ). "
                        "Non-TFIM families always use ground states.")
    g.add_argument("--state-family", type=str, default="tfim_ground",
                   choices=["haar", "tfim_ground", "tfim_excited"],
                   help="(TFIM only) State-preparation mode. "
                        "'haar': Haar-random pure states (baseline); "
                        "'tfim_ground': ground states; "
                        "'tfim_excited': random low-energy eigenstates.")
    g.add_argument("--excited-max-level", type=int, default=4,
                   dest="excited_max_level",
                   help="(TFIM tfim_excited only) Number of low-energy levels to "
                        "sample from (levels 0..K-1).")

    # ── Non-TFIM family parameters ────────────────────────────────────────────
    g.add_argument("--ising-J", type=float, default=1.0, dest="ising_J",
                   help="Ising-general: ZZ coupling J (fixed).")
    g.add_argument("--xxz-J", type=float, default=1.0, dest="xxz_J",
                   help="XXZ: exchange coupling J (fixed).")
    g.add_argument("--xxz-delta-min", type=float, default=0.0, dest="xxz_delta_min",
                   help="XXZ: minimum delta.")
    g.add_argument("--xxz-delta-max", type=float, default=2.0, dest="xxz_delta_max",
                   help="XXZ: maximum delta.")
    g.add_argument("--heis-J-min", type=float, default=0.5, dest="heis_J_min",
                   help="Heisenberg: minimum J.")
    g.add_argument("--heis-J-max", type=float, default=2.0, dest="heis_J_max",
                   help="Heisenberg: maximum J.")
    g.add_argument("--tfim-h-min", type=float, default=0.1, dest="tfim_h_min",
                   help="Minimum transverse field h (TFIM ground/excited).")
    g.add_argument("--tfim-h-max", type=float, default=2.0, dest="tfim_h_max",
                   help="Maximum transverse field h. Range [0.1, 2.0] covers ordered "
                        "(h<J), critical (h~J), and disordered (h>J) phases.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("Dataset")
    g.add_argument("--n-qubits", type=int, default=6,
                   help="Number of qubits per quantum state.")
    g.add_argument("--n-states", type=int, default=100,
                   help="Number of quantum states to generate.")
    g.add_argument("--n-shadows", type=int, default=100,
                   dest="n_shadows_per_state",
                   help="Shadow measurements per state.")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--d-model", type=int, default=128,
                   help="Embedding / hidden dimension.")
    g.add_argument("--n-heads", type=int, default=4,
                   help="Number of attention heads (must divide d_model).")
    g.add_argument("--n-layers", type=int, default=4,
                   help="Number of transformer decoder layers.")
    g.add_argument("--d-ff", type=int, default=512,
                   help="Feed-forward inner dimension.")
    g.add_argument("--dropout", type=float, default=0.1)

    # ── Training ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Training")
    g.add_argument("--n-epochs", type=int, default=30,
                   help="Number of training epochs.")
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=3e-4,
                   help="AdamW learning rate.")
    g.add_argument("--weight-decay", type=float, default=1e-2,
                   help="AdamW weight decay.")
    g.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature for generative inference.")

    # ── Hamiltonian ───────────────────────────────────────────────────────────
    g = p.add_argument_group("Hamiltonian")
    g.add_argument("--tfim-J", type=float, default=1.0,
                   help="TFIM ZZ coupling constant J.")
    g.add_argument("--tfim-h", type=float, default=0.5,
                   help="TFIM transverse-field strength h.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--output-dir", type=str, default="./outputs",
                   help="Directory for checkpoints and tokenizer.")
    g.add_argument("--multi-seed", type=int, default=0,
                   dest="multi_seed",
                   help="If > 0, run this many seeds (0..N-1) and report mean±std. "
                        "--seed is ignored in multi-seed mode.")

    return p.parse_args()


def run_multi_seed(args: argparse.Namespace) -> None:
    """Run train() over seeds 0..args.multi_seed-1 and report mean±std."""
    import copy

    n_seeds = args.multi_seed
    all_results: list[dict] = []

    for seed in range(n_seeds):
        seed_args = copy.copy(args)
        seed_args.seed = seed
        seed_args.multi_seed = 0  # prevent recursion
        seed_args.output_dir = f"{args.output_dir}/seed_{seed}"
        print(f"\n{'='*60}")
        print(f"  Multi-seed run: seed {seed}/{n_seeds - 1}")
        print(f"{'='*60}")
        result = train(seed_args)
        if result is not None:
            all_results.append(result)

    if not all_results:
        return

    print(f"\n{'='*60}")
    print(f"  Multi-seed summary ({n_seeds} seeds)")
    print(f"{'='*60}")
    keys = list(all_results[0].keys())
    for k in keys:
        vals = [r[k] for r in all_results if k in r]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            print(f"  {k}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed > 0:
        run_multi_seed(args)
    else:
        train(args)
