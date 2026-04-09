"""
train.py — End-to-end training script for classical-shadow physical quantity prediction.

Pipeline
--------
    random quantum states
        → ShadowCollector (random Pauli measurements)
        → ShadowProcessor (magnetization, correlations, renyi_entropy)
        → exact TFIM energy from state vector (column 2)
        → ShadowDataModule (tokenize, split, DataLoaders)
        → ShadowTransformer (4-target regression)
        → ShadowTrainer (fit, validate, checkpoint)

Target ordering (fixed throughout):
    column 0 — magnetization    (1/n) sum_i <Z_i>       shadow estimate
    column 1 — correlations     avg <Z_i Z_{i+1}>        shadow estimate
    column 2 — energy           <H_TFIM>                 exact from state vector
    column 3 — renyi_entropy    S_2 half-chain           shadow estimate

Energy target: transverse-field Ising model (TFIM)
---------------------------------------------------
The Hamiltonian is:

    H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}   (nearest-neighbour ZZ coupling)
        -h * sum_{i=0}^{n-1} X_i            (transverse field)

with J = 1.0 (coupling), h = 0.5 (transverse field), open boundary conditions.

Why TFIM?
  - Standard benchmark for quantum many-body physics and quantum computing.
  - Contains both ZZ (two-site) and X (single-site) terms, so <H> is a genuinely
    different number from magnetization or correlations.
  - Non-redundant: for Haar-random states, Pearson r(<H_TFIM>, magnetization) ~ 0
    because the ZZ terms dominate and the X terms add an orthogonal component.
  - Exactly computable as a real scalar via sv.conj() @ H_matrix @ sv using a
    sparse matrix built once at startup.  No extra libraries needed.
  - Energy per site for 6-qubit Haar-random states spans roughly [-1.5, 0.5]
    (well-separated from magnetization's [-1, 1] range).

Implementation note
-------------------
_build_tfim_matrix(n_qubits, J, h) builds the full 2^n x 2^n sparse Hamiltonian
once per run.  For n <= 12 this is fast and cheap.  For n > 14 you should switch
to a matrix-free approach (TFIM matrix-vector product via einsum).

Broadcasting design
-------------------
ShadowDataModule maps one token sequence (one shadow measurement) to one target.
All n_shadows_per_state measurements from the same state share the same 4D target.
This teaches the model to infer physical quantities from a single measurement,
with the MSE loss averaged across measurements in each mini-batch.

Usage
-----
    cd "shadow GPT/code"
    python train.py

    # Quick smoke test (~30s on CPU):
    python train.py --n-states 20 --n-shadows 50 --n-epochs 5

    # Larger run:
    python train.py --n-states 500 --n-shadows 200 --n-qubits 8 --d-model 256

    # Change coupling constants:
    python train.py --tfim-J 1.0 --tfim-h 1.0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── path setup (works from any working directory) ────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shadows.config import create_default_config
from shadows.collector import ShadowCollector
from shadows.processor import ShadowProcessor
from shadows.tokenization import create_default_tokenizer
from shadows.datasets import ShadowDataModule, DatasetConfig
from shadows.model import (
    create_model_from_tokenizer,
    ShadowTrainer,
    TargetScaler,
    TARGET_NAMES,   # ["magnetization", "correlations", "energy", "renyi_entropy"]
)

# Pauli matrices — used to build the TFIM Hamiltonian
_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1,  0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# TFIM Hamiltonian
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _kron_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Embed a single-qubit operator `op` acting on `qubit` into the full
    2^n x 2^n Hilbert space via tensor products with identity.

    Example: for n=3, qubit=1 returns I ⊗ op ⊗ I.
    """
    mats = [_I2] * n_qubits
    mats[qubit] = op
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _build_tfim_matrix(n_qubits: int, J: float, h: float) -> np.ndarray:
    """
    Build the full 2^n x 2^n matrix for the transverse-field Ising Hamiltonian:

        H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}
            -h * sum_{i=0}^{n-1} X_i

    with open boundary conditions.

    Parameters
    ----------
    n_qubits : int   Number of qubits (system size).
    J        : float ZZ coupling constant.
    h        : float Transverse-field strength.

    Returns
    -------
    H : np.ndarray, shape (2**n_qubits, 2**n_qubits), complex128
        Hermitian Hamiltonian matrix.
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # ZZ terms: -J * Z_i Z_{i+1} for i = 0 .. n-2
    for i in range(n_qubits - 1):
        # Z_i ⊗ Z_{i+1} embedded in full space
        zi   = _kron_op(_Z, i,     n_qubits)
        zi1  = _kron_op(_Z, i + 1, n_qubits)
        H   -= J * (zi @ zi1)

    # Transverse-field terms: -h * X_i for i = 0 .. n-1
    for i in range(n_qubits):
        H -= h * _kron_op(_X, i, n_qubits)

    return H


def _tfim_energy(state_vector: np.ndarray, H_matrix: np.ndarray) -> float:
    """
    Compute the exact TFIM energy expectation value <psi|H|psi>.

    Parameters
    ----------
    state_vector : np.ndarray, shape (2**n,), complex
    H_matrix     : np.ndarray, shape (2**n, 2**n), complex  (from _build_tfim_matrix)

    Returns
    -------
    float   Real part of <psi|H|psi>  (imaginary part is machine-precision zero
            for a Hermitian H and normalised psi).
    """
    return float(np.real(state_vector.conj() @ H_matrix @ state_vector))


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Other helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def set_seed(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state_vector(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a normalised Haar-random pure state.

    Real and imaginary parts are i.i.d. N(0,1), then normalised.
    Haar-random states span a wide range of all four physical quantities,
    giving the model diverse training signal.
    """
    dim = 2 ** n_qubits
    sv = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return sv / np.linalg.norm(sv)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Dataset generation
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def generate_dataset(
    n_states: int,
    n_qubits: int,
    n_shadows_per_state: int,
    seed: int,
    tfim_J: float,
    tfim_h: float,
) -> tuple:
    """
    Generate the training dataset.

    For each of the n_states random quantum states:
      1. Draw a Haar-random pure state.
      2. Build the TFIM Hamiltonian matrix (once, shared across all states).
      3. Collect n_shadows_per_state random-Pauli shadow measurements.
      4. Estimate magnetization, correlations, renyi_entropy via ShadowProcessor.
      5. Compute exact TFIM energy from the state vector.
      6. Broadcast the 4D target across all n_shadows measurements from this state.

    Broadcasting rationale
    ----------------------
    All measurements from the same state share the same target vector
    (the processor's per-state estimates).  The model learns to infer
    physical quantities from a single measurement's token pattern, with
    the MSE loss averaged across measurements per mini-batch.

    Returns
    -------
    merged_collector : ShadowCollector
        Holds all n_states * n_shadows_per_state measurements.
    targets : np.ndarray, shape (n_states * n_shadows_per_state, 4), float32
        Columns: [magnetization, correlations, energy, renyi_entropy]
    """
    rng = np.random.default_rng(seed)
    total = n_states * n_shadows_per_state

    # Build TFIM matrix once — shared for all states (same n_qubits, J, h).
    print(f"\nBuilding TFIM Hamiltonian  "
          f"(n={n_qubits}, J={tfim_J}, h={tfim_h}, OBC) ...")
    H_matrix = _build_tfim_matrix(n_qubits, J=tfim_J, h=tfim_h)
    print(f"  H shape: {H_matrix.shape}  "
          f"(Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)})")

    all_measurements = []
    all_targets = []

    print(f"\nGenerating {n_states} states x {n_shadows_per_state} shadows "
          f"= {total} measurements ...")
    t0 = time.time()

    for state_idx in range(n_states):
        # 1. Haar-random pure state
        sv = random_state_vector(n_qubits, rng)

        # 2. Collect random-Pauli shadows
        cfg = create_default_config(
            n_qubits=n_qubits,
            n_shadows=n_shadows_per_state,
            measurement_basis="random",
        )
        cfg.seed = int(rng.integers(0, 2**31))
        collector = ShadowCollector(cfg)
        collector.sample_dense(sv)

        # 3. Shadow-based processor estimates
        proc_cfg = create_default_config(n_qubits=n_qubits,
                                         n_shadows=n_shadows_per_state)
        proc_cfg.median_of_means = False   # plain mean: faster, fine for labels
        processor = ShadowProcessor(proc_cfg)
        estimates = processor.process_shadows(collector)

        # 4. 4D target vector — column order must match TARGET_NAMES exactly:
        #
        #   col 0: magnetization  — (1/n) sum_i <Z_i>,        shadow estimate
        #   col 1: correlations   — avg <Z_i Z_{i+1}>,        shadow estimate
        #   col 2: energy         — <H_TFIM>,                  exact from state
        #   col 3: renyi_entropy  — S_2 half-chain purity,     shadow estimate
        #
        # energy is NOT the same as magnetization:
        #   - magnetization = (1/n) sum <Z_i>      [single-qubit Z terms only]
        #   - energy = -J sum <Z_i Z_{i+1}> - h sum <X_i>  [ZZ + X terms]
        # The ZZ coupling makes energy sensitive to two-body correlations, and
        # the X term adds a contribution orthogonal to all Z-basis quantities.
        target_4d = np.array([
            estimates["magnetization"].estimate,   # col 0 — magnetization
            estimates["correlations"].estimate,    # col 1 — correlations
            _tfim_energy(sv, H_matrix),            # col 2 — energy (exact TFIM)
            estimates["renyi_entropy"].estimate,   # col 3 — renyi_entropy
        ], dtype=np.float32)

        # 5. Broadcast this state's target across all its measurements
        all_measurements.extend(collector.measurements)
        all_targets.append(np.tile(target_4d, (n_shadows_per_state, 1)))

        if (state_idx + 1) % max(1, n_states // 10) == 0:
            print(f"  {state_idx + 1}/{n_states} states  "
                  f"({time.time() - t0:.1f}s elapsed)")

    print(f"Dataset generation complete ({time.time() - t0:.1f}s).")

    # Merge all measurements into one collector
    merged_cfg = create_default_config(n_qubits=n_qubits, n_shadows=total)
    merged_collector = ShadowCollector(merged_cfg)
    merged_collector.measurements = all_measurements

    targets = np.vstack(all_targets).astype(np.float32)   # (total, 4)
    assert targets.shape == (total, 4), f"targets shape mismatch: {targets.shape}"

    return merged_collector, targets


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Training entry point
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Shadow Transformer -- 4-target regression baseline")
    print(f"{'='*60}")
    print(f"Device      : {device}")
    print(f"n_qubits    : {args.n_qubits}")
    print(f"n_states    : {args.n_states}")
    print(f"n_shadows   : {args.n_shadows_per_state}  per state")
    print(f"n_epochs    : {args.n_epochs}")
    print(f"batch_size  : {args.batch_size}")
    print(f"d_model     : {args.d_model}")
    print(f"TFIM        : J={args.tfim_J}  h={args.tfim_h}  OBC")
    print(f"Targets     : {TARGET_NAMES}")

    # ── 1. Generate dataset ───────────────────────────────────────────────────
    collector, targets = generate_dataset(
        n_states=args.n_states,
        n_qubits=args.n_qubits,
        n_shadows_per_state=args.n_shadows_per_state,
        seed=args.seed,
        tfim_J=args.tfim_J,
        tfim_h=args.tfim_h,
    )
    total = len(collector.measurements)

    print(f"\nTarget statistics (over {total} measurements):")
    for i, name in enumerate(TARGET_NAMES):
        col = targets[:, i]
        print(f"  {name:15s}  mean={col.mean():+.4f}  "
              f"std={col.std():.4f}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}")

    # Sanity check: energy should not be highly correlated with magnetization.
    r_em = float(np.corrcoef(targets[:, 0], targets[:, 2])[0, 1])
    r_ec = float(np.corrcoef(targets[:, 1], targets[:, 2])[0, 1])
    print(f"\n  Pearson r(magnetization, energy) = {r_em:+.3f}  (expect low)")
    print(f"  Pearson r(correlations,  energy) = {r_ec:+.3f}  "
          f"(moderate expected: ZZ terms shared)")

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = create_default_tokenizer(
        n_qubits=args.n_qubits,
        token_type=args.token_type,
    )
    print(f"\nTokenizer   : {tokenizer}")

    # ── 3. DataModule ─────────────────────────────────────────────────────────
    dm_cfg = DatasetConfig(
        batch_size=args.batch_size,
        shuffle=True,
        shuffle_seed=args.seed,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        pin_memory=(device == "cuda"),
        num_workers=0,
    )
    dm = ShadowDataModule(dm_cfg)
    dm.setup(collector, tokenizer, targets=targets)

    n_train = len(dm.train_dataset)
    n_val   = len(dm.val_dataset)
    n_test  = len(dm.test_dataset)
    print(f"\nDataset split : {n_train} train / {n_val} val / {n_test} test")
    print(f"Sequence length: {tokenizer.config.max_sequence_length} tokens")

    # ── 4. Target scaler (fit on train split only — no data leakage) ──────────
    train_targets = np.stack([
        dm.train_dataset[i]["target"].numpy()
        for i in range(n_train)
    ])   # (n_train, 4)
    scaler = TargetScaler(n_outputs=4)
    scaler.fit(train_targets)
    print(f"\nTarget scaler (fitted on {n_train} training samples):")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name:15s}  mean={scaler.mean_[i]:+.4f}  "
              f"std={scaler.std_[i]:.4f}")

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model = create_model_from_tokenizer(
        tokenizer,
        n_outputs=4,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pooling="mean",
        use_scheduler=True,
    )
    print(f"\nModel       : ShadowTransformer  ({model.count_parameters():,} params)")
    print(f"Architecture: d_model={args.d_model}  n_heads={args.n_heads}  "
          f"n_layers={args.n_layers}  d_ff={args.d_ff}")

    # ── 6. Trainer ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    trainer = ShadowTrainer(
        model,
        optimizer=optimizer,
        device=device,
        scaler=scaler,
    )

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n{'-'*60}")
    print(f"Training for {args.n_epochs} epochs  "
          f"(best checkpoint -> {checkpoint_path})")
    print(f"{'-'*60}")
    t_train = time.time()

    trainer.fit(
        dm.get_train_dataloader(),
        dm.get_val_dataloader(),
        n_epochs=args.n_epochs,
        print_every=max(1, args.n_epochs // 10),
        save_best_to=checkpoint_path,
    )

    print(f"\nTraining finished in {time.time() - t_train:.1f}s")
    print(f"Best val loss : {trainer._best_val_loss:.6f}  (scaled MSE)")

    # ── 8. Evaluate on test set with best weights ─────────────────────────────
    trainer.restore_best()
    print(f"\nRestored best checkpoint. Evaluating on test set ...")

    per_target = trainer.predict_per_target(dm.get_test_dataloader())

    test_targets = np.stack([
        dm.test_dataset[i]["target"].numpy()
        for i in range(n_test)
    ])   # (n_test, 4)

    print(f"\n{'-'*60}")
    print(f"Test set results  (n={n_test})")
    print(f"{'-'*60}")
    print(f"  {'Target':<16} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'-'*38}")
    for i, name in enumerate(TARGET_NAMES):
        preds = per_target[name]
        truth = test_targets[:, i]
        mae  = float(np.mean(np.abs(preds - truth)))
        rmse = float(np.sqrt(np.mean((preds - truth) ** 2)))
        print(f"  {name:<16} {mae:>10.5f} {rmse:>10.5f}")
    print(f"  {'-'*38}")

    # ── 9. Save artefacts ─────────────────────────────────────────────────────
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save_tokenizer(tokenizer_path)
    print(f"\nSaved tokenizer    -> {tokenizer_path}")
    print(f"Saved best model   -> {checkpoint_path}")
    print(f"\nDone.")


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# CLI
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a shadow transformer for 4-target physical quantity regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Dataset")
    g.add_argument("--n-qubits",  type=int, default=6,
                   help="Number of qubits per quantum state.")
    g.add_argument("--n-states",  type=int, default=100,
                   help="Number of random quantum states to generate.")
    g.add_argument("--n-shadows", type=int, default=100,
                   dest="n_shadows_per_state",
                   help="Shadow measurements per state.")
    g.add_argument("--token-type", type=str, default="basis_outcome",
                   choices=["basis_outcome", "pauli_string", "binary"],
                   help="Tokenization mode.")

    g = p.add_argument_group("Hamiltonian")
    g.add_argument("--tfim-J", type=float, default=1.0,
                   help="TFIM ZZ coupling constant J.")
    g.add_argument("--tfim-h", type=float, default=0.5,
                   help="TFIM transverse-field strength h.")

    g = p.add_argument_group("Model")
    g.add_argument("--d-model",  type=int,   default=128)
    g.add_argument("--n-heads",  type=int,   default=4)
    g.add_argument("--n-layers", type=int,   default=4)
    g.add_argument("--d-ff",     type=int,   default=512)
    g.add_argument("--dropout",  type=float, default=0.1)

    g = p.add_argument_group("Training")
    g.add_argument("--n-epochs",     type=int,   default=30)
    g.add_argument("--batch-size",   type=int,   default=32)
    g.add_argument("--lr",           type=float, default=3e-4)
    g.add_argument("--weight-decay", type=float, default=1e-2)

    g = p.add_argument_group("Misc")
    g.add_argument("--seed",       type=int, default=42)
    g.add_argument("--output-dir", type=str, default="./outputs",
                   help="Directory for checkpoints and tokenizer.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
