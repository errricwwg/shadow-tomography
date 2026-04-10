"""
train.py — End-to-end training script for classical-shadow physical quantity prediction.

Pipeline
--------
    random quantum states
        → ShadowCollector (random Pauli measurements)
        → ShadowProcessor (magnetization, correlations)
        → exact TFIM energy from state vector (column 2)
        → ShadowDataModule (tokenize, split, DataLoaders)
        → ShadowTransformer (3-target regression)
        → ShadowTrainer (fit, validate, checkpoint)

Target ordering (fixed throughout):
    column 0 — magnetization    (1/n) sum_i <Z_i>       shadow estimate
    column 1 — correlations     avg <Z_i Z_{i+1}>        shadow estimate
    column 2 — energy           <H_TFIM>                 exact from state vector

Energy target: transverse-field Ising model (TFIM)
---------------------------------------------------
The Hamiltonian is:

    H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}   (nearest-neighbour ZZ coupling)
        -h * sum_{i=0}^{n-1} X_i            (transverse field)

with J = 1.0 (coupling), h = 0.5 (transverse field), open boundary conditions.

Why TFIM?
  - Standard benchmark for quantum many-body physics.
  - Contains both ZZ (two-site) and X (single-site) terms, so <H> is genuinely
    different from magnetization (1/n sum <Z_i>) or correlations (avg <Z_i Z_{i+1}>).
  - Non-redundant: Pearson r(magnetization, energy) ~ 0 for Haar-random states.
  - Exactly computable via sv.conj() @ H_matrix @ sv; no extra dependencies.

Usage
-----
    cd "shadow GPT/code/shadows"
    python train.py

    # Quick smoke test:
    python train.py --n-states 20 --n-shadows 50 --n-epochs 5

    # Larger run:
    python train.py --n-states 500 --n-shadows 200 --n-qubits 8 --d-model 256

    # Change coupling constants:
    python train.py --tfim-J 1.0 --tfim-h 1.0

    # Multi-seed evaluation (runs seeds 0..4, prints mean ± std):
    python train.py --n-states 100 --n-shadows 100 --n-epochs 20 --multi-seed 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Pauli matrices — used to build the TFIM Hamiltonian
_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1,  0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)

# ── path setup ────────────────────────────────────────────────────────────────
# This file lives inside shadows/, so we go up one level to reach code/ where
# the shadows package itself lives.
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from shadows.config import create_default_config
from shadows.collector import ShadowCollector
from shadows.processor import ShadowProcessor
from shadows.tokenization import create_default_tokenizer
from shadows.datasets import ShadowDataModule, ShadowDataset, DatasetConfig
from shadows.model import (
    create_model_from_tokenizer,
    ShadowTrainer,
    TargetScaler,
    TARGET_NAMES,
)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def set_seed(seed: int) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_state_vector(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """
    Return a normalised random pure state in the 2^n-dimensional Hilbert space.

    Both real and imaginary parts are drawn i.i.d. from N(0,1), then
    normalised.  The resulting distribution is Haar-uniform over pure states.
    This produces a wide range of physical quantity values, which is exactly
    what supervised regression needs.
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


def _build_tfim_matrix(n_qubits: int, J: float, h: float) -> np.ndarray:
    """
    Build the 2^n x 2^n matrix for the transverse-field Ising Hamiltonian:

        H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1}   (ZZ coupling, OBC)
            -h * sum_{i=0}^{n-1} X_i            (transverse field)

    Parameters
    ----------
    n_qubits : int    System size.
    J        : float  ZZ coupling (default 1.0).
    h        : float  Transverse-field strength (default 0.5).

    Returns
    -------
    H : np.ndarray, shape (2**n_qubits, 2**n_qubits), complex128.
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):          # ZZ terms
        H -= J * (_kron_op(_Z, i, n_qubits) @ _kron_op(_Z, i + 1, n_qubits))
    for i in range(n_qubits):              # transverse-field terms
        H -= h * _kron_op(_X, i, n_qubits)
    return H


def _tfim_energy(state_vector: np.ndarray, H_matrix: np.ndarray) -> float:
    """
    Return the exact TFIM energy <psi|H|psi> for a normalised state vector.

    Parameters
    ----------
    state_vector : np.ndarray, shape (2**n,), complex.
    H_matrix     : np.ndarray, shape (2**n, 2**n), complex  — from _build_tfim_matrix.
    """
    return float(np.real(state_vector.conj() @ H_matrix @ state_vector))


def generate_dataset(
    n_states: int,
    n_qubits: int,
    n_shadows_per_state: int,
    seed: int,
    tfim_J: float,
    tfim_h: float,
) -> tuple:
    """
    Generate the full training dataset by:
        1. Creating n_states random quantum states.
        2. Building the TFIM Hamiltonian matrix once (shared across all states).
        3. Collecting n_shadows_per_state measurements per state.
        4. Computing 3D physical quantity targets per state via ShadowProcessor
           plus exact TFIM energy from the state vector.
        5. Broadcasting each state's target across its n_shadows measurements.

    Broadcasting design note
    ------------------------
    ShadowDataModule maps one token sequence → one target.  Measurements from
    the same state all carry the same (processor-estimated) target.  This
    teaches the model to infer the physical quantity from the statistical
    pattern of a single measurement, averaged implicitly via the loss over
    all measurements from the state.

    Returns
    -------
    merged_collector : ShadowCollector
        Single collector holding all n_states × n_shadows_per_state measurements.
    targets : np.ndarray, shape (n_states * n_shadows_per_state, 3), float32
        Target matrix; rows align with measurements in merged_collector.
        Columns: [magnetization, correlations, energy]
    """
    rng = np.random.default_rng(seed)
    total = n_states * n_shadows_per_state

    # Build TFIM matrix once — reused for every state (same n_qubits, J, h).
    print(f"\nBuilding TFIM Hamiltonian  "
          f"(n={n_qubits}, J={tfim_J}, h={tfim_h}, OBC) ...")
    H_matrix = _build_tfim_matrix(n_qubits, J=tfim_J, h=tfim_h)
    print(f"  H shape: {H_matrix.shape}  "
          f"(Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)})")

    all_measurements = []
    all_targets      = []

    print(f"\nGenerating {n_states} states x {n_shadows_per_state} shadows "
          f"= {total} measurements ...")
    t0 = time.time()

    for state_idx in range(n_states):
        # ── 1. Random pure state ──────────────────────────────────────────────
        sv = random_state_vector(n_qubits, rng)

        # ── 2. Collect shadows ────────────────────────────────────────────────
        cfg = create_default_config(
            n_qubits=n_qubits,
            n_shadows=n_shadows_per_state,
            measurement_basis="random",
        )
        cfg.seed = int(rng.integers(0, 2**31))   # per-state reproducible seed
        collector = ShadowCollector(cfg)
        collector.sample_dense(sv)

        # ── 3. Processor estimates (magnetization, correlations) ─────────────
        proc_cfg = create_default_config(
            n_qubits=n_qubits,
            n_shadows=n_shadows_per_state,
        )
        proc_cfg.median_of_means = False   # plain mean: faster, fine for labels
        processor = ShadowProcessor(proc_cfg)
        estimates = processor.process_shadows(collector)

        # ── 4. Build 3D target vector for this state ──────────────────────────
        #   col 0: magnetization  — (1/n) sum <Z_i>,            shadow estimate
        #   col 1: correlations   — avg <Z_i Z_{i+1}>,          shadow estimate
        #   col 2: energy         — <H_TFIM> exact from state   (NOT redundant:
        #                           contains ZZ two-body + X transverse terms)
        target_3d = np.array([
            estimates["magnetization"].estimate,   # col 0
            estimates["correlations"].estimate,    # col 1
            _tfim_energy(sv, H_matrix),            # col 2 — exact TFIM energy
        ], dtype=np.float32)

        # ── 5. Broadcast target across all measurements from this state ───────
        all_measurements.extend(collector.measurements)
        all_targets.append(
            np.tile(target_3d, (n_shadows_per_state, 1))
        )   # shape (n_shadows_per_state, 3)

        if (state_idx + 1) % max(1, n_states // 10) == 0:
            elapsed = time.time() - t0
            print(f"  {state_idx + 1}/{n_states} states  "
                  f"({elapsed:.1f}s elapsed)")

    print(f"Dataset generation complete ({time.time() - t0:.1f}s total).")

    # ── Merge into one collector ──────────────────────────────────────────────
    merged_cfg = create_default_config(n_qubits=n_qubits, n_shadows=total)
    merged_collector = ShadowCollector(merged_cfg)
    merged_collector.measurements = all_measurements

    targets = np.vstack(all_targets).astype(np.float32)   # (total, 3)

    assert len(merged_collector.measurements) == total
    assert targets.shape == (total, 3)

    return merged_collector, targets


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Training entry point
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Shadow Transformer — 3-target regression baseline")
    print(f"{'='*60}")
    print(f"Device      : {device}")
    print(f"n_qubits    : {args.n_qubits}")
    print(f"n_states    : {args.n_states}")
    print(f"n_shadows   : {args.n_shadows_per_state}  per state")
    print(f"n_epochs    : {args.n_epochs}")
    print(f"batch_size  : {args.batch_size}")
    print(f"d_model     : {args.d_model}")
    print(f"Targets     : {TARGET_NAMES}")
    print(f"TFIM J      : {args.tfim_J}  (ZZ coupling)")
    print(f"TFIM h      : {args.tfim_h}  (transverse field)")

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

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = create_default_tokenizer(
        n_qubits=args.n_qubits,
        token_type=args.token_type,
    )
    print(f"\nTokenizer   : {tokenizer}")

    # ── 3. State-level split → DataModule ─────────────────────────────────────
    # generate_dataset() appends measurements in state order:
    # measurements 0..S-1 = state 0, S..2S-1 = state 1, etc.
    # Splitting at the measurement level (the old approach) would allow
    # measurements from the same state to appear in multiple splits, leaking
    # label information from train into val/test.
    # Fix: assign each state to exactly one split first, then flatten.
    S = args.n_shadows_per_state

    split_rng = np.random.default_rng(args.seed)
    state_order = np.arange(args.n_states)
    split_rng.shuffle(state_order)
    n_train_s = int(args.n_states * 0.8)
    n_val_s   = int(args.n_states * 0.1)
    train_states = set(state_order[:n_train_s])
    val_states   = set(state_order[n_train_s:n_train_s + n_val_s])

    # measurement k was produced by state k // S
    state_of_meas = np.repeat(np.arange(args.n_states), S)
    train_mask = np.array([s in train_states for s in state_of_meas])
    val_mask   = np.array([s in val_states   for s in state_of_meas])
    test_mask  = ~(train_mask | val_mask)

    # Tokenize the full collector once, then index by split mask
    token_seqs = tokenizer.tokenize_collector(collector)
    all_seqs   = tokenizer.create_sequences(token_seqs, add_special_tokens=True)

    def _select_seqs(mask):
        return [all_seqs[i] for i, m in enumerate(mask) if m]

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
    dm.tokenizer = tokenizer
    dm.train_dataset = ShadowDataset(_select_seqs(train_mask), tokenizer, targets[train_mask], dm_cfg)
    dm.val_dataset   = ShadowDataset(_select_seqs(val_mask),   tokenizer, targets[val_mask],   dm_cfg)
    dm.test_dataset  = ShadowDataset(_select_seqs(test_mask),  tokenizer, targets[test_mask],  dm_cfg)

    n_train_states_test = args.n_states - n_train_s - n_val_s
    n_train = len(dm.train_dataset)
    n_val   = len(dm.val_dataset)
    n_test  = len(dm.test_dataset)
    print(f"\nDataset split : {n_train} train / {n_val} val / {n_test} test"
          f"  (state-level: {n_train_s} / {n_val_s} / {n_train_states_test} states)")
    print(f"Sequence length: {tokenizer.config.max_sequence_length} tokens")

    # ── 4. Target scaler (fit on train split only) ────────────────────────────
    # We fit the scaler on training targets only to avoid data leakage.
    # Scaler is stored in the trainer so checkpoints are self-contained and
    # predictions are always returned in the original (unscaled) target space.
    train_targets = np.stack([
        dm.train_dataset[i]["target"].numpy()
        for i in range(n_train)
    ])   # (n_train, 4)
    scaler = TargetScaler(n_outputs=3)
    scaler.fit(train_targets)
    print(f"\nTarget scaler fitted on {n_train} training samples.")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name:15s}  mean={scaler.mean_[i]:+.4f}  std={scaler.std_[i]:.4f}")

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model = create_model_from_tokenizer(
        tokenizer,
        n_outputs=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pooling="mean",
        use_scheduler=True,
    )
    n_params = model.count_parameters()
    print(f"\nModel       : ShadowTransformer")
    print(f"Parameters  : {n_params:,}")
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
    print(f"\n{'─'*60}")
    print(f"Training for {args.n_epochs} epochs  "
          f"(best checkpoint → {checkpoint_path})")
    print(f"{'─'*60}")
    t_train = time.time()

    history = trainer.fit(
        dm.get_train_dataloader(),
        dm.get_val_dataloader(),
        n_epochs=args.n_epochs,
        print_every=max(1, args.n_epochs // 10),
        save_best_to=checkpoint_path,
    )

    print(f"\nTraining finished in {time.time() - t_train:.1f}s")
    print(f"Best val loss : {trainer._best_val_loss:.6f}  (scaled MSE)")

    # ── 8. Restore best weights and evaluate on test set ─────────────────────
    trainer.restore_best()
    print(f"\nRestored best checkpoint. Evaluating on test set …")

    per_target = trainer.predict_per_target(dm.get_test_dataloader())

    # Load test targets (unscaled) for MAE calculation
    test_targets = np.stack([
        dm.test_dataset[i]["target"].numpy()
        for i in range(n_test)
    ])   # (n_test, 4)

    print(f"\n{'─'*60}")
    print(f"Test set results  (n={n_test})")
    print(f"{'─'*60}")
    print(f"{'Target':<18} {'MAE':>10} {'RMSE':>10}")
    print(f"{'─'*40}")
    for i, name in enumerate(TARGET_NAMES):
        preds = per_target[name]
        truth = test_targets[:, i]
        mae  = float(np.mean(np.abs(preds - truth)))
        rmse = float(np.sqrt(np.mean((preds - truth) ** 2)))
        print(f"  {name:<16} {mae:>10.5f} {rmse:>10.5f}")
    print(f"{'─'*40}")

    # ── 9. Save final artefacts ───────────────────────────────────────────────
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save_tokenizer(tokenizer_path)
    print(f"\nSaved tokenizer → {tokenizer_path}")
    print(f"Saved best checkpoint → {checkpoint_path}")
    print(f"\nDone.")

    # Return per-target metrics for multi-seed aggregation.
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        preds = per_target[name]
        truth = test_targets[:, i]
        metrics[name] = {
            "mae":  float(np.mean(np.abs(preds - truth))),
            "rmse": float(np.sqrt(np.mean((preds - truth) ** 2))),
        }
    return metrics


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Multi-seed evaluation
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def run_multi_seed(args: argparse.Namespace) -> None:
    """
    Run the full training pipeline for seeds 0 … args.multi_seed-1.
    Each seed controls: dataset generation, state-level split, model init,
    and all training randomness.  Results are collected and summarised.
    """
    import copy
    n_seeds = args.multi_seed
    all_metrics = []   # list of dicts, one per seed

    base_output_dir = args.output_dir

    for seed in range(n_seeds):
        print(f"\n{'#'*60}")
        print(f"# Multi-seed run  seed={seed}  ({seed+1}/{n_seeds})")
        print(f"{'#'*60}")

        # Give each seed its own checkpoint directory so runs don't overwrite
        # each other.
        seed_args = copy.copy(args)
        seed_args.seed = seed
        seed_args.output_dir = os.path.join(base_output_dir, f"seed_{seed}")

        metrics = train(seed_args)
        all_metrics.append(metrics)

    # ── Per-seed table ────────────────────────────────────────────────────────
    col_w = 16
    header = f"{'Seed':>4}  " + "  ".join(
        f"{name+'_MAE':>{col_w}}  {name+'_RMSE':>{col_w}}"
        for name in TARGET_NAMES
    )
    print(f"\n{'='*60}")
    print(f"Multi-seed results  ({n_seeds} seeds)")
    print(f"{'='*60}")
    print(header)
    print("─" * len(header))
    for seed, m in enumerate(all_metrics):
        row = f"{seed:>4}  " + "  ".join(
            f"{m[name]['mae']:>{col_w}.5f}  {m[name]['rmse']:>{col_w}.5f}"
            for name in TARGET_NAMES
        )
        print(row)

    # ── Aggregated mean ± std ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Target':<18} {'Mean MAE':>12} {'Std MAE':>12} {'Mean RMSE':>12} {'Std RMSE':>12}")
    print(f"{'─'*60}")
    for name in TARGET_NAMES:
        maes  = [m[name]["mae"]  for m in all_metrics]
        rmses = [m[name]["rmse"] for m in all_metrics]
        print(
            f"  {name:<16} "
            f"{np.mean(maes):>12.5f} {np.std(maes):>12.5f} "
            f"{np.mean(rmses):>12.5f} {np.std(rmses):>12.5f}"
        )
    print(f"{'─'*60}")


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# CLI
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a shadow transformer for 3-target physical quantity regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    g = p.add_argument_group("Dataset")
    g.add_argument("--n-qubits", type=int, default=6,
                   help="Number of qubits per quantum state.")
    g.add_argument("--n-states", type=int, default=100,
                   help="Number of random quantum states to generate.")
    g.add_argument("--n-shadows", type=int, default=100,
                   dest="n_shadows_per_state",
                   help="Shadow measurements per state.")
    g.add_argument("--token-type", type=str, default="basis_outcome",
                   choices=["basis_outcome", "pauli_string", "binary"],
                   help="Tokenization mode.")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--d-model", type=int, default=128,
                   help="Embedding / hidden dimension.")
    g.add_argument("--n-heads", type=int, default=4,
                   help="Number of attention heads (must divide d_model).")
    g.add_argument("--n-layers", type=int, default=4,
                   help="Number of transformer encoder layers.")
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
                   help="If > 0, run this many seeds (0..N-1) and report mean±std. "
                        "--seed is ignored in multi-seed mode.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed > 0:
        run_multi_seed(args)
    else:
        train(args)
