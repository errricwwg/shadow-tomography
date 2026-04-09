"""
train.py — End-to-end training script for classical-shadow physical quantity prediction.

Pipeline
--------
    random quantum states
        → ShadowCollector (random Pauli measurements)
        → ShadowProcessor (magnetization, correlations, renyi_entropy)
        → exact X-magnetization from state vector (column 2)
        → ShadowDataModule (tokenize, split, DataLoaders)
        → ShadowTransformer (4-target regression)
        → ShadowTrainer (fit, validate, checkpoint)

Target ordering (fixed throughout):
    column 0 — magnetization      (1/n) sum_i <Z_i>   — shadow estimate
    column 1 — correlations       avg <Z_i Z_j>        — shadow estimate
    column 2 — x_magnetization    (1/n) sum_i <X_i>   — exact from state vector
    column 3 — renyi_entropy      S_2 of half-chain    — shadow estimate

Why these four targets?
-----------------------
The four targets measure genuinely distinct properties of the quantum state:

  - magnetization    : Z-basis order / spin polarization
  - correlations     : Z-Z two-point correlations (pair structure)
  - x_magnetization  : transverse (X-basis) order / quantum coherence
  - renyi_entropy    : bipartite entanglement / entanglement entropy

magnetization and x_magnetization come from non-commuting observables (Z vs X),
so they are statistically independent for generic states: knowing <Z> tells you
nothing about <X>.  This makes the target set non-redundant.

Note on x_magnetization (column 2)
-----------------------------------
processor.estimate_energy() requires a pyclifford Hamiltonian, which is an
optional dependency.  Rather than use an exact Z-energy (which would be
essentially identical to the shadow-estimated magnetization in column 0 --
same observable, just zero variance), we compute the exact X-magnetization:

    x_mag = (1/n) sum_i <X_i>
           = (1/n) sum_i [p(qubit i in |+>) - p(qubit i in |->)]

where |+> / |-> are the X eigenstates.  In code: apply a Hadamard to qubit i,
then read its marginal probability.  This is always available (no extra
dependencies), genuinely different from all other targets, and physically
meaningful as a measure of transverse quantum coherence.

If you have pyclifford and a real Hamiltonian, replace _exact_x_magnetization()
with processor.estimate_energy(collector, hamiltonian).estimate.

Broadcasting design
-------------------
ShadowDataModule maps one token sequence (one shadow measurement) to one target
vector.  All n_shadows_per_state measurements from the same quantum state share
the same 4D target (the processor's per-state estimates).  This teaches the
model to infer physical quantities from a single measurement's token pattern,
with the MSE loss averaged across measurements in each mini-batch.

Usage
-----
    cd "shadow GPT/code"
    python train.py

    # Quick smoke test (~30s on CPU):
    python train.py --n-states 20 --n-shadows 50 --n-epochs 5

    # Larger run:
    python train.py --n-states 500 --n-shadows 200 --n-qubits 8 --d-model 256
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── path setup (so the script works from any working directory) ───────────────
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
    TARGET_NAMES,   # ["magnetization", "correlations", "x_magnetization", "renyi_entropy"]
)

# Hadamard gate (reused in _exact_x_magnetization)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


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
    Return a normalised Haar-random pure state in the 2^n-dimensional Hilbert space.

    Both real and imaginary parts are drawn i.i.d. from N(0,1) then normalised.
    Haar-random states naturally span a wide range of Z-magnetization, ZZ-correlations,
    X-magnetization, and entanglement values, giving the model diverse training signal.
    """
    dim = 2 ** n_qubits
    sv = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return sv / np.linalg.norm(sv)


def _exact_x_magnetization(state_vector: np.ndarray, n_qubits: int) -> float:
    """
    Compute (1/n) sum_i <X_i> exactly from the state vector.

    Algorithm: for each qubit i, apply the Hadamard gate H to rotate the state
    into the X eigenbasis, then read off the marginal probability of qubit i
    being 0 vs 1.  In the X eigenbasis, outcome 0 corresponds to eigenvalue +1
    and outcome 1 to eigenvalue -1, so:

        <X_i> = p(qubit i = 0 in X-basis) - p(qubit i = 1 in X-basis)

    This is the exact (zero-variance) X-magnetization.  It uses the same
    state-vector access pattern as computing Z-marginals, but with a basis
    rotation first.

    Why not just use the Z-magnetization exactly?
    That would duplicate column 0 (magnetization), which is the same observable
    estimated from shadows.  X and Z are non-commuting, so <X_i> and <Z_i> are
    statistically independent for generic states.
    """
    state = state_vector.reshape([2] * n_qubits)
    total_x = 0.0
    for i in range(n_qubits):
        # Rotate qubit i to the X eigenbasis via Hadamard
        rotated = np.tensordot(_H, state, axes=([1], [i]))   # contracts on phys index
        rotated = np.moveaxis(rotated, 0, i)                 # put qubit i axis back
        probs = np.abs(rotated.reshape(-1)) ** 2
        probs_nd = probs.reshape([2] * n_qubits)
        # Marginalise over all qubits except i
        axes = tuple(j for j in range(n_qubits) if j != i)
        marginal = probs_nd.sum(axis=axes)   # [p(|+>), p(|->)]
        total_x += float(marginal[0] - marginal[1])
    return total_x / n_qubits


def generate_dataset(
    n_states: int,
    n_qubits: int,
    n_shadows_per_state: int,
    seed: int,
) -> tuple:
    """
    Generate the training dataset.

    For each of the n_states random quantum states:
      1. Draw a Haar-random pure state.
      2. Collect n_shadows_per_state random-Pauli shadow measurements.
      3. Estimate magnetization, correlations, renyi_entropy via ShadowProcessor.
      4. Compute exact X-magnetization from the state vector (column 2).
      5. Broadcast the 4D target across all n_shadows measurements from this state.

    Broadcasting rationale
    ----------------------
    The dataset has one row per measurement.  All measurements from the same
    state share the same target vector.  The loss is averaged over measurements
    in each mini-batch, so the effective per-state weight is proportional to
    n_shadows_per_state (uniform across states since all states contribute equally).

    Returns
    -------
    merged_collector : ShadowCollector
        Holds all n_states * n_shadows_per_state measurements.
    targets : np.ndarray, shape (n_states * n_shadows_per_state, 4), float32
        Columns: [magnetization, correlations, x_magnetization, renyi_entropy]
    """
    rng = np.random.default_rng(seed)
    total = n_states * n_shadows_per_state

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
        cfg.seed = int(rng.integers(0, 2**31))   # per-state reproducible seed
        collector = ShadowCollector(cfg)
        collector.sample_dense(sv)

        # 3. Shadow-based processor estimates
        proc_cfg = create_default_config(n_qubits=n_qubits,
                                         n_shadows=n_shadows_per_state)
        proc_cfg.median_of_means = False   # plain mean: faster, fine for labels
        processor = ShadowProcessor(proc_cfg)
        estimates = processor.process_shadows(collector)

        # 4. 4D target vector — column order must match TARGET_NAMES
        #
        #   col 0: magnetization   — (1/n) sum_i <Z_i>, shadow estimate
        #   col 1: correlations    — avg <Z_i Z_j> for |i-j|<=2, shadow estimate
        #   col 2: x_magnetization — (1/n) sum_i <X_i>, exact from state vector
        #                            (non-commuting with Z -> independent of col 0)
        #   col 3: renyi_entropy   — S_2 half-chain, shadow estimate
        target_4d = np.array([
            estimates["magnetization"].estimate,    # col 0 — magnetization
            estimates["correlations"].estimate,     # col 1 — correlations
            _exact_x_magnetization(sv, n_qubits),  # col 2 — x_magnetization (exact)
            estimates["renyi_entropy"].estimate,    # col 3 — renyi_entropy
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
    print(f"Targets     : {TARGET_NAMES}")

    # ── 1. Generate dataset ───────────────────────────────────────────────────
    collector, targets = generate_dataset(
        n_states=args.n_states,
        n_qubits=args.n_qubits,
        n_shadows_per_state=args.n_shadows_per_state,
        seed=args.seed,
    )
    total = len(collector.measurements)

    print(f"\nTarget statistics (over {total} measurements):")
    for i, name in enumerate(TARGET_NAMES):
        col = targets[:, i]
        print(f"  {name:17s}  mean={col.mean():+.4f}  "
              f"std={col.std():.4f}  "
              f"min={col.min():+.4f}  max={col.max():+.4f}")

    # Sanity check: x_magnetization (col 2) vs magnetization (col 0)
    # Their Pearson correlation should be near zero for Haar-random states.
    r = float(np.corrcoef(targets[:, 0], targets[:, 2])[0, 1])
    print(f"\n  Pearson r(magnetization, x_magnetization) = {r:+.3f}  "
          f"(expect ~0 for independent observables)")

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

    # ── 4. Target scaler (fit on train split only -- no data leakage) ─────────
    train_targets = np.stack([
        dm.train_dataset[i]["target"].numpy()
        for i in range(n_train)
    ])   # (n_train, 4)
    scaler = TargetScaler(n_outputs=4)
    scaler.fit(train_targets)
    print(f"\nTarget scaler (fitted on {n_train} training samples):")
    for i, name in enumerate(TARGET_NAMES):
        print(f"  {name:17s}  mean={scaler.mean_[i]:+.4f}  "
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
    print(f"  {'Target':<18} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'-'*40}")
    for i, name in enumerate(TARGET_NAMES):
        preds = per_target[name]
        truth = test_targets[:, i]
        mae  = float(np.mean(np.abs(preds - truth)))
        rmse = float(np.sqrt(np.mean((preds - truth) ** 2)))
        print(f"  {name:<18} {mae:>10.5f} {rmse:>10.5f}")
    print(f"  {'-'*40}")

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

    g = p.add_argument_group("Model")
    g.add_argument("--d-model",   type=int,   default=128)
    g.add_argument("--n-heads",   type=int,   default=4)
    g.add_argument("--n-layers",  type=int,   default=4)
    g.add_argument("--d-ff",      type=int,   default=512)
    g.add_argument("--dropout",   type=float, default=0.1)

    g = p.add_argument_group("Training")
    g.add_argument("--n-epochs",      type=int,   default=30)
    g.add_argument("--batch-size",    type=int,   default=32)
    g.add_argument("--lr",            type=float, default=3e-4)
    g.add_argument("--weight-decay",  type=float, default=1e-2)

    g = p.add_argument_group("Misc")
    g.add_argument("--seed",       type=int, default=42)
    g.add_argument("--output-dir", type=str, default="./outputs",
                   help="Directory for checkpoints and tokenizer.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
