"""
inference_engine.py — Exact diagonalization backend for the NL Hamiltonian interface.

Takes a Step-2 ``ParsedHamiltonian`` and computes ground-state physical properties
using the existing ``hamiltonians.build_hamiltonian_spec()`` factory and dense exact
diagonalization.  No learned model is involved.

Observable definitions (consistent with shadow estimator conventions in processor.py)
--------------------------------------------------------------------------------------
energy         : total ground-state energy  E_0 = ⟨ψ_0|H|ψ_0⟩  (not per site)
magnetization  : average Z magnetization  (1/n) Σ_i ⟨Z_i⟩
correlations   : average nearest-neighbor ZZ  (1/(n−1)) Σ_{i=0}^{n−2} ⟨Z_i Z_{i+1}⟩
                 (matches ShadowProcessor.estimate_correlations with correlation_length=1)
renyi2_entropy : S_2(ρ_A) = −log Tr(ρ_A²), subsystem A = first ⌊n/2⌋ qubits (≥1)

Qubit ordering convention
--------------------------
Follows ``hamiltonians._kron_op``: qubit 0 is the MSB of the computational-basis
index.  For basis index k and qubit i: bit_i(k) = (k >> (n−1−i)) & 1.
"""

from __future__ import annotations

import warnings as _warnings
from typing import List, Tuple

import numpy as np

from .hamiltonians import build_hamiltonian_spec, HamiltonianSpec
from .nl_schema import ParsedHamiltonian, PropertyResult
from .family_registry import get_family_spec
from .nl_parser import parse_hamiltonian_text


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Safety threshold
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

# Dense matrices of size 2^n × 2^n become impractical above n ≈ 20.
# A UserWarning is emitted (not an error) to allow intentional large-n use.
_EXACT_WARN_THRESHOLD: int = 16


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Input validation
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _validate_parsed_for_exact(parsed: ParsedHamiltonian) -> None:
    """
    Raise ``ValueError`` if *parsed* is not ready for exact evaluation.

    Validation checks (in order):
        1. ``family`` is not None.
        2. ``n_qubits`` is not None.
        3. ``boundary == "obc"`` (only OBC supported by the dense builders).
        4. Family exists in the registry.
        5. All required parameters for the family are present in ``params``.
        6. ``supported`` is True (belt-and-suspenders gate).

    Collects all problems before raising so the error message is complete.
    """
    errors: List[str] = []

    if parsed.family is None:
        errors.append("family is not identified")

    if parsed.n_qubits is None:
        errors.append("n_qubits is not specified")

    if parsed.boundary != "obc":
        errors.append(
            f"boundary='{parsed.boundary}' is unsupported; "
            "only 'obc' is currently implemented"
        )

    if parsed.family is not None:
        spec = get_family_spec(parsed.family)
        if spec is None:
            errors.append(f"family '{parsed.family}' is not in the registry")
        else:
            missing = [p for p in spec.required_params if p not in parsed.params]
            if missing:
                errors.append(
                    f"required parameters missing for '{parsed.family}': "
                    + ", ".join(f"'{p}'" for p in missing)
                )

    if errors:
        detail = "; ".join(errors)
        raise ValueError(
            f"Cannot run exact inference — {detail}. "
            "Check parser warnings for more detail."
        )

    # Redundant once the above pass, but explicit for clarity.
    if not parsed.supported:
        raise ValueError(
            "ParsedHamiltonian.supported is False. "
            "Resolve all parser warnings before calling evaluate_exact()."
        )


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Ground-state solver
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _ground_state_from_spec(spec: HamiltonianSpec) -> Tuple[np.ndarray, float]:
    """
    Compute the ground state by exact diagonalization of ``spec.dense_matrix``.

    Uses ``numpy.linalg.eigh`` (Hermitian eigensolver), which returns eigenvalues
    in ascending order, guaranteeing that index 0 is the ground state.

    Returns
    -------
    psi : np.ndarray, shape (2^n,), dtype complex128
        Normalised ground-state vector.
    E0  : float
        Ground-state energy (smallest eigenvalue).

    Note: if the ground state is degenerate, the first eigenvector from eigh is
    returned (an arbitrary element of the ground-state subspace).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(spec.dense_matrix)
    psi: np.ndarray = eigenvectors[:, 0]
    E0: float = float(eigenvalues[0])
    return psi, E0


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Observable helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _exact_energy(E0: float) -> float:
    """Return the total ground-state energy (direct pass-through from eigh)."""
    return E0


def _exact_magnetization(psi: np.ndarray, n_qubits: int) -> float:
    """
    Compute the average Z magnetization  m_z = (1/n) Σ_{i=0}^{n−1} ⟨Z_i⟩.

    Qubit ordering: qubit 0 = MSB of basis index k.
        bit_i(k) = (k >> (n−1−i)) & 1
        Z_i eigenvalue: +1 for bit_i=0 (|0⟩), −1 for bit_i=1 (|1⟩).

    Complexity: O(n · 2^n) — practical for n ≤ 20.
    """
    dim = 1 << n_qubits
    probs = np.abs(psi) ** 2          # shape (2^n,)
    indices = np.arange(dim, dtype=np.int64)
    total = 0.0
    for i in range(n_qubits):
        bit_i = (indices >> (n_qubits - 1 - i)) & 1
        z_vals = 1.0 - 2.0 * bit_i.astype(np.float64)
        total += float(np.dot(z_vals, probs))
    return total / n_qubits


def _exact_correlations(psi: np.ndarray, n_qubits: int) -> float:
    """
    Compute the average nearest-neighbor ZZ correlation:

        C_ZZ = (1/(n−1)) Σ_{i=0}^{n−2} ⟨Z_i Z_{i+1}⟩

    Matches ``ShadowProcessor.estimate_correlations()`` with
    ``correlation_length=1`` under open boundary conditions.

    Returns 0.0 for n_qubits < 2 (no nearest-neighbor pairs exist).
    """
    if n_qubits < 2:
        return 0.0
    dim = 1 << n_qubits
    probs = np.abs(psi) ** 2
    indices = np.arange(dim, dtype=np.int64)
    total = 0.0
    for i in range(n_qubits - 1):
        bit_i = (indices >> (n_qubits - 1 - i)) & 1
        bit_j = (indices >> (n_qubits - 2 - i)) & 1
        z_i = 1.0 - 2.0 * bit_i.astype(np.float64)
        z_j = 1.0 - 2.0 * bit_j.astype(np.float64)
        total += float(np.dot(z_i * z_j, probs))
    return total / (n_qubits - 1)


def _exact_renyi2(psi: np.ndarray, n_qubits: int) -> Tuple[float, int]:
    """
    Compute the Rényi-2 entropy  S_2(ρ_A) = −log Tr(ρ_A²).

    Subsystem choice
    ----------------
    A = first k qubits, where k = max(1, n_qubits // 2).
    This is the standard half-chain bipartition (or nearest half for odd n).
    For n=1 the result is trivially 0 (pure state, no bipartition possible).

    Algorithm
    ---------
    Reshape |ψ⟩ ∈ ℂ^{2^n} as matrix M of shape (2^k, 2^{n−k}), where
    M_{a,b} = ψ_{a · 2^{n−k} + b}.  The reduced density matrix is:
        ρ_A = M M†   (shape: 2^k × 2^k)
    Purity:  Tr(ρ_A²) = Tr((MM†)²).
    S_2 = −log(max(purity, ε))  with ε = 10⁻¹² to avoid log(0).

    Returns
    -------
    S2 : float — Rényi-2 entropy (≥ 0).
    k  : int   — subsystem size used (recorded in PropertyResult.notes).
    """
    k = max(1, n_qubits // 2)
    dim_A = 1 << k
    dim_B = 1 << (n_qubits - k)
    M = psi.reshape(dim_A, dim_B)
    rho_A = M @ M.conj().T                       # (dim_A, dim_A)
    purity = float(np.real(np.trace(rho_A @ rho_A)))
    purity_clipped = float(np.clip(purity, 1e-12, 1.0))
    return -np.log(purity_clipped), k


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Main entry point
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def evaluate_exact(parsed: ParsedHamiltonian) -> PropertyResult:
    """
    Compute exact ground-state properties for a validated ``ParsedHamiltonian``.

    This function is the main public interface for the exact backend.  It
    builds the Hamiltonian via the existing ``hamiltonians.build_hamiltonian_spec()``
    factory, runs exact diagonalization, and returns a fully populated
    ``PropertyResult``.

    Parameters
    ----------
    parsed : ParsedHamiltonian
        Output of ``parse_hamiltonian_text()``.  Must satisfy:
        ``supported=True``, ``family`` set, ``n_qubits`` set, all required
        parameters present, ``boundary="obc"``.

    Returns
    -------
    PropertyResult
        Populated with energy, magnetization, correlations, renyi2_entropy,
        and notes describing the computation details.

    Raises
    ------
    ValueError
        If validation fails (family unknown, n_qubits missing, required params
        absent, unsupported boundary, or ``supported=False``).
    """
    _validate_parsed_for_exact(parsed)

    # Narrow types: validation guarantees these are not None.
    family: str = parsed.family        # type: ignore[assignment]
    n_qubits: int = parsed.n_qubits   # type: ignore[assignment]
    params = parsed.params

    if n_qubits > _EXACT_WARN_THRESHOLD:
        _warnings.warn(
            f"Exact diagonalization for n_qubits={n_qubits} allocates a "
            f"{2**n_qubits}×{2**n_qubits} dense matrix "
            f"({(4 * 16 * 4**n_qubits) / 2**30:.1f} GiB at float64 complex). "
            "This may be very slow or exhaust memory.",
            UserWarning,
            stacklevel=2,
        )

    # ── Build HamiltonianSpec via the project's existing factory ───────────────
    spec: HamiltonianSpec = build_hamiltonian_spec(family, n_qubits, **params)

    # ── Exact diagonalization ──────────────────────────────────────────────────
    psi, E0 = _ground_state_from_spec(spec)

    # ── Observables ────────────────────────────────────────────────────────────
    energy = _exact_energy(E0)
    magnetization = _exact_magnetization(psi, n_qubits)
    correlations = _exact_correlations(psi, n_qubits)
    renyi2, k_sub = _exact_renyi2(psi, n_qubits)

    # ── Notes: record choices so callers / report generators can explain them ──
    notes: List[str] = [
        "Backend: exact diagonalization (numpy.linalg.eigh).",
        "Energy: total ground-state energy E0 = <psi|H|psi> (not normalised per site).",
        "Magnetization: average Z magnetization (1/n) sum_i <Z_i>.",
        "Correlations: average nearest-neighbor ZZ (1/(n-1)) sum_i <Z_i Z_{i+1}> (OBC).",
        f"Renyi-2 entropy: S2(rho_A) = -log Tr(rho_A^2), "
        f"subsystem A = first {k_sub} of {n_qubits} qubits (half-chain bipartition).",
    ]

    return PropertyResult(
        family=family,
        params=dict(params),
        n_qubits=n_qubits,
        energy=energy,
        magnetization=magnetization,
        correlations=correlations,
        renyi2_entropy=renyi2,
        notes=notes,
    )


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Convenience end-to-end function
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_and_evaluate_exact(text: str) -> PropertyResult:
    """
    Parse a free-text Hamiltonian description and immediately run exact evaluation.

    Equivalent to::

        parsed = parse_hamiltonian_text(text)
        return evaluate_exact(parsed)

    Raises ``ValueError`` (from ``evaluate_exact``) if the description does not
    produce a fully supported ``ParsedHamiltonian``.

    Parameters
    ----------
    text : str
        Human-readable description, e.g.
        ``"4-qubit TFIM with J=1 and h=0.8"``.

    Returns
    -------
    PropertyResult with exact ground-state properties.
    """
    parsed = parse_hamiltonian_text(text)
    return evaluate_exact(parsed)


__all__ = [
    "evaluate_exact",
    "parse_and_evaluate_exact",
    "evaluate_with_shadowgpt",
    "parse_and_evaluate_with_shadowgpt",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Learned ShadowGPT inference backend
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Checkpoint routing constants
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

_CHECKPOINTS_DIR_ENV = "SHADOWGPT_CHECKPOINTS_DIR"
_MODEL_FILENAME = "best_gpt.pt"
_TOKENIZER_FILENAME = "tokenizer_gpt.json"

# Parsed param key → uppercase token-name prefix used in vocab.
# TFIM uses encode_hamiltonian_prefix (1 d.p.); all others use
# encode_multi_hamiltonian_prefix (2 d.p.).
_PARAM_TOKEN_PREFIX: "Dict[str, Dict[str, str]]" = {
    "tfim":          {"J": "J_", "h": "H_"},
    "ising_general": {"J": "J_", "hx": "HX_", "hz": "HZ_"},
    "xxz":           {"J": "J_", "delta": "DELTA_"},
    "heisenberg":    {"J": "J_"},
}

# For encode_multi_hamiltonian_prefix: parsed param key → uppercase key expected
# by the tokenizer (must match the keys used when the tokenizer was built).
_PARAM_KEY_MAP: "Dict[str, Dict[str, str]]" = {
    "ising_general": {"J": "J", "hx": "HX", "hz": "HZ"},
    "xxz":           {"J": "J", "delta": "DELTA"},
    "heisenberg":    {"J": "J"},
}

# Maximum allowed distance between a requested param value and the nearest grid
# token before a UserWarning is emitted.
_SNAP_WARN_THRESHOLD: float = 0.05


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Input validation (learned backend)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _validate_parsed_for_learned(parsed: ParsedHamiltonian) -> None:
    """
    Raise ``ValueError`` if *parsed* is not ready for ShadowGPT evaluation.

    Applies the same checks as ``_validate_parsed_for_exact`` plus an
    additional guard that the Torch runtime is available.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for evaluate_with_shadowgpt() but is not installed. "
            "Install it with: pip install torch"
        )
    _validate_parsed_for_exact(parsed)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Checkpoint helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _resolve_checkpoint_dir(family: str, checkpoint_dir: "Optional[str]") -> str:
    """
    Return the path to the per-family checkpoint directory.

    Search order:
      1. ``checkpoint_dir`` argument (if not None).
      2. ``$SHADOWGPT_CHECKPOINTS_DIR`` environment variable.

    Within the resolved root, looks for ``{root}/{family}/``.

    Raises ``FileNotFoundError`` with an actionable message if the directory or
    either required file (``best_gpt.pt``, ``tokenizer_gpt.json``) is absent.
    """
    import os

    if checkpoint_dir is not None:
        root = checkpoint_dir
    else:
        root = os.environ.get(_CHECKPOINTS_DIR_ENV, "")
        if not root:
            raise FileNotFoundError(
                f"No checkpoint directory provided and the environment variable "
                f"'{_CHECKPOINTS_DIR_ENV}' is not set.\n"
                f"Set it with:  export {_CHECKPOINTS_DIR_ENV}=/path/to/checkpoints\n"
                f"or pass checkpoint_dir= to evaluate_with_shadowgpt()."
            )

    family_dir = os.path.join(root, family)
    if not os.path.isdir(family_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {family_dir!r}\n"
            f"Expected layout: {{root}}/{family}/{_MODEL_FILENAME} and "
            f"{{root}}/{family}/{_TOKENIZER_FILENAME}\n"
            f"Train a ShadowGPT for family='{family}' first, then save with "
            f"--output_dir pointing to {{root}}/{family}/."
        )

    for fname in (_MODEL_FILENAME, _TOKENIZER_FILENAME):
        fpath = os.path.join(family_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required checkpoint file missing: {fpath!r}\n"
                f"Ensure training completed and saved both {_MODEL_FILENAME} "
                f"and {_TOKENIZER_FILENAME} to {family_dir!r}."
            )

    return family_dir


def _load_family_tokenizer(ckpt_dir: str) -> "ShadowTokenizer":
    """Load and return the ``ShadowTokenizer`` saved in *ckpt_dir*.

    ``load_tokenizer`` is an instance method that populates an existing
    tokenizer in-place; it does not return a value.  We create a minimal
    instance first, then call the method on it.
    """
    import os
    from .tokenization import ShadowTokenizer, TokenizationConfig

    path = os.path.join(ckpt_dir, _TOKENIZER_FILENAME)
    tokenizer = ShadowTokenizer(TokenizationConfig())
    tokenizer.load_tokenizer(path)
    return tokenizer


def _load_family_model(ckpt_dir: str, device: str) -> "ShadowGPT":
    """
    Load the ``ShadowGPT`` from *ckpt_dir* onto *device*.

    The checkpoint is expected to be a dict with keys:
        ``"model_state"`` — ``state_dict`` saved by ``torch.save``
        ``"config"``      — ``GPTConfig`` instance
    (matching the format written by ``train.py``).
    """
    import os
    from .model import ShadowGPT, create_gpt_from_tokenizer

    path = os.path.join(ckpt_dir, _MODEL_FILENAME)
    ckpt = _torch.load(path, map_location=device)
    config = ckpt["config"]
    # Reconstruct model from GPTConfig; then load weights.
    model = ShadowGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Param snapping
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _snap_param_to_grid(
    param_key: str,
    value: float,
    token_prefix: str,
    vocab: "Dict[str, int]",
    family: str,
    notes: "List[str]",
) -> float:
    """
    Find the closest grid value for *param_key* in *vocab*.

    Scans all tokens whose name starts with *token_prefix* (e.g. ``"H_"``),
    decodes each to a float (by replacing ``"P"`` with ``"."`` in the suffix),
    and returns the value nearest to *value*.

    Emits a ``UserWarning`` if the snap distance exceeds ``_SNAP_WARN_THRESHOLD``.
    Records a note in *notes* about the snap.

    Returns the original *value* unchanged if no matching tokens are found
    (caller's responsibility to handle a subsequent KeyError on prefix lookup).
    """
    candidates: List[Tuple[float, str]] = []
    for tok in vocab:
        if tok.startswith(token_prefix):
            suffix = tok[len(token_prefix):]
            try:
                grid_val = float(suffix.replace("P", "."))
                candidates.append((grid_val, tok))
            except ValueError:
                pass

    if not candidates:
        return value

    snapped, _ = min(candidates, key=lambda x: abs(x[0] - value))
    dist = abs(snapped - value)
    if dist > 1e-9:
        if dist > _SNAP_WARN_THRESHOLD:
            _warnings.warn(
                f"[evaluate_with_shadowgpt] {family}/{param_key}={value} snapped to "
                f"{snapped} (distance {dist:.4f} > threshold {_SNAP_WARN_THRESHOLD}). "
                "The model may not have been trained for this parameter value.",
                UserWarning,
                stacklevel=4,
            )
        notes.append(
            f"Parameter snap: {param_key}={value} → {snapped} "
            f"(nearest grid token '{token_prefix}{f'{snapped:.2f}'.replace('.','P')}')."
        )
    return snapped


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Conditioning prefix builder
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _build_conditioning_prefix(
    parsed: ParsedHamiltonian,
    tokenizer: "ShadowTokenizer",
    notes: "List[str]",
) -> "List[int]":
    """
    Build the Hamiltonian conditioning prefix token ID list.

    For TFIM uses ``encode_hamiltonian_prefix`` (1 d.p. tokens).
    For all other families uses ``encode_multi_hamiltonian_prefix`` (2 d.p. tokens).

    Snaps each parameter value to the nearest grid token in the loaded tokenizer
    vocabulary, warning if the snap distance exceeds ``_SNAP_WARN_THRESHOLD``.

    Returns
    -------
    List[int] — token IDs: [MODEL_*, param_0, …, param_k, SEP]
    """
    from .tokenization import encode_hamiltonian_prefix, encode_multi_hamiltonian_prefix

    family: str = parsed.family        # type: ignore[assignment]
    params = parsed.params
    vocab = tokenizer.vocab
    token_prefixes = _PARAM_TOKEN_PREFIX.get(family, {})

    # Snap every param to its nearest grid value.
    snapped: "Dict[str, float]" = {}
    for pk, val in params.items():
        tp = token_prefixes.get(pk)
        if tp is not None:
            snapped[pk] = _snap_param_to_grid(pk, val, tp, vocab, family, notes)
        else:
            snapped[pk] = val

    if family == "tfim":
        return encode_hamiltonian_prefix(
            tokenizer,
            J=snapped.get("J", 1.0),
            h=snapped.get("h", 0.5),
        )

    # Non-TFIM: map param keys to uppercase, build dict for encode_multi_hamiltonian_prefix.
    key_map = _PARAM_KEY_MAP.get(family, {})
    upper_params: "Dict[str, float]" = {
        key_map.get(pk, pk): sv for pk, sv in snapped.items()
    }
    return encode_multi_hamiltonian_prefix(tokenizer, family, upper_params)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Shadow generation (inline, avoids train.py import chain)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _generate_synthetic_shadows(
    model: "ShadowGPT",
    tokenizer: "ShadowTokenizer",
    h_prefix: "List[int]",
    n_shadows: int,
    n_qubits: int,
    device: str,
    temperature: float,
    rng: np.ndarray,
) -> "ShadowCollector":
    """
    Autoregressively generate *n_shadows* synthetic shadow measurements.

    Replicates the logic of ``train.generate_shadows_from_gpt`` without
    importing from ``train.py``.

    For each shadow:
      1. Sample a random Pauli basis P ∈ {0=X, 1=Y, 2=Z}^n_qubits.
      2. Build [BOS, h_prefix…, basis_token_qubit_0].
      3. Sample outcome tokens autoregressively (restricted to GO0 / GO1).
      4. Collect (basis, outcome) into a ShadowCollector.
    """
    from .collector import ShadowCollector, ShadowMeasurement
    from .config import create_default_config

    model.eval()
    bos = tokenizer.special_tokens["BOS"]
    basis_ids   = [tokenizer.vocab[f"G{'XYZ'[b]}"] for b in range(3)]
    outcome_ids = [tokenizer.vocab["GO0"], tokenizer.vocab["GO1"]]

    all_bases:    "List[List[int]]" = []
    all_outcomes: "List[List[int]]" = []

    with _torch.no_grad():
        for _ in range(n_shadows):
            P = rng.integers(0, 3, size=n_qubits).tolist()
            current = [bos] + h_prefix + [basis_ids[P[0]]]
            outcomes: "List[int]" = []

            for q in range(n_qubits):
                ids_t = _torch.tensor([current], dtype=_torch.long, device=device)
                o_tok = model.generate_next_token(
                    ids_t, temperature=temperature, allowed_ids=outcome_ids
                )
                o_val = outcome_ids.index(o_tok)
                outcomes.append(o_val)
                current.append(o_tok)
                if q < n_qubits - 1:
                    current.append(basis_ids[P[q + 1]])

            all_bases.append(P)
            all_outcomes.append(outcomes)

    cfg = create_default_config(n_qubits=n_qubits, n_shadows=n_shadows)
    collector = ShadowCollector(cfg)
    collector.measurements = [
        ShadowMeasurement(
            basis=np.array(b, dtype=np.int32),
            outcome=np.array(o, dtype=np.int32),
        )
        for b, o in zip(all_bases, all_outcomes)
    ]
    return collector


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Property estimation from generated shadows
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _estimate_properties_from_generated_shadows(
    collector: "ShadowCollector",
    parsed: ParsedHamiltonian,
    notes: "List[str]",
) -> "Dict[str, Optional[float]]":
    """
    Run ShadowProcessor estimators on *collector* and return observable values.

    Estimates:
    - ``magnetization`` via ``estimate_magnetization`` (always).
    - ``correlations``  via ``estimate_correlations`` with ``correlation_length=1``
      (matches exact backend; n_qubits ≥ 2 required).
    - ``renyi2_entropy`` via ``estimate_renyi_entropy`` with subsystem size
      ``max(1, n_qubits // 2)`` (matches exact backend; needs ≥ 2 shadows).
    - ``energy``        via ``estimate_energy`` using ``spec.pauli_hamiltonian``
      (falls back to None if the Hamiltonian build fails or pyclifford is absent).

    Returns a dict with keys ``"energy"``, ``"magnetization"``, ``"correlations"``,
    ``"renyi2_entropy"`` mapping to float or None.
    """
    from .processor import ShadowProcessor
    from .config import create_default_config

    n_qubits = parsed.n_qubits      # type: ignore[assignment]
    family   = parsed.family        # type: ignore[assignment]
    params   = parsed.params

    # Configure processor: plain mean (median_of_means=False avoids needing many
    # shots per mean block), correlation_length=1 to match the exact backend.
    cfg = create_default_config(
        n_qubits=n_qubits,
        n_shadows=len(collector.measurements),
    )
    cfg.median_of_means = False
    cfg.correlation_length = 1
    cfg.renyi_entropy = False        # call directly with custom n_subsystem below
    cfg.observables = ["magnetization", "correlations"]

    proc = ShadowProcessor(cfg)
    results: "Dict[str, Optional[float]]" = {
        "energy": None,
        "magnetization": None,
        "correlations": None,
        "renyi2_entropy": None,
    }

    # ── Magnetization ──────────────────────────────────────────────────────────
    try:
        est = proc.estimate_magnetization(collector)
        results["magnetization"] = float(est.estimate)
        notes.append(
            f"Magnetization from {len(collector.measurements)} generated shadows "
            f"(plain mean, error={est.error:.4f})."
        )
    except Exception as exc:
        notes.append(f"Magnetization estimation failed: {exc}")

    # ── Correlations ───────────────────────────────────────────────────────────
    if n_qubits >= 2:
        try:
            est = proc.estimate_correlations(collector)
            results["correlations"] = float(est.estimate)
            notes.append(
                f"Correlations (nearest-neighbour ZZ, correlation_length=1) "
                f"from generated shadows (error={est.error:.4f})."
            )
        except Exception as exc:
            notes.append(f"Correlations estimation failed: {exc}")
    else:
        results["correlations"] = 0.0
        notes.append("Correlations set to 0 (n_qubits < 2, no pairs).")

    # ── Rényi-2 entropy ────────────────────────────────────────────────────────
    n_sub = max(1, n_qubits // 2)
    n_meas = len(collector.measurements)
    if n_meas >= 2:
        try:
            est = proc.estimate_renyi_entropy(collector, n_subsystem=n_sub)
            results["renyi2_entropy"] = float(est.estimate)
            notes.append(
                f"Rényi-2 entropy (subsystem A = first {n_sub} of {n_qubits} qubits, "
                f"paired-shot estimator) from generated shadows (error={est.error:.4f})."
            )
        except Exception as exc:
            notes.append(f"Rényi-2 entropy estimation failed: {exc}")
    else:
        notes.append("Rényi-2 entropy skipped (fewer than 2 shadows).")

    # ── Energy via pauli_hamiltonian ───────────────────────────────────────────
    try:
        spec: HamiltonianSpec = build_hamiltonian_spec(family, n_qubits, **params)
        if spec.pauli_hamiltonian is not None:
            est = proc.estimate_energy(collector, spec.pauli_hamiltonian)
            results["energy"] = float(est.estimate)
            notes.append(
                f"Energy from generated shadows via pauli_hamiltonian "
                f"(error={est.error:.4f})."
            )
        else:
            notes.append("Energy skipped (pauli_hamiltonian is None for this family).")
    except Exception as exc:
        notes.append(f"Energy estimation skipped: {exc}")

    return results


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Main entry point (learned backend)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def evaluate_with_shadowgpt(
    parsed: ParsedHamiltonian,
    checkpoint_dir: "Optional[str]" = None,
    n_shadows: int = 200,
    temperature: float = 1.0,
    device: "Optional[str]" = None,
    seed: "Optional[int]" = None,
) -> PropertyResult:
    """
    Estimate ground-state properties using the trained ShadowGPT model.

    This function is the main public interface for the learned backend.  It
    loads a family-specific checkpoint, autoregressively generates synthetic
    classical shadow measurements conditioned on the Hamiltonian parameters,
    and estimates physical observables from those shadows.

    Parameters
    ----------
    parsed : ParsedHamiltonian
        Output of ``parse_hamiltonian_text()``.  Must satisfy the same
        preconditions as ``evaluate_exact()``:
        ``supported=True``, ``family`` set, ``n_qubits`` set, all required
        parameters present, ``boundary="obc"``.
    checkpoint_dir : str or None
        Root directory containing per-family checkpoints laid out as::

            {checkpoint_dir}/{family}/best_gpt.pt
            {checkpoint_dir}/{family}/tokenizer_gpt.json

        If None, reads from the ``SHADOWGPT_CHECKPOINTS_DIR`` environment
        variable.  Raises ``FileNotFoundError`` if neither is set or the
        files are absent.
    n_shadows : int
        Number of synthetic shadow measurements to generate (default 200).
        Higher values reduce estimator variance.
    temperature : float
        Softmax temperature for autoregressive sampling (default 1.0).
        Values < 1 concentrate mass on high-probability tokens.
    device : str or None
        Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).  If None,
        auto-selects CUDA if available, else CPU.
    seed : int or None
        Random seed for the numpy basis-sampling RNG.  None = non-deterministic.

    Returns
    -------
    PropertyResult
        Populated with energy, magnetization, correlations, renyi2_entropy,
        and notes describing estimation details and any parameter snapping.

    Raises
    ------
    ImportError
        If PyTorch is not installed.
    ValueError
        If ``parsed`` fails validation (same conditions as ``evaluate_exact``).
    FileNotFoundError
        If the checkpoint directory or required files are absent.
    """
    _validate_parsed_for_learned(parsed)

    family: str   = parsed.family    # type: ignore[assignment]
    n_qubits: int = parsed.n_qubits  # type: ignore[assignment]

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)

    notes: List[str] = [
        f"Backend: ShadowGPT generative model (autoregressive shadow generation).",
        f"Device: {device}, n_shadows={n_shadows}, temperature={temperature}.",
    ]

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_dir  = _resolve_checkpoint_dir(family, checkpoint_dir)
    tokenizer = _load_family_tokenizer(ckpt_dir)
    model     = _load_family_model(ckpt_dir, device)

    notes.append(f"Checkpoint: {ckpt_dir!r}")

    # ── Build conditioning prefix (with param snapping) ───────────────────────
    h_prefix = _build_conditioning_prefix(parsed, tokenizer, notes)

    # ── Generate synthetic shadows ─────────────────────────────────────────────
    collector = _generate_synthetic_shadows(
        model=model,
        tokenizer=tokenizer,
        h_prefix=h_prefix,
        n_shadows=n_shadows,
        n_qubits=n_qubits,
        device=device,
        temperature=temperature,
        rng=rng,
    )

    # ── Estimate properties ────────────────────────────────────────────────────
    obs = _estimate_properties_from_generated_shadows(collector, parsed, notes)

    return PropertyResult(
        family=family,
        params=dict(parsed.params),
        n_qubits=n_qubits,
        energy=obs["energy"],
        magnetization=obs["magnetization"],
        correlations=obs["correlations"],
        renyi2_entropy=obs["renyi2_entropy"],
        notes=notes,
    )


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Convenience end-to-end function (learned backend)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_and_evaluate_with_shadowgpt(text: str, **kwargs) -> PropertyResult:
    """
    Parse a free-text Hamiltonian description and run the ShadowGPT backend.

    Equivalent to::

        parsed = parse_hamiltonian_text(text)
        return evaluate_with_shadowgpt(parsed, **kwargs)

    All keyword arguments are forwarded to ``evaluate_with_shadowgpt``
    (``checkpoint_dir``, ``n_shadows``, ``temperature``, ``device``, ``seed``).

    Parameters
    ----------
    text : str
        Human-readable description, e.g. ``"4-qubit TFIM with J=1 and h=0.8"``.

    Returns
    -------
    PropertyResult with shadow-estimated ground-state properties.
    """
    parsed = parse_hamiltonian_text(text)
    return evaluate_with_shadowgpt(parsed, **kwargs)
