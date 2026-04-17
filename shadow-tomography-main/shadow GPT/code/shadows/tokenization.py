"""
Tokenization for the ShadowGPT generative pipeline.

Primary interface
-----------------
    create_generative_tokenizer(n_qubits, family, ...)
        One-call factory: builds a ShadowTokenizer, adds Hamiltonian
        conditioning tokens for the chosen family, and adds generative
        basis/outcome tokens (GX/GY/GZ, GO0/GO1).  Returns a tokenizer
        ready for build_generative_sequence().

    add_generative_tokens(tokenizer)
        Extend an existing tokenizer with GX, GY, GZ, GO0, GO1 in-place.

    build_generative_sequence(tokenizer, h_prefix, basis, outcome)
        Encode one shadow measurement as:
            [BOS] h_prefix [GX|GY|GZ  GO0|GO1] × n_qubits [EOS]

    decode_generative_outcomes(tokenizer, token_ids, n_qubits, h_prefix_len)
        Recover (basis, outcome) arrays from a generated sequence.

    encode_hamiltonian_prefix / encode_multi_hamiltonian_prefix
        Build the Hamiltonian conditioning prefix (MODEL + params + SEP).

Generative token vocabulary
---------------------------
    GX, GY, GZ     — Pauli basis indicators (one per qubit step)
    GO0, GO1        — Measurement outcomes (0 or 1)

ShadowTokenizer is a vocabulary container: it holds BOS/EOS/PAD/UNK special
tokens plus any domain-specific tokens added by the extension functions above.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_SPECIAL_TOKENS: Dict[str, int] = {
    "BOS": 0,
    "EOS": 1,
    "PAD": 2,
    "UNK": 3,
}


@dataclass
class TokenizationConfig:
    """Configuration for ShadowTokenizer."""
    n_qubits: Optional[int] = None          # informational; used in __repr__
    max_sequence_length: int = 1024         # maximum token-sequence length
    special_tokens: Optional[Dict[str, int]] = None  # override default IDs


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class ShadowTokenizer:
    """
    Vocabulary container for the ShadowGPT generative pipeline.

    Holds a mapping from token names to integer IDs and provides
    save/load and introspection helpers.  Domain-specific tokens
    (Hamiltonian conditioning tokens, generative basis/outcome tokens)
    are added in-place by the extension functions in this module.
    """

    def __init__(self, config: TokenizationConfig) -> None:
        self.config = config
        self.special_tokens: Dict[str, int] = dict(
            config.special_tokens if config.special_tokens is not None
            else _DEFAULT_SPECIAL_TOKENS
        )
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.reverse_vocab: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_tokenizer(self, filepath: str) -> None:
        """Save tokenizer config and vocabulary to a JSON file."""
        data = {
            "config": {
                "n_qubits": self.config.n_qubits,
                "max_sequence_length": self.config.max_sequence_length,
            },
            "vocab": self.vocab,
            "reverse_vocab": {str(k): v for k, v in self.reverse_vocab.items()},
            "special_tokens": self.special_tokens,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_tokenizer(self, filepath: str) -> None:
        """Load tokenizer config and vocabulary from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        cfg = data["config"]
        self.config = TokenizationConfig(
            n_qubits=cfg.get("n_qubits"),
            max_sequence_length=cfg.get("max_sequence_length", 1024),
        )
        self.special_tokens = data["special_tokens"]
        self.vocab = data["vocab"]
        self.reverse_vocab = {int(k): v for k, v in data["reverse_vocab"].items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        """Return the number of distinct tokens in the vocabulary."""
        return len(self.vocab)

    def get_special_token_ids(self) -> Dict[str, int]:
        """Return a copy of the special-token ID mapping."""
        return dict(self.special_tokens)

    def __repr__(self) -> str:
        return (
            f"ShadowTokenizer("
            f"vocab_size={self.get_vocab_size()}, "
            f"max_length={self.config.max_sequence_length})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_default_tokenizer(n_qubits: int) -> ShadowTokenizer:
    """
    Create a base ShadowTokenizer with BOS/EOS/PAD/UNK special tokens.

    Extension functions (add_hamiltonian_conditioning,
    add_multi_hamiltonian_conditioning, add_generative_tokens) are called
    on top of this to build the full generative vocabulary.
    Prefer create_generative_tokenizer() for the complete one-call setup.

    Args:
        n_qubits: Number of qubits (stored in config for reference).

    Returns:
        ShadowTokenizer with only special tokens in vocab.
    """
    config = TokenizationConfig(
        n_qubits=n_qubits,
        max_sequence_length=n_qubits + 2,   # placeholder; overridden by extensions
    )
    return ShadowTokenizer(config)


# ---------------------------------------------------------------------------
# Hamiltonian conditioning extension  (TFIM)
# ---------------------------------------------------------------------------

def add_hamiltonian_conditioning(
    tokenizer: ShadowTokenizer,
    tfim_J: float,
    tfim_h_values: List[float],
) -> None:
    """
    Extend a ShadowTokenizer with TFIM Hamiltonian conditioning tokens.

    Adds (idempotent):
        MODEL_TFIM        — family identifier
        J_{J_str}         — one token for the J value  (e.g. "J_1P0")
        H_{h_str} × N     — one token per value in tfim_h_values (e.g. "H_0P5")
        SEP               — separator between conditioning prefix and measurements

    Increments max_sequence_length by 4 to fit:
        [BOS, MODEL_TFIM, J_xxx, H_xxx, SEP, meas_tokens…, EOS]

    Args:
        tokenizer:       ShadowTokenizer to extend in-place.
        tfim_J:          J value; token "J_XPY" is created for it.
        tfim_h_values:   Discrete h values; one "H_XPY" token per value.
    """
    next_id = max(tokenizer.vocab.values()) + 1

    def _add(name: str) -> None:
        nonlocal next_id
        if name in tokenizer.vocab:
            return
        tokenizer.vocab[name] = next_id
        tokenizer.reverse_vocab[next_id] = name
        next_id += 1

    _add("MODEL_TFIM")
    _add(f"J_{tfim_J:.1f}".replace(".", "P"))
    for h in tfim_h_values:
        _add(f"H_{h:.1f}".replace(".", "P"))
    _add("SEP")

    tokenizer.config.max_sequence_length += 4  # MODEL + J + H + SEP


def encode_hamiltonian_prefix(
    tokenizer: ShadowTokenizer,
    J: float,
    h: float,
) -> List[int]:
    """
    Return the 4 TFIM conditioning token IDs for a given (J, h) pair.

    Token order: [MODEL_TFIM, J_xxx, H_xxx, SEP]

    Args:
        tokenizer: Extended by add_hamiltonian_conditioning().
        J:         ZZ coupling (must match a J_xxx token in vocab).
        h:         Transverse field (rounded to 1 d.p.; must match an H_xxx token).

    Returns:
        List of 4 integer token IDs.
    """
    j_str = f"J_{J:.1f}".replace(".", "P")
    h_str = f"H_{round(h, 1):.1f}".replace(".", "P")
    return [
        tokenizer.vocab["MODEL_TFIM"],
        tokenizer.vocab[j_str],
        tokenizer.vocab[h_str],
        tokenizer.vocab["SEP"],
    ]


# ---------------------------------------------------------------------------
# Multi-family Hamiltonian conditioning
# ---------------------------------------------------------------------------

_FAMILY_MODEL_TOKENS: Dict[str, str] = {
    "tfim":          "MODEL_TFIM",
    "ising_general": "MODEL_ISING",
    "xxz":           "MODEL_XXZ",
    "heisenberg":    "MODEL_HEIS",
}


def add_multi_hamiltonian_conditioning(
    tokenizer: ShadowTokenizer,
    family: str,
    param_grids: Dict[str, List[float]],
) -> None:
    """
    Extend a ShadowTokenizer with multi-family Hamiltonian conditioning tokens.

    Adds (idempotent):
        MODEL_{FAMILY}          — e.g. MODEL_XXZ, MODEL_HEIS, MODEL_ISING
        {PARAM}_{XPY} × N       — one token per (param-name, grid-value) pair,
                                   formatted as 2 d.p. with '.' → 'P'
        SEP                     — shared separator

    Increments max_sequence_length by (1 + len(param_grids) + 1).

    Args:
        tokenizer:   ShadowTokenizer to extend in-place.
        family:      One of "tfim", "ising_general", "xxz", "heisenberg".
        param_grids: param_name → list of all grid values that appear during training.
    """
    if family not in _FAMILY_MODEL_TOKENS:
        raise ValueError(
            f"Unknown Hamiltonian family {family!r}. "
            f"Supported: {sorted(_FAMILY_MODEL_TOKENS)}."
        )

    next_id = max(tokenizer.vocab.values()) + 1

    def _add(name: str) -> None:
        nonlocal next_id
        if name in tokenizer.vocab:
            return
        tokenizer.vocab[name] = next_id
        tokenizer.reverse_vocab[next_id] = name
        next_id += 1

    _add(_FAMILY_MODEL_TOKENS[family])
    for param_name, values in param_grids.items():
        for v in values:
            _add(f"{param_name}_{v:.2f}".replace(".", "P"))
    _add("SEP")

    tokenizer.config.max_sequence_length += 1 + len(param_grids) + 1


def encode_multi_hamiltonian_prefix(
    tokenizer: ShadowTokenizer,
    family: str,
    params: Dict[str, float],
) -> List[int]:
    """
    Return conditioning token IDs for a specific Hamiltonian family + parameters.

    Token order: [MODEL_{FAMILY}, param_0_tok, …, param_n_tok, SEP]

    Args:
        tokenizer: Extended by add_multi_hamiltonian_conditioning().
        family:    Hamiltonian family name.
        params:    param_name → value (rounded to 2 d.p. for token lookup).

    Returns:
        List of 1 + len(params) + 1 integer token IDs.
    """
    ids = [tokenizer.vocab[_FAMILY_MODEL_TOKENS[family]]]
    for param_name, v in params.items():
        tok = f"{param_name}_{round(v, 2):.2f}".replace(".", "P")
        ids.append(tokenizer.vocab[tok])
    ids.append(tokenizer.vocab["SEP"])
    return ids


# ---------------------------------------------------------------------------
# Generative token extension
# ---------------------------------------------------------------------------

_GEN_BASIS_NAMES: Dict[int, str]   = {0: "GX", 1: "GY", 2: "GZ"}
_GEN_OUTCOME_NAMES: Dict[int, str] = {0: "GO0", 1: "GO1"}


def add_generative_tokens(tokenizer: ShadowTokenizer) -> None:
    """
    Extend a ShadowTokenizer with generative basis/outcome tokens (idempotent).

    Adds:
        GX, GY, GZ   — Pauli basis tokens (one per qubit per step)
        GO0, GO1     — Measurement outcome tokens (0 or 1)

    These tokens form the measurement part of every generative sequence:
        [BOS] [g_tokens…] [GX|GY|GZ  GO0|GO1] × n_qubits [EOS]
    """
    next_id = max(tokenizer.vocab.values()) + 1

    def _add(name: str) -> None:
        nonlocal next_id
        if name in tokenizer.vocab:
            return
        tokenizer.vocab[name] = next_id
        tokenizer.reverse_vocab[next_id] = name
        next_id += 1

    for name in _GEN_BASIS_NAMES.values():
        _add(name)
    for name in _GEN_OUTCOME_NAMES.values():
        _add(name)


def build_generative_sequence(
    tokenizer: ShadowTokenizer,
    h_prefix: List[int],
    basis: np.ndarray,
    outcome: np.ndarray,
) -> List[int]:
    """
    Build a single generative training sequence.

    Format:
        [BOS] h_prefix [GX|GY|GZ  GO0|GO1] × n_qubits [EOS]

    Args:
        tokenizer: ShadowTokenizer extended with add_generative_tokens().
        h_prefix:  Conditioning token IDs (MODEL + params + SEP).
        basis:     1-D int array, length n_qubits; values 0=X, 1=Y, 2=Z.
        outcome:   1-D int array, length n_qubits; values 0 or 1.

    Returns:
        List of integer token IDs (no padding; fixed length).
    """
    bos = tokenizer.special_tokens["BOS"]
    eos = tokenizer.special_tokens["EOS"]
    meas: List[int] = []
    for b, o in zip(basis.tolist(), outcome.tolist()):
        meas.append(tokenizer.vocab[_GEN_BASIS_NAMES[int(b)]])
        meas.append(tokenizer.vocab[_GEN_OUTCOME_NAMES[int(o)]])
    return [bos] + h_prefix + meas + [eos]


def decode_generative_outcomes(
    tokenizer: ShadowTokenizer,
    token_ids: List[int],
    n_qubits: int,
    h_prefix_len: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract (basis, outcome) arrays from a generated token sequence.

    The measurement block starts at position 1 + h_prefix_len (BOS + prefix).
    Each qubit occupies two tokens: [GX|GY|GZ, GO0|GO1].

    Args:
        tokenizer:    ShadowTokenizer with generative tokens.
        token_ids:    Full generated sequence (may include EOS/PAD at the end).
        n_qubits:     Expected number of (basis, outcome) pairs.
        h_prefix_len: Number of conditioning tokens (after BOS, before measurements).

    Returns:
        (basis, outcome) as int8 numpy arrays of shape (n_qubits,), or None if
        the sequence is too short or contains unexpected tokens.
    """
    basis_rev:   Dict[int, int] = {tokenizer.vocab[v]: k for k, v in _GEN_BASIS_NAMES.items()}
    outcome_rev: Dict[int, int] = {tokenizer.vocab[v]: k for k, v in _GEN_OUTCOME_NAMES.items()}

    meas_start = 1 + h_prefix_len
    meas_end   = meas_start + 2 * n_qubits

    if len(token_ids) < meas_end:
        return None

    basis_arr   = np.empty(n_qubits, dtype=np.int8)
    outcome_arr = np.empty(n_qubits, dtype=np.int8)
    for q in range(n_qubits):
        b_tok = token_ids[meas_start + 2 * q]
        o_tok = token_ids[meas_start + 2 * q + 1]
        if b_tok not in basis_rev or o_tok not in outcome_rev:
            return None
        basis_arr[q]   = basis_rev[b_tok]
        outcome_arr[q] = outcome_rev[o_tok]

    return basis_arr, outcome_arr


def create_generative_tokenizer(
    n_qubits: int,
    family: str = "tfim",
    tfim_h_min: float = 0.1,
    tfim_h_max: float = 2.0,
    tfim_J: float = 1.0,
    ising_J: float = 1.0,
    xxz_J: float = 1.0,
    xxz_delta_min: float = 0.0,
    xxz_delta_max: float = 2.0,
    heis_J_min: float = 0.5,
    heis_J_max: float = 2.0,
) -> ShadowTokenizer:
    """
    Build a ShadowTokenizer ready for the generative ShadowGPT pipeline.

    Steps:
    1. Create a base tokenizer with BOS/EOS/PAD/UNK.
    2. Add Hamiltonian conditioning tokens for the chosen family.
    3. Add generative basis/outcome tokens (GX, GY, GZ, GO0, GO1).
    4. Set max_sequence_length = 1 + h_prefix_len + 2*n_qubits + 1.

    Args:
        n_qubits:       Number of qubits.
        family:         Hamiltonian family — 'tfim', 'ising_general', 'xxz', 'heisenberg'.
        tfim_h_min:     TFIM minimum transverse field h.
        tfim_h_max:     TFIM maximum transverse field h.
        tfim_J:         TFIM ZZ coupling.
        ising_J:        Ising-general ZZ coupling.
        xxz_J:          XXZ exchange coupling.
        xxz_delta_min:  XXZ minimum delta.
        xxz_delta_max:  XXZ maximum delta.
        heis_J_min:     Heisenberg minimum J.
        heis_J_max:     Heisenberg maximum J.

    Returns:
        ShadowTokenizer ready for build_generative_sequence().
    """
    tokenizer = create_default_tokenizer(n_qubits)

    if family == "tfim":
        h_grid = np.round(np.arange(tfim_h_min, tfim_h_max + 1e-9, 0.1), 2).tolist()
        add_hamiltonian_conditioning(
            tokenizer,
            tfim_J=round(tfim_J, 2),
            tfim_h_values=h_grid,
        )
    elif family == "ising_general":
        hx_grid = np.round(np.arange(tfim_h_min, tfim_h_max + 1e-9, 0.1), 2).tolist()
        add_multi_hamiltonian_conditioning(
            tokenizer, "ising_general",
            {"J": [round(ising_J, 2)], "HX": hx_grid, "HZ": [0.00, 0.20, 0.50, 1.00]},
        )
    elif family == "xxz":
        delta_grid = np.round(np.arange(xxz_delta_min, xxz_delta_max + 1e-9, 0.25), 2).tolist()
        add_multi_hamiltonian_conditioning(
            tokenizer, "xxz",
            {"J": [round(xxz_J, 2)], "DELTA": delta_grid},
        )
    elif family == "heisenberg":
        J_grid = np.round(np.arange(heis_J_min, heis_J_max + 1e-9, 0.25), 2).tolist()
        add_multi_hamiltonian_conditioning(
            tokenizer, "heisenberg",
            {"J": J_grid},
        )
    else:
        raise ValueError(
            f"Unknown family: {family!r}. "
            f"Choose from 'tfim', 'ising_general', 'xxz', 'heisenberg'."
        )

    add_generative_tokens(tokenizer)

    # h_prefix_len: MODEL token + one per param + SEP
    _param_counts = {"tfim": 2, "ising_general": 3, "xxz": 2, "heisenberg": 1}
    h_prefix_len = 1 + _param_counts[family] + 1

    tokenizer.config.max_sequence_length = 1 + h_prefix_len + 2 * n_qubits + 1
    return tokenizer
