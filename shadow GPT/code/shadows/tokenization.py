"""
Tokenization for classical shadow data to prepare for language model training.

This module provides the ShadowTokenizer class for converting classical shadow
measurements into token sequences suitable for training GPT-style transformers.

Tokenization modes
------------------
basis_outcome
    One token per qubit, encoding the (basis, outcome) pair.
    Basis is 0=X / 1=Y / 2=Z; outcome is 0 or 1.
    Vocabulary: B0O0, B0O1, B1O0, B1O1, B2O0, B2O1  (6 content tokens).
    Sequence length per measurement: n_qubits.

pauli_string
    One token per qubit, encoding only the measurement basis as a Pauli
    character ('X', 'Y', or 'Z').  Outcome information is intentionally
    discarded — this mode captures only which basis was measured, not the
    result.  Use basis_outcome if you need both.
    Vocabulary: X, Y, Z  (3 content tokens).
    Sequence length per measurement: n_qubits.

binary
    Character-level encoding of the full measurement: for each qubit,
    2 bits encode the basis (00=X, 01=Y, 10=Z) and 1 bit encodes the
    outcome, giving 3 characters per qubit.
    Vocabulary: '0', '1'  (2 content tokens).
    Sequence length per measurement: 3 * n_qubits.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .collector import ShadowCollector, ShadowMeasurement

# ShadowConfig is intentionally not imported — this module only needs
# the collector types.  Add the import here if future features require it.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TokenizationConfig:
    """Configuration for shadow data tokenization."""

    n_qubits: Optional[int] = None          # Required for pauli_string / binary modes
    vocab_size: int = 256                    # Informational upper bound (not enforced)
    max_sequence_length: int = 1024         # Maximum token-sequence length
    token_type: str = "basis_outcome"       # "basis_outcome" | "pauli_string" | "binary"
    special_tokens: Optional[Dict[str, int]] = None  # Override default BOS/EOS/PAD/UNK IDs
    padding_strategy: str = "right"         # "left" | "right" | "none"
    truncation_strategy: str = "right"      # "left" | "right" | "none"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_VALID_TOKEN_TYPES = {"basis_outcome", "pauli_string", "binary"}

_DEFAULT_SPECIAL_TOKENS: Dict[str, int] = {
    "BOS": 0,
    "EOS": 1,
    "PAD": 2,
    "UNK": 3,
}

_PAULI_MAP = {0: "X", 1: "Y", 2: "Z"}


class ShadowTokenizer:
    """
    Tokenizer for classical shadow measurements.

    Converts ShadowMeasurement objects into integer token sequences suitable
    for training GPT-style language models.
    """

    def __init__(self, config: TokenizationConfig) -> None:
        if config.token_type not in _VALID_TOKEN_TYPES:
            raise ValueError(
                f"Unknown token_type {config.token_type!r}. "
                f"Must be one of: {sorted(_VALID_TOKEN_TYPES)}."
            )

        self.config = config
        self.special_tokens: Dict[str, int] = dict(
            config.special_tokens if config.special_tokens is not None
            else _DEFAULT_SPECIAL_TOKENS
        )
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}

        self._build_vocabulary()

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def _build_vocabulary(self) -> None:
        """Populate vocab / reverse_vocab for the configured token_type."""
        # Register special tokens first so content token IDs start after them.
        self.vocab = dict(self.special_tokens)
        self.reverse_vocab = {v: k for k, v in self.special_tokens.items()}

        next_id = max(self.special_tokens.values()) + 1

        if self.config.token_type == "basis_outcome":
            # 3 bases × 2 outcomes = 6 content tokens
            for basis in range(3):
                for outcome in range(2):
                    token = f"B{basis}O{outcome}"
                    self.vocab[token] = next_id
                    self.reverse_vocab[next_id] = token
                    next_id += 1

        elif self.config.token_type == "pauli_string":
            # Character-level: one token per qubit — just X / Y / Z
            for ch in ("X", "Y", "Z"):
                self.vocab[ch] = next_id
                self.reverse_vocab[next_id] = ch
                next_id += 1

        elif self.config.token_type == "binary":
            # Character-level: one token per bit — '0' or '1'
            for ch in ("0", "1"):
                self.vocab[ch] = next_id
                self.reverse_vocab[next_id] = ch
                next_id += 1

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize_measurement(self, measurement: ShadowMeasurement) -> List[int]:
        """
        Tokenize a single ShadowMeasurement into a list of token IDs.

        Supported measurement modes: "random" and "pauli" (basis values 0=X,
        1=Y, 2=Z).  Clifford mode (basis values 0–23) and custom modes with
        non-Pauli basis labels are not supported and will raise ValueError.

        Args:
            measurement: A ShadowMeasurement with .basis and .outcome arrays
                         of equal length (n_qubits,).

        Returns:
            List of token IDs (length depends on token_type and n_qubits).
        """
        basis = np.asarray(measurement.basis)
        outcome = np.asarray(measurement.outcome)

        if basis.shape != outcome.shape:
            raise ValueError(
                f"basis shape {basis.shape} does not match outcome shape {outcome.shape}."
            )
        if basis.ndim != 1:
            raise ValueError("basis and outcome must be 1-D arrays.")
        if self.config.n_qubits is not None and len(basis) != self.config.n_qubits:
            raise ValueError(
                f"Measurement length {len(basis)} does not match "
                f"configured n_qubits={self.config.n_qubits}."
            )
        if len(basis) == 0:
            return []

        unk = self.special_tokens["UNK"]
        tokens: List[int] = []

        if self.config.token_type == "basis_outcome":
            for b, o in zip(basis.tolist(), outcome.tolist()):
                b_int, o_int = int(b), int(o)
                if b_int not in (0, 1, 2):
                    raise ValueError(
                        f"Invalid basis value {b_int} at qubit; "
                        f"expected 0 (X), 1 (Y), or 2 (Z)."
                    )
                if o_int not in (0, 1):
                    raise ValueError(
                        f"Invalid outcome value {o_int}; expected 0 or 1."
                    )
                tokens.append(self.vocab.get(f"B{b_int}O{o_int}", unk))

        elif self.config.token_type == "pauli_string":
            for b in basis.tolist():
                b_int = int(b)
                ch = _PAULI_MAP.get(b_int)
                if ch is None:
                    raise ValueError(
                        f"Invalid basis value {b_int} for pauli_string mode; "
                        f"expected 0 (X), 1 (Y), or 2 (Z)."
                    )
                tokens.append(self.vocab.get(ch, unk))

        elif self.config.token_type == "binary":
            for b, o in zip(basis.tolist(), outcome.tolist()):
                b_int, o_int = int(b), int(o)
                if b_int not in (0, 1, 2):
                    raise ValueError(
                        f"Invalid basis value {b_int} for binary mode; "
                        f"expected 0 (X), 1 (Y), or 2 (Z)."
                    )
                if o_int not in (0, 1):
                    raise ValueError(
                        f"Invalid outcome value {o_int}; expected 0 or 1."
                    )
                for ch in format(b_int, "02b") + format(o_int, "01b"):
                    tokens.append(self.vocab.get(ch, unk))

        return tokens

    def tokenize_collector(self, collector: ShadowCollector) -> List[List[int]]:
        """
        Tokenize all measurements in a ShadowCollector.

        Args:
            collector: ShadowCollector whose .measurements list is non-empty.

        Returns:
            List of token-ID lists, one per measurement.
        """
        if not collector.measurements:
            raise ValueError("No measurements to tokenize. Collect shadows first.")

        return [self.tokenize_measurement(m) for m in collector.measurements]

    def create_sequences(
        self,
        token_sequences: List[List[int]],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Wrap token sequences with BOS/EOS and apply padding/truncation.

        Args:
            token_sequences: Raw token-ID lists (e.g. from tokenize_collector).
            add_special_tokens: If True, prepend BOS and append EOS.

        Returns:
            Processed sequences ready for model input.
        """
        sequences = []
        for tokens in token_sequences:
            seq = []
            if add_special_tokens:
                seq.append(self.special_tokens["BOS"])
            seq.extend(tokens)
            if add_special_tokens:
                seq.append(self.special_tokens["EOS"])
            sequences.append(self._apply_length_constraints(seq))
        return sequences

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _handle_sequence_length(self, sequence: List[int]) -> List[int]:
        """Backward-compatibility alias for _apply_length_constraints."""
        return self._apply_length_constraints(sequence)

    def _apply_length_constraints(self, sequence: List[int]) -> List[int]:
        """Truncate then pad a single sequence according to config."""
        max_len = self.config.max_sequence_length

        # Truncation
        if len(sequence) > max_len:
            if self.config.truncation_strategy == "right":
                sequence = sequence[:max_len]
            elif self.config.truncation_strategy == "left":
                sequence = sequence[-max_len:]
            # "none" → no truncation

        # Padding
        if self.config.padding_strategy != "none" and len(sequence) < max_len:
            pad = self.special_tokens["PAD"]
            deficit = max_len - len(sequence)
            if self.config.padding_strategy == "right":
                sequence = sequence + [pad] * deficit
            elif self.config.padding_strategy == "left":
                sequence = [pad] * deficit + sequence

        return sequence

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back to a human-readable string."""
        return " ".join(
            self.reverse_vocab.get(tid, "UNK") for tid in token_ids
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_tokenizer(self, filepath: str) -> None:
        """Save tokenizer config and vocabulary to a JSON file."""
        data = {
            "config": self.config.__dict__,
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

        self.config = TokenizationConfig(**data["config"])
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
        """Return a copy of the special-token id mapping."""
        return dict(self.special_tokens)

    def __repr__(self) -> str:
        return (
            f"ShadowTokenizer(token_type={self.config.token_type!r}, "
            f"vocab_size={self.get_vocab_size()}, "
            f"max_length={self.config.max_sequence_length})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_default_tokenizer(
    n_qubits: int,
    token_type: str = "basis_outcome",
) -> ShadowTokenizer:
    """
    Create a default ShadowTokenizer for an n-qubit system.

    Args:
        n_qubits: Number of qubits per measurement.
        token_type: One of "basis_outcome", "pauli_string", "binary".

    Returns:
        A ready-to-use ShadowTokenizer.
    """
    if token_type not in _VALID_TOKEN_TYPES:
        raise ValueError(
            f"Unknown token_type {token_type!r}. "
            f"Must be one of: {sorted(_VALID_TOKEN_TYPES)}."
        )

    # Tokens per measurement (excluding BOS/EOS)
    tokens_per_measurement = 3 * n_qubits if token_type == "binary" else n_qubits
    max_seq_len = tokens_per_measurement + 2  # +2 for BOS / EOS

    config = TokenizationConfig(
        n_qubits=n_qubits,
        vocab_size=256,
        max_sequence_length=max_seq_len,
        token_type=token_type,
    )
    return ShadowTokenizer(config)
