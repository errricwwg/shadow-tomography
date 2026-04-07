"""
Tokenization for classical shadow data to prepare for language model training.

This module provides the ShadowTokenizer class for converting classical shadow
measurements into token sequences suitable for training GPT-style transformers.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import json
import os

from .config import ShadowConfig
from .collector import ShadowCollector, ShadowMeasurement


@dataclass
class TokenizationConfig:
    """Configuration for shadow data tokenization."""
    vocab_size: int = 256  # Vocabulary size for tokens
    max_sequence_length: int = 1024  # Maximum sequence length
    token_type: str = "basis_outcome"  # "basis_outcome", "pauli_string", "binary"
    special_tokens: Dict[str, int] = None  # Special tokens (BOS, EOS, PAD, etc.)
    padding_strategy: str = "right"  # "left", "right", "none"
    truncation_strategy: str = "right"  # "left", "right", "none"


class ShadowTokenizer:
    """
    Tokenizer for classical shadow data.
    
    Converts shadow measurements into token sequences suitable for training
    language models. Supports different tokenization strategies and formats.
    """
    
    def __init__(self, config: TokenizationConfig):
        """
        Initialize shadow tokenizer.
        
        Args:
            config: Tokenization configuration
        """
        self.config = config
        self.vocab = {}
        self.reverse_vocab = {}
        self.special_tokens = config.special_tokens or {
            "BOS": 0,  # Beginning of sequence
            "EOS": 1,  # End of sequence
            "PAD": 2,  # Padding
            "UNK": 3   # Unknown token
        }
        
        # Initialize vocabulary
        self._build_vocabulary()
    
    def _build_vocabulary(self) -> None:
        """Build vocabulary for tokenization."""
        # TODO: Implement proper vocabulary construction
        # This would involve:
        # 1. Analyzing the shadow data to determine token types
        # 2. Creating mappings for different measurement bases and outcomes
        # 3. Handling special tokens and unknown tokens
        
        print(f"TODO: Implement vocabulary construction for {self.config.token_type} tokenization")
        
        # Placeholder implementation
        token_id = len(self.special_tokens)
        
        if self.config.token_type == "basis_outcome":
            # Tokens for basis-outcome pairs
            for basis in range(3):  # X, Y, Z
                for outcome in range(2):  # 0, 1
                    token = f"B{basis}O{outcome}"
                    self.vocab[token] = token_id
                    self.reverse_vocab[token_id] = token
                    token_id += 1
        
        elif self.config.token_type == "pauli_string":
            # Tokens for Pauli strings
            for i in range(self.config.vocab_size - len(self.special_tokens)):
                token = f"PAULI_{i}"
                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token
                token_id += 1
        
        elif self.config.token_type == "binary":
            # Binary tokens
            for i in range(self.config.vocab_size - len(self.special_tokens)):
                token = f"BIN_{i:08b}"
                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token
                token_id += 1
        
        # Add special tokens to vocabulary
        for token, token_id in self.special_tokens.items():
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token
    
    def tokenize_measurement(self, measurement: ShadowMeasurement) -> List[int]:
        """
        Tokenize a single shadow measurement.
        
        Args:
            measurement: Shadow measurement to tokenize
            
        Returns:
            List of token IDs
        """
        # TODO: Implement proper measurement tokenization
        # This would involve converting the measurement basis and outcome
        # into appropriate tokens based on the tokenization strategy
        
        tokens = []
        
        if self.config.token_type == "basis_outcome":
            # Tokenize basis-outcome pairs
            for i in range(len(measurement.basis)):
                basis = int(measurement.basis[i])
                outcome = int(measurement.outcome[i])
                token = f"B{basis}O{outcome}"
                tokens.append(self.vocab.get(token, self.special_tokens["UNK"]))
        
        elif self.config.token_type == "pauli_string":
            # Tokenize as Pauli string
            pauli_string = self._basis_to_pauli_string(measurement.basis)
            tokens.append(self.vocab.get(pauli_string, self.special_tokens["UNK"]))
        
        elif self.config.token_type == "binary":
            # Tokenize as binary representation
            binary_repr = self._measurement_to_binary(measurement)
            tokens.append(self.vocab.get(binary_repr, self.special_tokens["UNK"]))
        
        return tokens
    
    def tokenize_collector(self, collector: ShadowCollector) -> List[List[int]]:
        """
        Tokenize all measurements from a shadow collector.
        
        Args:
            collector: ShadowCollector with measurements
            
        Returns:
            List of token sequences (one per measurement)
        """
        if not collector.measurements:
            raise ValueError("No measurements to tokenize. Collect shadows first.")
        
        token_sequences = []
        
        for measurement in collector.measurements:
            tokens = self.tokenize_measurement(measurement)
            token_sequences.append(tokens)
        
        return token_sequences
    
    def _basis_to_pauli_string(self, basis: np.ndarray) -> str:
        """Convert measurement basis to Pauli string representation."""
        # TODO: Implement proper Pauli string conversion
        # This would convert basis indices to Pauli operators (X, Y, Z)
        
        pauli_map = {0: "X", 1: "Y", 2: "Z"}
        pauli_string = "".join([pauli_map.get(int(b), "I") for b in basis])
        return pauli_string
    
    def _measurement_to_binary(self, measurement: ShadowMeasurement) -> str:
        """Convert measurement to binary representation."""
        # TODO: Implement proper binary conversion
        # This would encode both basis and outcome information
        
        # Simple binary encoding: basis as 2-bit, outcome as 1-bit
        binary_repr = ""
        for i in range(len(measurement.basis)):
            basis_bits = format(int(measurement.basis[i]), "02b")
            outcome_bit = format(int(measurement.outcome[i]), "01b")
            binary_repr += basis_bits + outcome_bit
        
        return binary_repr
    
    def create_sequences(self, token_sequences: List[List[int]], 
                        add_special_tokens: bool = True) -> List[List[int]]:
        """
        Create training sequences from token sequences.
        
        Args:
            token_sequences: List of token sequences
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of training sequences
        """
        sequences = []
        
        for tokens in token_sequences:
            sequence = []
            
            if add_special_tokens:
                sequence.append(self.special_tokens["BOS"])
            
            sequence.extend(tokens)
            
            if add_special_tokens:
                sequence.append(self.special_tokens["EOS"])
            
            # Handle sequence length constraints
            sequence = self._handle_sequence_length(sequence)
            sequences.append(sequence)
        
        return sequences
    
    def _handle_sequence_length(self, sequence: List[int]) -> List[int]:
        """Handle sequence length constraints (truncation and padding)."""
        max_length = self.config.max_sequence_length
        
        if len(sequence) > max_length:
            if self.config.truncation_strategy == "right":
                sequence = sequence[:max_length]
            elif self.config.truncation_strategy == "left":
                sequence = sequence[-max_length:]
            elif self.config.truncation_strategy == "none":
                pass  # Keep full sequence
        
        if len(sequence) < max_length and self.config.padding_strategy != "none":
            pad_token = self.special_tokens["PAD"]
            padding_length = max_length - len(sequence)
            
            if self.config.padding_strategy == "right":
                sequence.extend([pad_token] * padding_length)
            elif self.config.padding_strategy == "left":
                sequence = [pad_token] * padding_length + sequence
        
        return sequence
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text representation.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Text representation of the sequence
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append("UNK")
        
        return " ".join(tokens)
    
    def save_tokenizer(self, filepath: str) -> None:
        """
        Save tokenizer configuration and vocabulary.
        
        Args:
            filepath: Path to save tokenizer
        """
        tokenizer_data = {
            "config": self.config.__dict__,
            "vocab": self.vocab,
            "reverse_vocab": {str(k): v for k, v in self.reverse_vocab.items()},
            "special_tokens": self.special_tokens
        }
        
        with open(filepath, 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
        
        print(f"Saved tokenizer to {filepath}")
    
    def load_tokenizer(self, filepath: str) -> None:
        """
        Load tokenizer configuration and vocabulary.
        
        Args:
            filepath: Path to load tokenizer from
        """
        with open(filepath, 'r') as f:
            tokenizer_data = json.load(f)
        
        # Update configuration
        self.config = TokenizationConfig(**tokenizer_data["config"])
        
        # Update vocabulary
        self.vocab = tokenizer_data["vocab"]
        self.reverse_vocab = {int(k): v for k, v in tokenizer_data["reverse_vocab"].items()}
        self.special_tokens = tokenizer_data["special_tokens"]
        
        print(f"Loaded tokenizer from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        return self.special_tokens.copy()
    
    def __repr__(self) -> str:
        return (f"ShadowTokenizer(vocab_size={self.get_vocab_size()}, "
                f"token_type={self.config.token_type}, "
                f"max_length={self.config.max_sequence_length})")


def create_default_tokenizer(n_qubits: int, token_type: str = "basis_outcome") -> ShadowTokenizer:
    """
    Create a default tokenizer for shadow data.
    
    Args:
        n_qubits: Number of qubits
        token_type: Type of tokenization strategy
        
    Returns:
        Default shadow tokenizer
    """
    config = TokenizationConfig(
        vocab_size=256,
        max_sequence_length=n_qubits * 4,  # Reasonable default
        token_type=token_type
    )
    
    return ShadowTokenizer(config)
