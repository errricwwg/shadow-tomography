"""
PyTorch dataset for classical shadow data.

This module provides the ShadowDataset class for loading and processing
classical shadow data for training language models.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .collector import ShadowCollector
from .tokenization import ShadowTokenizer, TokenizationConfig


@dataclass
class DatasetConfig:
    """Configuration for shadow dataset."""
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    # Set to an integer for reproducible train/val/test splitting.
    # None means the global numpy random state is used (non-reproducible).
    shuffle_seed: Optional[int] = None

    def __post_init__(self) -> None:
        for name, val in [
            ("train_split", self.train_split),
            ("val_split", self.val_split),
            ("test_split", self.test_split),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}.")
        total = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"train_split + val_split + test_split must sum to ~1.0, "
                f"got {total:.4f}."
            )


class ShadowDataset(Dataset):
    """
    PyTorch dataset for classical shadow data.

    Loads shadow measurements and converts them to token sequences
    suitable for training language models.
    """

    def __init__(
        self,
        token_sequences: List[List[int]],
        tokenizer: ShadowTokenizer,
        config: Optional[DatasetConfig] = None,
    ) -> None:
        """
        Initialize shadow dataset.

        Args:
            token_sequences: List of token sequences
            tokenizer: ShadowTokenizer for processing
            config: Dataset configuration (uses defaults if None)
        """
        self.token_sequences = token_sequences
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig()

        # Convert to tensors
        self.sequences = self._prepare_sequences()

    def _prepare_sequences(self) -> List[torch.Tensor]:
        """Prepare token sequences as PyTorch tensors."""
        sequences = []
        for tokens in self.token_sequences:
            sequence = torch.tensor(tokens, dtype=torch.long)
            sequences.append(sequence)
        return sequences

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            idx: Index of the item

        Returns:
            Dictionary with input_ids and labels for language modeling.
            For a sequence [t0, t1, ..., tN]:
                input_ids = [t0, ..., t_{N-1}]
                labels    = [t1, ..., tN]
        """
        sequence = self.sequences[idx]

        # For causal language modeling: input is sequence[:-1], target is sequence[1:]
        input_ids = sequence[:-1] if len(sequence) > 1 else sequence
        labels = sequence[1:] if len(sequence) > 1 else sequence

        return {"input_ids": input_ids, "labels": labels}

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching variable-length sequences.

        Pads input_ids with PAD token and labels with -100 (ignored by
        cross-entropy loss).  Attention mask is aligned with input_ids padding.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched tensors: input_ids, labels, attention_mask
        """
        max_length = max(len(sample["input_ids"]) for sample in batch)
        pad_token = self.tokenizer.special_tokens["PAD"]

        input_ids_list = []
        labels_list = []
        attention_masks_list = []

        for sample in batch:
            input_seq = sample["input_ids"]
            label_seq = sample["labels"]

            # Compute padding amounts separately to keep attention mask correct.
            input_pad = max_length - len(input_seq)
            label_pad = max_length - len(label_seq)

            if input_pad > 0:
                input_seq = torch.cat([
                    input_seq,
                    torch.full((input_pad,), pad_token, dtype=torch.long),
                ])
            if label_pad > 0:
                # -100 is ignored by PyTorch cross-entropy loss
                label_seq = torch.cat([
                    label_seq,
                    torch.full((label_pad,), -100, dtype=torch.long),
                ])

            # Attention mask is 1 for real tokens, 0 for input padding.
            attention_mask = torch.ones(max_length, dtype=torch.long)
            if input_pad > 0:
                attention_mask[-input_pad:] = 0

            input_ids_list.append(input_seq)
            labels_list.append(label_seq)
            attention_masks_list.append(attention_mask)

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_masks_list),
        }


class ShadowDataModule:
    """
    Data module for managing shadow datasets and data loaders.

    Handles dataset creation, splitting, and data loader setup for training.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """
        Initialize shadow data module.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.train_dataset: Optional[ShadowDataset] = None
        self.val_dataset: Optional[ShadowDataset] = None
        self.test_dataset: Optional[ShadowDataset] = None
        self.tokenizer: Optional[ShadowTokenizer] = None

    def setup(self, collector: ShadowCollector, tokenizer: ShadowTokenizer) -> None:
        """
        Setup datasets from shadow collector.

        Args:
            collector: ShadowCollector with measurements
            tokenizer: ShadowTokenizer for processing
        """
        self.tokenizer = tokenizer

        # Tokenize measurements and wrap with BOS/EOS
        token_sequences = tokenizer.tokenize_collector(collector)
        sequences = tokenizer.create_sequences(token_sequences, add_special_tokens=True)

        self._split_dataset(sequences)

    def _split_dataset(self, sequences: List[List[int]]) -> None:
        """
        Split sequences into train/val/test sets.

        Train and val sizes are determined by their configured fractions
        (rounded down via int()).  Test receives all remaining sequences,
        which may differ slightly from test_split due to integer rounding.
        """
        n_total = len(sequences)
        if n_total == 0:
            raise ValueError("Cannot split an empty sequence list.")

        # Shuffle with optional seed for reproducibility
        if self.config.shuffle:
            rng = np.random.default_rng(self.config.shuffle_seed)
            indices = np.arange(n_total)
            rng.shuffle(indices)
            sequences = [sequences[i] for i in indices]

        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.val_split)
        # Test receives the remainder; may differ slightly from test_split
        # due to int() rounding on train and val.

        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train + n_val]
        test_sequences = sequences[n_train + n_val:]

        # Warn if any split is unexpectedly empty
        for name, split in [("train", train_sequences), ("val", val_sequences), ("test", test_sequences)]:
            if len(split) == 0:
                warnings.warn(
                    f"{name} split is empty (n_total={n_total}, "
                    f"train_split={self.config.train_split}, "
                    f"val_split={self.config.val_split}).",
                    UserWarning,
                    stacklevel=2,
                )

        self.train_dataset = ShadowDataset(train_sequences, self.tokenizer, self.config)
        self.val_dataset = ShadowDataset(val_sequences, self.tokenizer, self.config)
        self.test_dataset = ShadowDataset(test_sequences, self.tokenizer, self.config)

        print(
            f"Dataset split: {len(train_sequences)} train, "
            f"{len(val_sequences)} val, {len(test_sequences)} test"
        )

    def get_train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        if self.train_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            collate_fn=self.train_dataset.collate_fn,
        )

    def get_val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        if self.val_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn,
        )

    def get_test_dataloader(self) -> DataLoader:
        """Get test data loader."""
        if self.test_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            collate_fn=self.test_dataset.collate_fn,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets."""
        return {
            "tokenizer": {
                "vocab_size": self.tokenizer.get_vocab_size() if self.tokenizer else None,
                "token_type": self.tokenizer.config.token_type if self.tokenizer else None,
                "max_length": self.tokenizer.config.max_sequence_length if self.tokenizer else None,
            },
            "datasets": {
                "train_size": len(self.train_dataset) if self.train_dataset else 0,
                "val_size": len(self.val_dataset) if self.val_dataset else 0,
                "test_size": len(self.test_dataset) if self.test_dataset else 0,
            },
            "config": self.config.__dict__,
        }

    def save_datasets(self, output_dir: str) -> None:
        """
        Save tokenizer and dataset metadata to disk.

        Note: this method saves the tokenizer vocabulary (tokenizer.json) and
        a dataset_info.json summary file.  It does NOT serialize the raw token
        sequences or PyTorch tensors.  To persist the actual data, save the
        token sequences separately before calling setup().

        Args:
            output_dir: Directory in which to write the output files
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")

        os.makedirs(output_dir, exist_ok=True)

        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        self.tokenizer.save_tokenizer(tokenizer_path)

        info_path = os.path.join(output_dir, "dataset_info.json")
        with open(info_path, "w") as f:
            json.dump(self.get_dataset_info(), f, indent=2)

        print(f"Saved tokenizer and dataset info to {output_dir}")

    def __repr__(self) -> str:
        return (
            f"ShadowDataModule(batch_size={self.config.batch_size}, "
            f"train_size={len(self.train_dataset) if self.train_dataset else 0})"
        )


def create_data_module(
    collector: ShadowCollector,
    tokenizer: ShadowTokenizer,
    config: Optional[DatasetConfig] = None,
) -> ShadowDataModule:
    """
    Convenience function to create a shadow data module.

    Args:
        collector: ShadowCollector with measurements
        tokenizer: ShadowTokenizer for processing
        config: Dataset configuration (uses defaults if None)

    Returns:
        Configured ShadowDataModule
    """
    if config is None:
        config = DatasetConfig()

    data_module = ShadowDataModule(config)
    data_module.setup(collector, tokenizer)
    return data_module
