"""
PyTorch dataset for classical shadow data — physical quantity prediction.

This module provides ShadowDataset and ShadowDataModule for training a
transformer to predict physical quantities (regression) from classical
shadow token sequences.

Pipeline position
-----------------
    collector  →  tokenizer  →  ShadowDataModule  →  ShadowTransformer
                                     ↑
                              targets (physical quantities,
                              provided externally per measurement)

Target convention
-----------------
One target value (scalar or vector) per shadow measurement.  Targets must
be provided as a 1-D or 2-D array of length n_measurements alongside the
collector when calling setup().  Typical sources:

- ShadowProcessor.estimate_magnetization / estimate_energy (classical shadow
  estimates, one per state — repeat the same value for all measurements from
  that state).
- Exact diagonalization or DMRG (ground-truth labels).
- Any per-state physical quantity known externally.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .collector import ShadowCollector
from .tokenization import ShadowTokenizer, TokenizationConfig


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShadowDataset(Dataset):
    """
    PyTorch dataset for physical quantity prediction from shadow sequences.

    Each item contains:
        input_ids      — token sequence (already padded by the tokenizer)
        attention_mask — 1 for real tokens, 0 for PAD positions
        target         — physical quantity value(s), shape (n_targets,)
                         (only present when targets were provided at init)
    """

    def __init__(
        self,
        token_sequences: List[List[int]],
        tokenizer: ShadowTokenizer,
        targets: Optional[Union[np.ndarray, List]] = None,
        config: Optional[DatasetConfig] = None,
    ) -> None:
        """
        Args:
            token_sequences: List of integer token lists (one per measurement).
            tokenizer:       ShadowTokenizer used to produce token_sequences.
            targets:         Physical quantity labels, shape (n_samples,) or
                             (n_samples, n_targets).  None for inference-only use.
            config:          Dataset configuration.
        """
        if targets is not None and len(targets) != len(token_sequences):
            raise ValueError(
                f"len(targets)={len(targets)} must equal "
                f"len(token_sequences)={len(token_sequences)}."
            )

        self.token_sequences = token_sequences
        self.tokenizer = tokenizer
        self.config = config or DatasetConfig()
        self.sequences = self._prepare_sequences()

        # Store targets as float32 tensor, always 2-D: (N, n_targets)
        if targets is not None:
            t = np.asarray(targets, dtype=np.float32)
            if t.ndim == 1:
                t = t[:, None]
            self.targets: Optional[torch.Tensor] = torch.tensor(t, dtype=torch.float32)
        else:
            self.targets = None

    def _prepare_sequences(self) -> List[torch.Tensor]:
        """Convert token lists to LongTensors."""
        return [
            torch.tensor(tokens, dtype=torch.long)
            for tokens in self.token_sequences
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return one sample for regression.

        Returns:
            input_ids      shape (seq_len,)
            attention_mask shape (seq_len,)  — 1 = real, 0 = PAD
            target         shape (n_targets,)  — only if targets were given
        """
        sequence = self.sequences[idx]
        pad_id = self.tokenizer.special_tokens["PAD"]
        attention_mask = (sequence != pad_id).long()

        item: Dict[str, torch.Tensor] = {
            "input_ids": sequence,
            "attention_mask": attention_mask,
        }
        if self.targets is not None:
            item["target"] = self.targets[idx]
        return item

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch.

        Pads input_ids / attention_mask to the longest sequence in the batch
        (no-op when the tokenizer already pads everything to max_seq_len).
        Targets are simply stacked.
        """
        max_len = max(len(s["input_ids"]) for s in batch)
        pad_id = self.tokenizer.special_tokens["PAD"]

        input_ids_list: List[torch.Tensor] = []
        attn_mask_list: List[torch.Tensor] = []

        for s in batch:
            seq = s["input_ids"]
            mask = s["attention_mask"]
            pad_needed = max_len - len(seq)
            if pad_needed > 0:
                seq = torch.cat([seq, torch.full((pad_needed,), pad_id, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_needed, dtype=torch.long)])
            input_ids_list.append(seq)
            attn_mask_list.append(mask)

        result: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids_list),       # (B, L)
            "attention_mask": torch.stack(attn_mask_list),  # (B, L)
        }
        if "target" in batch[0]:
            result["target"] = torch.stack([s["target"] for s in batch])  # (B, n_targets)
        return result


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class ShadowDataModule:
    """
    Data module for managing shadow datasets and data loaders.

    Handles tokenization, target alignment, dataset splitting, and
    DataLoader setup for training a regression model.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.train_dataset: Optional[ShadowDataset] = None
        self.val_dataset:   Optional[ShadowDataset] = None
        self.test_dataset:  Optional[ShadowDataset] = None
        self.tokenizer:     Optional[ShadowTokenizer] = None

    def setup(
        self,
        collector: ShadowCollector,
        tokenizer: ShadowTokenizer,
        targets: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        """
        Tokenize measurements and split into train / val / test datasets.

        Args:
            collector: ShadowCollector with measurements already collected.
            tokenizer: ShadowTokenizer matching the measurement mode.
            targets:   Physical quantity labels, shape (n_measurements,) or
                       (n_measurements, n_targets).  Must be provided for
                       training; may be None for inference-only use.
        """
        self.tokenizer = tokenizer

        token_sequences = tokenizer.tokenize_collector(collector)
        sequences = tokenizer.create_sequences(token_sequences, add_special_tokens=True)

        targets_arr = np.asarray(targets) if targets is not None else None
        self._split_dataset(sequences, targets_arr)

    def _split_dataset(
        self,
        sequences: List[List[int]],
        targets: Optional[np.ndarray] = None,
    ) -> None:
        """
        Split sequences (and optional targets) into train / val / test sets.

        Train and val sizes are determined by their configured fractions
        (rounded down via int()).  Test receives all remaining sequences.
        """
        n_total = len(sequences)
        if n_total == 0:
            raise ValueError("Cannot split an empty sequence list.")

        if self.config.shuffle:
            rng = np.random.default_rng(self.config.shuffle_seed)
            indices = np.arange(n_total)
            rng.shuffle(indices)
            sequences = [sequences[i] for i in indices]
            if targets is not None:
                targets = targets[indices]

        n_train = int(n_total * self.config.train_split)
        n_val   = int(n_total * self.config.val_split)

        splits = {
            "train": (sequences[:n_train],              targets[:n_train]              if targets is not None else None),
            "val":   (sequences[n_train:n_train+n_val], targets[n_train:n_train+n_val] if targets is not None else None),
            "test":  (sequences[n_train+n_val:],        targets[n_train+n_val:]        if targets is not None else None),
        }

        for name, (seqs, _) in splits.items():
            if len(seqs) == 0:
                warnings.warn(
                    f"{name} split is empty (n_total={n_total}, "
                    f"train_split={self.config.train_split}, "
                    f"val_split={self.config.val_split}).",
                    UserWarning,
                    stacklevel=2,
                )

        self.train_dataset = ShadowDataset(splits["train"][0], self.tokenizer, splits["train"][1], self.config)
        self.val_dataset   = ShadowDataset(splits["val"][0],   self.tokenizer, splits["val"][1],   self.config)
        self.test_dataset  = ShadowDataset(splits["test"][0],  self.tokenizer, splits["test"][1],  self.config)

        print(
            f"Dataset split: {len(splits['train'][0])} train, "
            f"{len(splits['val'][0])} val, {len(splits['test'][0])} test"
        )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def get_train_dataloader(self) -> DataLoader:
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return a summary dict describing the current datasets."""
        has_targets = (
            self.train_dataset is not None and self.train_dataset.targets is not None
        )
        n_targets = (
            self.train_dataset.targets.shape[1]
            if has_targets else 0
        )
        return {
            "tokenizer": {
                "vocab_size": self.tokenizer.get_vocab_size() if self.tokenizer else None,
                "token_type": self.tokenizer.config.token_type if self.tokenizer else None,
                "max_length": self.tokenizer.config.max_sequence_length if self.tokenizer else None,
            },
            "datasets": {
                "train_size": len(self.train_dataset) if self.train_dataset else 0,
                "val_size":   len(self.val_dataset)   if self.val_dataset   else 0,
                "test_size":  len(self.test_dataset)  if self.test_dataset  else 0,
            },
            "targets": {
                "has_targets": has_targets,
                "n_targets": n_targets,
            },
            "config": self.config.__dict__,
        }

    def save_datasets(self, output_dir: str) -> None:
        """
        Save tokenizer and dataset metadata to disk.

        Saves:
          tokenizer.json    — vocabulary and tokenizer config
          dataset_info.json — split sizes and config summary

        Note: raw token sequences and regression targets are NOT serialized
        here.  To persist them, save targets separately before calling setup().
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")

        os.makedirs(output_dir, exist_ok=True)

        self.tokenizer.save_tokenizer(os.path.join(output_dir, "tokenizer.json"))

        with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
            json.dump(self.get_dataset_info(), f, indent=2)

        print(f"Saved tokenizer and dataset info to {output_dir}")

    def __repr__(self) -> str:
        return (
            f"ShadowDataModule(batch_size={self.config.batch_size}, "
            f"train_size={len(self.train_dataset) if self.train_dataset else 0})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_data_module(
    collector: ShadowCollector,
    tokenizer: ShadowTokenizer,
    targets: Optional[Union[np.ndarray, List]] = None,
    config: Optional[DatasetConfig] = None,
) -> ShadowDataModule:
    """
    Convenience function to create and setup a ShadowDataModule.

    Args:
        collector: ShadowCollector with measurements.
        tokenizer: ShadowTokenizer for encoding.
        targets:   Physical quantity labels, shape (n_measurements,) or
                   (n_measurements, n_targets).
        config:    Dataset configuration (uses defaults if None).

    Returns:
        Configured and ready ShadowDataModule.
    """
    dm = ShadowDataModule(config or DatasetConfig())
    dm.setup(collector, tokenizer, targets)
    return dm
