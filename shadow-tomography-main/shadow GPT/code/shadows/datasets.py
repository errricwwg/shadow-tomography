"""
PyTorch dataset for the ShadowGPT generative pipeline.

Each training example is a complete generative sequence:

    [BOS] [MODEL_FAM, params..., SEP] [GX|GY|GZ  GO0|GO1] × n_qubits [EOS]

Pipeline
--------
    ShadowCollector → build_generative_sequence → GenerativeShadowDataset → ShadowGPT

Each item returned by __getitem__ contains:

    input_ids  : (L,) int64 — full sequence token IDs
    labels     : (L,) int64 — input_ids shifted left by 1;
                 last position gets −100 (ignored by CrossEntropyLoss)
    loss_mask  : (L,) int64 — 1 at positions whose label is GO0 or GO1,
                 0 elsewhere; restricts the training loss to outcome tokens only
"""

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenization import ShadowTokenizer


class GenerativeShadowDataset(Dataset):
    """
    Dataset for next-token prediction over classical shadow sequences.

    All sequences must have equal length L (guaranteed when sequences are built
    with build_generative_sequence() from a fixed-size tokenizer).
    """

    def __init__(
        self,
        sequences: List[List[int]],
        tokenizer: ShadowTokenizer,
        outcome_token_ids: Optional[set] = None,
    ) -> None:
        """
        Args:
            sequences:         Pre-built token ID lists (from build_generative_sequence).
            tokenizer:         ShadowTokenizer with GX/GY/GZ/GO0/GO1 in vocab.
            outcome_token_ids: Token IDs to mark in loss_mask.  Auto-detected as
                               {GO0_id, GO1_id} from tokenizer.vocab if None.
        """
        self.tokenizer = tokenizer

        if outcome_token_ids is None:
            outcome_token_ids = {
                tokenizer.vocab[name]
                for name in ("GO0", "GO1")
                if name in tokenizer.vocab
            }
        self.outcome_token_ids: set = outcome_token_ids

        self._sequences: List[torch.Tensor] = [
            torch.tensor(s, dtype=torch.long) for s in sequences
        ]
        self._loss_masks: List[torch.Tensor] = [
            self._build_loss_mask(t) for t in self._sequences
        ]

    def _build_loss_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """1 where the label (next token) is an outcome token, 0 elsewhere."""
        L = len(seq)
        mask = torch.zeros(L, dtype=torch.long)
        for j in range(L - 1):
            if seq[j + 1].item() in self.outcome_token_ids:
                mask[j] = 1
        return mask

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self._sequences[idx]
        labels = torch.cat([seq[1:], torch.tensor([-100], dtype=torch.long)])
        return {
            "input_ids":  seq,
            "labels":     labels,
            "loss_mask":  self._loss_masks[idx],
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack a list of equal-length samples into a batch."""
        return {
            "input_ids":  torch.stack([s["input_ids"]  for s in batch]),
            "labels":     torch.stack([s["labels"]     for s in batch]),
            "loss_mask":  torch.stack([s["loss_mask"]  for s in batch]),
        }

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )
