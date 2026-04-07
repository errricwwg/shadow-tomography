"""
PyTorch dataset for classical shadow data.

This module provides the ShadowDataset class for loading and processing
classical shadow data for training language models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import os

from .config import ShadowConfig
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


class ShadowDataset(Dataset):
    """
    PyTorch dataset for classical shadow data.
    
    Loads shadow measurements and converts them to token sequences
    suitable for training language models.
    """
    
    def __init__(self, token_sequences: List[List[int]], 
                 tokenizer: ShadowTokenizer,
                 config: Optional[DatasetConfig] = None):
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
            # Convert to tensor
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
            Dictionary with input_ids and labels for language modeling
        """
        sequence = self.sequences[idx]
        
        # For language modeling, input and target are the same (shifted by 1)
        input_ids = sequence[:-1] if len(sequence) > 1 else sequence
        labels = sequence[1:] if len(sequence) > 1 else sequence
        
        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched tensors
        """
        # Pad sequences to the same length
        max_length = max(len(sample["input_ids"]) for sample in batch)
        
        input_ids = []
        labels = []
        attention_masks = []
        
        for sample in batch:
            # Pad input_ids
            input_seq = sample["input_ids"]
            pad_length = max_length - len(input_seq)
            if pad_length > 0:
                pad_token = self.tokenizer.special_tokens["PAD"]
                input_seq = torch.cat([input_seq, torch.full((pad_length,), pad_token, dtype=torch.long)])
            
            # Pad labels
            label_seq = sample["labels"]
            if len(label_seq) < max_length:
                pad_length = max_length - len(label_seq)
                label_seq = torch.cat([label_seq, torch.full((pad_length,), -100, dtype=torch.long)])
            
            # Create attention mask
            attention_mask = torch.ones(len(input_seq), dtype=torch.long)
            if pad_length > 0:
                attention_mask[-pad_length:] = 0
            
            input_ids.append(input_seq)
            labels.append(label_seq)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks)
        }


class ShadowDataModule:
    """
    Data module for managing shadow datasets and data loaders.
    
    Handles dataset creation, splitting, and data loader setup for training.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize shadow data module.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tokenizer = None
    
    def setup(self, collector: ShadowCollector, 
              tokenizer: ShadowTokenizer) -> None:
        """
        Setup datasets from shadow collector.
        
        Args:
            collector: ShadowCollector with measurements
            tokenizer: ShadowTokenizer for processing
        """
        self.tokenizer = tokenizer
        
        # Tokenize measurements
        token_sequences = tokenizer.tokenize_collector(collector)
        
        # Create training sequences
        sequences = tokenizer.create_sequences(token_sequences, add_special_tokens=True)
        
        # Split dataset
        self._split_dataset(sequences)
    
    def _split_dataset(self, sequences: List[List[int]]) -> None:
        """Split dataset into train/val/test sets."""
        n_total = len(sequences)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.val_split)
        
        # Shuffle sequences
        if self.config.shuffle:
            np.random.shuffle(sequences)
        
        # Split
        train_sequences = sequences[:n_train]
        val_sequences = sequences[n_train:n_train + n_val]
        test_sequences = sequences[n_train + n_val:]
        
        # Create datasets
        self.train_dataset = ShadowDataset(train_sequences, self.tokenizer, self.config)
        self.val_dataset = ShadowDataset(val_sequences, self.tokenizer, self.config)
        self.test_dataset = ShadowDataset(test_sequences, self.tokenizer, self.config)
        
        print(f"Dataset split: {len(train_sequences)} train, {len(val_sequences)} val, {len(test_sequences)} test")
    
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
            collate_fn=self.train_dataset.collate_fn
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
            collate_fn=self.val_dataset.collate_fn
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
            collate_fn=self.test_dataset.collate_fn
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets."""
        info = {
            "tokenizer": {
                "vocab_size": self.tokenizer.get_vocab_size() if self.tokenizer else None,
                "token_type": self.tokenizer.config.token_type if self.tokenizer else None,
                "max_length": self.tokenizer.config.max_sequence_length if self.tokenizer else None
            },
            "datasets": {
                "train_size": len(self.train_dataset) if self.train_dataset else 0,
                "val_size": len(self.val_dataset) if self.val_dataset else 0,
                "test_size": len(self.test_dataset) if self.test_dataset else 0
            },
            "config": self.config.__dict__
        }
        
        return info
    
    def save_datasets(self, output_dir: str) -> None:
        """
        Save datasets to disk.
        
        Args:
            output_dir: Directory to save datasets
        """
        if self.train_dataset is None:
            raise ValueError("Dataset not setup. Call setup() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer.json")
        self.tokenizer.save_tokenizer(tokenizer_path)
        
        # Save dataset info
        info_path = os.path.join(output_dir, "dataset_info.json")
        import json
        with open(info_path, 'w') as f:
            json.dump(self.get_dataset_info(), f, indent=2)
        
        print(f"Saved datasets to {output_dir}")
    
    def __repr__(self) -> str:
        return (f"ShadowDataModule(batch_size={self.config.batch_size}, "
                f"train_size={len(self.train_dataset) if self.train_dataset else 0})")


def create_data_module(collector: ShadowCollector,
                      tokenizer: ShadowTokenizer,
                      config: Optional[DatasetConfig] = None) -> ShadowDataModule:
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
