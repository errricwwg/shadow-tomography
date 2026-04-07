"""
Transformer regression model for physical quantity prediction from classical shadows.

Architecture
------------
    Token sequence  →  Embedding + Positional Encoding
                    →  TransformerEncoder (N layers)
                    →  Pooling (mean / CLS / last)
                    →  Linear regression head
                    →  Physical quantity prediction(s)

All components use PyTorch's built-in modules; no Hugging Face dependency.

Typical use
-----------
    from shadows.model import ShadowModelConfig, ShadowTransformer, ShadowTrainer
    from shadows.model import create_model_from_tokenizer

    model = create_model_from_tokenizer(tokenizer, n_outputs=1)
    trainer = ShadowTrainer(model)
    trainer.train(train_loader, val_loader, n_epochs=20)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tokenization import ShadowTokenizer


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ShadowModelConfig:
    """
    Configuration for the shadow transformer regression model.

    Key parameters
    --------------
    vocab_size    : must match tokenizer.get_vocab_size()
    max_seq_len   : must match tokenizer.config.max_sequence_length
    n_outputs     : 1 for scalar prediction; >1 for multi-target regression
    pooling       : how to aggregate the per-token encoder output
                    "mean"  — weighted mean over non-PAD positions (default)
                    "cls"   — use the first token's representation
                    "last"  — use the last non-PAD token's representation
    pad_token_id  : must match tokenizer.special_tokens["PAD"]
    """
    vocab_size:   int           # from tokenizer.get_vocab_size()
    max_seq_len:  int           # from tokenizer.config.max_sequence_length
    d_model:      int  = 128    # embedding / hidden dimension
    n_heads:      int  = 4      # attention heads (must divide d_model)
    n_layers:     int  = 4      # transformer encoder layers
    d_ff:         int  = 512    # feed-forward inner dimension
    dropout:      float = 0.1
    n_outputs:    int  = 1      # regression output dimension
    pooling:      str  = "mean" # "mean" | "cls" | "last"
    pad_token_id: int  = 2      # from tokenizer.special_tokens["PAD"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ShadowTransformer(nn.Module):
    """
    Transformer encoder with a regression head for physical quantity prediction.

    Input:  (batch, seq_len)  int64 token IDs
            (batch, seq_len)  int64 attention mask (1=real, 0=PAD)
    Output: (batch, n_outputs)  float32 predicted physical quantity
    """

    def __init__(self, config: ShadowModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.token_emb = nn.Embedding(
            config.vocab_size, config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Regression head
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_outputs)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; normal for embeddings."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L) int64
            attention_mask: (B, L) int64 — 1 for real tokens, 0 for PAD

        Returns:
            (B, n_outputs) float32
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_emb(input_ids) + self.pos_emb(positions)  # (B, L, d)

        # TransformerEncoder src_key_padding_mask: True where token is IGNORED.
        pad_mask = attention_mask == 0  # (B, L), bool

        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d)
        x = self.norm(x)

        # Pool sequence → fixed-size vector
        pooled = self._pool(x, attention_mask)  # (B, d)

        return self.head(pooled)  # (B, n_outputs)

    def _pool(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate (B, L, d) → (B, d) according to config.pooling."""
        if self.config.pooling == "cls":
            return x[:, 0]

        if self.config.pooling == "last":
            # Index of the last real (non-PAD) token per sample
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
            idx = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, x.size(-1))
            return x.gather(1, idx).squeeze(1)

        # Default: "mean" — average over non-PAD positions
        mask = attention_mask.float().unsqueeze(-1)  # (B, L, 1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ShadowTrainer:
    """
    Training loop for ShadowTransformer.

    Loss: MSELoss by default (appropriate for regression on physical quantities).

    Usage:
        trainer = ShadowTrainer(model)
        trainer.train(train_loader, val_loader, n_epochs=20)
        predictions = trainer.predict(test_loader)
    """

    def __init__(
        self,
        model: ShadowTransformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=1e-2
        )
        self.loss_fn = loss_fn or nn.MSELoss()
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target         = batch["target"].to(self.device)  # (B, n_targets)

            self.optimizer.zero_grad()
            pred = self.model(input_ids, attention_mask)      # (B, n_outputs)
            loss = self.loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def eval_epoch(self, dataloader: DataLoader) -> float:
        """Run one evaluation epoch. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target         = batch["target"].to(self.device)
            pred = self.model(input_ids, attention_mask)
            total_loss += self.loss_fn(pred, target).item()
        return total_loss / max(len(dataloader), 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        print_every: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Train for n_epochs, printing progress every print_every epochs.

        Returns:
            history dict with "train_loss" and "val_loss" lists.
        """
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.eval_epoch(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"train_loss={train_loss:.6f} | "
                    f"val_loss={val_loss:.6f}"
                )
        return self.history

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Run inference on a dataloader (targets not required).

        Returns:
            Tensor of shape (N, n_outputs) on CPU.
        """
        self.model.eval()
        preds = []
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            preds.append(self.model(input_ids, attention_mask).cpu())
        return torch.cat(preds, dim=0)

    def save(self, path: str) -> None:
        """Save model weights and config to a .pt file."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)
        print(f"Saved model to {path}")

    def load(self, path: str) -> None:
        """Load model weights from a .pt file saved by save()."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", self.history)
        print(f"Loaded model from {path}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model_from_tokenizer(
    tokenizer: ShadowTokenizer,
    n_outputs: int = 1,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    pooling: str = "mean",
) -> ShadowTransformer:
    """
    Instantiate a ShadowTransformer sized to match a given tokenizer.

    Args:
        tokenizer:  ShadowTokenizer whose vocab_size and max_seq_len are used.
        n_outputs:  Number of regression targets (1 for scalar prediction).
        d_model:    Embedding / hidden dimension.  128 trains quickly on CPU;
                    use 256 or 512 for larger datasets.
        n_heads:    Number of attention heads (must divide d_model).
        n_layers:   Number of TransformerEncoder layers.
        d_ff:       Feed-forward inner dimension (typically 4 × d_model).
        dropout:    Dropout probability.
        pooling:    "mean" | "cls" | "last".

    Returns:
        ShadowTransformer ready to train.
    """
    config = ShadowModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=tokenizer.config.max_sequence_length,
        pad_token_id=tokenizer.special_tokens["PAD"],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        n_outputs=n_outputs,
        pooling=pooling,
    )
    return ShadowTransformer(config)
