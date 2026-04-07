"""
Transformer regression model for physical quantity prediction from classical shadows.

Architecture
------------
    Token sequence  →  Embedding + Positional Encoding
                    →  TransformerEncoder (N pre-norm layers)
                    →  Masked-mean pooling (respects attention_mask)
                    →  LayerNorm → Linear regression head
                    →  4D physical quantity prediction

The four output targets are ordered as:
    [magnetization, correlations, energy, renyi_entropy]

This matches the outputs of ShadowProcessor in processor.py.

All components use PyTorch built-ins; no Hugging Face dependency.

Typical use
-----------
    from shadows.tokenization import create_default_tokenizer
    from shadows.model import create_model_from_tokenizer, ShadowTrainer

    tokenizer = create_default_tokenizer(n_qubits=8)
    model = create_model_from_tokenizer(tokenizer, n_outputs=4)
    trainer = ShadowTrainer(model)
    trainer.fit(train_loader, val_loader, n_epochs=30)

Target ordering convention
--------------------------
When building targets from ShadowProcessor estimates, use:

    targets = np.column_stack([
        magnetization_values,   # column 0
        correlation_values,     # column 1
        energy_values,          # column 2
        renyi_entropy_values,   # column 3
    ])

TARGET_NAMES is exported for convenience.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .tokenization import ShadowTokenizer

# Physical quantity names in the order expected by this model.
TARGET_NAMES: List[str] = ["magnetization", "correlations", "energy", "renyi_entropy"]

# Default number of physical targets.
_N_TARGETS: int = 4


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
    n_outputs     : 4 for the four physical quantities; 1 for a single target
    pad_token_id  : must match tokenizer.special_tokens["PAD"]
    pooling       : sequence aggregation strategy
                    "mean" — masked mean over non-PAD positions (default)
                    "cls"  — use the first token's representation
    use_scheduler : if True, ShadowTrainer applies cosine LR annealing
    """
    vocab_size:    int              # from tokenizer.get_vocab_size()
    max_seq_len:   int              # from tokenizer.config.max_sequence_length
    d_model:       int   = 128      # embedding / hidden dimension
    n_heads:       int   = 4        # attention heads (must divide d_model)
    n_layers:      int   = 4        # TransformerEncoder layers
    d_ff:          int   = 512      # feed-forward inner dimension
    dropout:       float = 0.1
    n_outputs:     int   = _N_TARGETS  # 4 physical quantities
    pooling:       str   = "mean"   # "mean" | "cls"
    pad_token_id:  int   = 2        # from tokenizer.special_tokens["PAD"]
    use_scheduler: bool  = True     # cosine LR annealing in ShadowTrainer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ShadowTransformer(nn.Module):
    """
    Transformer encoder with a multi-target regression head.

    Input:  (B, L) int64 token IDs
            (B, L) int64 attention mask  (1 = real token, 0 = PAD)
    Output: (B, n_outputs) float32 predicted physical quantities
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
        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer encoder (pre-norm, batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,          # pre-norm for more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Regression head: LayerNorm → Linear(d_model → n_outputs)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_outputs)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, L) int64
        attention_mask: torch.Tensor, # (B, L) int64  1=real, 0=PAD
    ) -> torch.Tensor:                # (B, n_outputs) float32
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # TransformerEncoder expects src_key_padding_mask=True where token is IGNORED
        pad_mask = (attention_mask == 0)   # (B, L) bool

        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (B, L, d)
        x = self.norm(x)

        pooled = self._masked_mean(x, attention_mask)        # (B, d)
        return self.head(pooled)                             # (B, n_outputs)

    def _masked_mean(
        self,
        x: torch.Tensor,              # (B, L, d)
        attention_mask: torch.Tensor, # (B, L) int64
    ) -> torch.Tensor:                # (B, d)
        """
        Masked mean pooling: average the encoder outputs over non-PAD positions.

        If config.pooling == "cls", returns the first-token representation instead.
        Falls back to cls when all positions are masked (should not happen in practice).
        """
        if self.config.pooling == "cls":
            return x[:, 0]

        # Default: masked mean
        mask = attention_mask.float().unsqueeze(-1)   # (B, L, 1)
        sum_x = (x * mask).sum(dim=1)                # (B, d)
        count = mask.sum(dim=1).clamp(min=1.0)        # (B, 1)
        return sum_x / count                          # (B, d)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Target scaler (optional, reversible)
# ---------------------------------------------------------------------------

class TargetScaler:
    """
    Per-target standard scaler: transforms targets to zero mean, unit variance.

    Each of the n_outputs targets is scaled independently so that targets with
    very different numeric ranges (e.g. magnetization ∈ [-1,1] vs energy ∈ [-J·n, J·n])
    contribute roughly equally to the MSE loss.

    Usage:
        scaler = TargetScaler(n_outputs=4)
        scaler.fit(targets_train)          # targets_train: np.ndarray (N, 4)
        targets_scaled = scaler.transform(targets_train)
        ...
        predictions_unscaled = scaler.inverse_transform(predictions_scaled)

    The scaler is serialised into the checkpoint by ShadowTrainer.save() so it
    can be restored with ShadowTrainer.load().
    """

    def __init__(self, n_outputs: int = _N_TARGETS) -> None:
        self.n_outputs = n_outputs
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, targets: np.ndarray) -> "TargetScaler":
        """
        Compute mean and std from targets.

        Args:
            targets: Array of shape (N, n_outputs) or (N,) for single output.
        """
        t = np.asarray(targets, dtype=np.float64)
        if t.ndim == 1:
            t = t[:, None]
        self.mean_ = t.mean(axis=0)
        self.std_  = t.std(axis=0)
        # Replace zero std with 1 to avoid division by zero for constant targets.
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        self._fitted = True
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Scale targets: (x − mean) / std."""
        self._check_fitted()
        t = np.asarray(targets, dtype=np.float64)
        return ((t - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Invert scaling: x * std + mean."""
        self._check_fitted()
        s = np.asarray(scaled, dtype=np.float64)
        return (s * self.std_ + self.mean_).astype(np.float32)

    def inverse_transform_tensor(self, scaled: torch.Tensor) -> torch.Tensor:
        """Invert scaling for a torch.Tensor (returns float32 CPU tensor)."""
        self._check_fitted()
        mean_t = torch.tensor(self.mean_, dtype=torch.float32)
        std_t  = torch.tensor(self.std_,  dtype=torch.float32)
        return scaled.cpu().float() * std_t + mean_t

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TargetScaler has not been fitted. Call fit() first."
            )

    def state_dict(self) -> Dict:
        """Return serialisable state for checkpointing."""
        return {
            "n_outputs": self.n_outputs,
            "mean_":    self.mean_.tolist() if self.mean_ is not None else None,
            "std_":     self.std_.tolist()  if self.std_  is not None else None,
            "_fitted":  self._fitted,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Restore state from a dict produced by state_dict()."""
        self.n_outputs = state["n_outputs"]
        self.mean_ = np.array(state["mean_"]) if state["mean_"] is not None else None
        self.std_  = np.array(state["std_"])  if state["std_"]  is not None else None
        self._fitted = state["_fitted"]

    def __repr__(self) -> str:
        if self._fitted:
            return (f"TargetScaler(n_outputs={self.n_outputs}, "
                    f"mean={np.round(self.mean_, 4)}, "
                    f"std={np.round(self.std_, 4)})")
        return f"TargetScaler(n_outputs={self.n_outputs}, not fitted)"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ShadowTrainer:
    """
    Training loop for ShadowTransformer (multi-target regression).

    Loss: MSELoss over all n_outputs simultaneously.

    If a TargetScaler is supplied (or scaler=True is passed to fit()), targets
    are scaled before the loss and predictions are un-scaled for logging and
    storage.  This keeps the MSE loss well-conditioned when targets have
    different numeric magnitudes.

    Usage:
        trainer = ShadowTrainer(model)
        # Without scaling:
        history = trainer.fit(train_loader, val_loader, n_epochs=30)
        # With scaling fitted from the train set:
        scaler = TargetScaler(n_outputs=4).fit(train_targets_np)
        trainer = ShadowTrainer(model, scaler=scaler)
        history = trainer.fit(train_loader, val_loader, n_epochs=30)
        # Save / load:
        trainer.save("best.pt")
        trainer.load("best.pt")
        # Predict (always returns unscaled values):
        preds = trainer.predict(test_loader)   # (N, 4) numpy array
    """

    def __init__(
        self,
        model: ShadowTransformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
        scaler: Optional[TargetScaler] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=3e-4, weight_decay=1e-2
        )
        self.loss_fn = nn.MSELoss()
        self.scaler  = scaler              # optional TargetScaler
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._best_val_loss: float = float("inf")
        self._best_state: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scale_target(self, target: torch.Tensor) -> torch.Tensor:
        """Apply scaler to a target tensor if a scaler is set."""
        if self.scaler is None or not self.scaler._fitted:
            return target
        mean = torch.tensor(self.scaler.mean_, dtype=torch.float32,
                            device=target.device)
        std  = torch.tensor(self.scaler.std_,  dtype=torch.float32,
                            device=target.device)
        return (target - mean) / std

    def _get_batch(self, batch: Dict[str, torch.Tensor]):
        """Move a batch dict to device and return (input_ids, mask, target)."""
        input_ids = batch["input_ids"].to(self.device)
        mask      = batch["attention_mask"].to(self.device)
        target    = batch["target"].to(self.device).float()
        return input_ids, mask, target

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns mean MSE loss (scaled if scaler is set)."""
        self.model.train()
        total = 0.0
        for batch in dataloader:
            ids, mask, target = self._get_batch(batch)
            target_in = self._scale_target(target)

            self.optimizer.zero_grad()
            pred = self.model(ids, mask)
            loss = self.loss_fn(pred, target_in)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total += loss.item()
        return total / max(len(dataloader), 1)

    @torch.no_grad()
    def eval_epoch(self, dataloader: DataLoader) -> float:
        """Run one evaluation epoch. Returns mean MSE loss (scaled if scaler is set)."""
        self.model.eval()
        total = 0.0
        for batch in dataloader:
            ids, mask, target = self._get_batch(batch)
            target_in = self._scale_target(target)
            pred = self.model(ids, mask)
            total += self.loss_fn(pred, target_in).item()
        return total / max(len(dataloader), 1)

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        print_every: int = 1,
        save_best_to: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train for n_epochs, tracking and optionally saving the best model.

        Args:
            train_loader:  Training DataLoader (batches must contain "target").
            val_loader:    Validation DataLoader.
            n_epochs:      Number of training epochs.
            print_every:   Print progress every this many epochs.
            save_best_to:  If a file path is given, save the best checkpoint
                           (lowest val_loss) to that path automatically.

        Returns:
            history dict with "train_loss" and "val_loss" lists.
        """
        if self.model.config.use_scheduler and self.scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=1e-6
            )

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.eval_epoch(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            # Track best checkpoint
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                if save_best_to is not None:
                    self.save(save_best_to)

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch:4d}/{n_epochs}  "
                    f"train={train_loss:.6f}  val={val_loss:.6f}"
                    + (f"  *best" if val_loss == self._best_val_loss else "")
                )

        return self.history

    def restore_best(self) -> None:
        """
        Reload the best-seen weights (by val_loss) into the model.

        Call this after fit() to ensure you are evaluating/predicting with the
        best checkpoint rather than the final epoch's weights.
        """
        if self._best_state is None:
            raise RuntimeError("No best state saved yet. Call fit() first.")
        self.model.load_state_dict(self._best_state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """
        Run inference on a dataloader.

        Returns predictions in the **original (unscaled)** target space.
        If no scaler is set, returns raw model outputs.

        Returns:
            np.ndarray of shape (N, n_outputs), float32.
        """
        self.model.eval()
        parts = []
        for batch in dataloader:
            ids  = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            pred = self.model(ids, mask).cpu()
            parts.append(pred)
        raw = torch.cat(parts, dim=0)   # (N, n_outputs)

        if self.scaler is not None and self.scaler._fitted:
            return self.scaler.inverse_transform_tensor(raw).numpy()
        return raw.float().numpy()

    @torch.no_grad()
    def predict_per_target(
        self, dataloader: DataLoader
    ) -> Dict[str, np.ndarray]:
        """
        Like predict(), but returns a dict keyed by TARGET_NAMES.

        Assumes n_outputs == 4 and the standard target ordering:
            [magnetization, correlations, energy, renyi_entropy]
        """
        preds = self.predict(dataloader)   # (N, 4)
        n = self.model.config.n_outputs
        names = TARGET_NAMES[:n] if n <= len(TARGET_NAMES) else [
            f"target_{i}" for i in range(n)
        ]
        return {name: preds[:, i] for i, name in enumerate(names)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights, config, optimizer state, scaler, and history.

        The checkpoint is a plain dict saved with torch.save().  Load it
        with ShadowTrainer.load().
        """
        checkpoint = {
            "model_state_dict":     self.model.state_dict(),
            "model_config":         self.model.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history":              self.history,
            "best_val_loss":        self._best_val_loss,
            "scaler_state":         self.scaler.state_dict() if self.scaler else None,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load a checkpoint saved by save().

        Restores model weights, optimizer, scaler, and history.
        The scheduler is re-created from scratch by the next fit() call.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history      = ckpt.get("history", self.history)
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))

        scaler_state = ckpt.get("scaler_state")
        if scaler_state is not None:
            if self.scaler is None:
                self.scaler = TargetScaler(n_outputs=scaler_state["n_outputs"])
            self.scaler.load_state_dict(scaler_state)

        sched_state = ckpt.get("scheduler_state_dict")
        if sched_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(sched_state)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model_from_tokenizer(
    tokenizer: ShadowTokenizer,
    n_outputs: int = _N_TARGETS,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
    pooling: str = "mean",
    use_scheduler: bool = True,
) -> ShadowTransformer:
    """
    Instantiate a ShadowTransformer sized to match a given tokenizer.

    Args:
        tokenizer:     ShadowTokenizer whose vocab_size and max_seq_len are used.
        n_outputs:     Number of regression targets.  4 for the full set of
                       [magnetization, correlations, energy, renyi_entropy].
        d_model:       Embedding / hidden dimension.  128 trains quickly on CPU;
                       use 256 or 512 for larger datasets.
        n_heads:       Attention heads (must divide d_model).
        n_layers:      Number of TransformerEncoder layers.
        d_ff:          Feed-forward inner dimension (typically 4 × d_model).
        dropout:       Dropout probability.
        pooling:       "mean" (masked mean, default) or "cls" (first token).
        use_scheduler: Pass to ShadowModelConfig; enables cosine LR in trainer.

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
        use_scheduler=use_scheduler,
    )
    return ShadowTransformer(config)
