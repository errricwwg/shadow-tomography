"""
ShadowGPT: decoder-only GPT for autoregressive classical shadow generation.

Model objective
---------------
    ShadowGPT learns p_model(b_0,...,b_{n-1} | P_0,...,P_{n-1}, g) via
    teacher-forced next-token prediction on sequences of the form:

        [BOS] [MODEL_FAM, params..., SEP] [GX|GY|GZ  GO0|GO1] × n_qubits [EOS]

    Cross-entropy loss is restricted to GO0/GO1 (outcome) positions only.
    At inference, sample P uniformly and autoregressively generate b tokens
    restricted to {GO0, GO1}; feed resulting synthetic shadows to
    ShadowProcessor to estimate physical observables.

Architecture
------------
    Token IDs → Embedding + Positional Encoding
              → CausalTransformerEncoder (N pre-norm layers, causal mask)
              → LayerNorm
              → Linear(d_model → vocab_size)  [weights tied to token embedding]

Typical use
-----------
    from shadows.tokenization import create_generative_tokenizer
    from shadows.model import create_gpt_from_tokenizer

    tokenizer = create_generative_tokenizer(n_qubits=6, family="tfim")
    model = create_gpt_from_tokenizer(tokenizer)
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .tokenization import ShadowTokenizer


@dataclass
class GPTConfig:
    """
    Configuration for ShadowGPT.

    Parameters
    ----------
    vocab_size    : must match tokenizer.get_vocab_size()
    max_seq_len   : positional-embedding table size; must fit the full sequence
                    [BOS + h_prefix + 2*n_qubits + EOS]
    d_model       : embedding / hidden dimension
    n_heads       : self-attention heads (must divide d_model)
    n_layers      : number of causal transformer layers
    d_ff          : feed-forward inner dimension (typically 4 × d_model)
    dropout       : dropout probability
    pad_token_id  : PAD token ID (used as embedding padding_idx)
    """
    vocab_size:   int
    max_seq_len:  int
    d_model:      int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 4
    d_ff:         int   = 512
    dropout:      float = 0.1
    pad_token_id: int   = 2


class ShadowGPT(nn.Module):
    """
    Decoder-only GPT for autoregressive shadow measurement generation.

    Input:  (B, L) int64 token IDs
    Output: (B, L, vocab_size) float32 logits

    A causal mask is applied inside forward() so callers only pass token IDs.
    Weight tying: lm_head.weight shares memory with token_emb.weight.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(
            config.vocab_size, config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.pos_emb  = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,          # pre-norm (GPT-2 style)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        self.norm    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: lm_head and token_emb share the same matrix.
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,   # (B, L) int64
    ) -> torch.Tensor:             # (B, L, vocab_size) float32
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(positions))

        # Upper-triangular causal mask: 0 on lower triangle, -inf on upper.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=causal_mask)
        return self.lm_head(self.norm(x))   # (B, L, vocab_size)

    @torch.no_grad()
    def generate_next_token(
        self,
        input_ids: torch.Tensor,  # (1, L) int64 — current prefix
        temperature: float = 1.0,
        allowed_ids: Optional[List[int]] = None,
    ) -> int:
        """
        Sample one token from the distribution at the last position.

        Args:
            input_ids:   Current prefix, shape (1, L).
            temperature: Softmax temperature.  1.0 = standard; < 1.0 = sharper.
            allowed_ids: Restrict sampling to these vocabulary IDs (e.g. {GO0, GO1}).

        Returns:
            Sampled token ID as a Python int.
        """
        logits = self.forward(input_ids)[0, -1, :]   # (V,)
        if allowed_ids is not None:
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed_ids] = 0.0
            logits = logits + mask
        logits = logits / max(temperature, 1e-8)
        probs  = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gpt_from_tokenizer(
    tokenizer: ShadowTokenizer,
    max_seq_len: Optional[int] = None,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    dropout: float = 0.1,
) -> ShadowGPT:
    """
    Instantiate a ShadowGPT sized to match a given tokenizer.

    Args:
        tokenizer:   ShadowTokenizer (with generative tokens already added).
        max_seq_len: Override positional-encoding table size.  None = tokenizer default.
        d_model:     Embedding / hidden dimension.
        n_heads:     Self-attention heads (must divide d_model).
        n_layers:    Number of causal transformer layers.
        d_ff:        Feed-forward inner dimension.
        dropout:     Dropout probability.

    Returns:
        ShadowGPT ready to train.
    """
    config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=(max_seq_len if max_seq_len is not None
                     else tokenizer.config.max_sequence_length),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_token_id=tokenizer.special_tokens["PAD"],
    )
    return ShadowGPT(config)
