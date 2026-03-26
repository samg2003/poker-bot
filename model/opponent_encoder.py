"""
Opponent Encoder — Transformer that processes action history
into a fixed-size embedding per opponent.

Input: variable-length sequence of action tokens per opponent
Output: fixed-size player embedding (embed_dim)

The opponent encoder is the HEART of the system. It learns to detect:
- Playing style (tight/loose, passive/aggressive)
- Situational tendencies (per-street, per-position)
- Tilt signals (behavioral shifts)
- Sizing patterns (tells)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from model.action_space import ACTION_FEATURE_DIM


class OpponentEncoder(nn.Module):
    """
    Causal Transformer encoder for opponent action histories.

    Processes a sequence of action tokens and outputs a fixed-size embedding
    representing everything the model knows about this opponent.

    Args:
        action_dim: dimension of each action token (default: ACTION_FEATURE_DIM = 14)
        embed_dim: dimension of the output embedding (default: 128)
        num_heads: number of attention heads (default: 4)
        num_layers: number of transformer layers (default: 3)
        max_seq_len: max actions to process per opponent (default: 512)
        dropout: dropout rate (default: 0.1)
    """

    def __init__(
        self,
        action_dim: int = ACTION_FEATURE_DIM,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Project raw action features to embed_dim
        self.input_proj = nn.Linear(action_dim, embed_dim)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer encoder (causal — can only attend to past actions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # Fixes MPS crash with padding masks
        )

        # Output projection: aggregate sequence → fixed embedding
        self.output_norm = nn.LayerNorm(embed_dim)

        # Learnable "query" token for aggregation
        # (appended to sequence, attends to all actions, output = embedding)
        self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(
        self,
        action_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode an opponent's action history.

        Args:
            action_seq: (batch, seq_len, action_dim) — raw action features
            mask: (batch, seq_len) — True for padded positions

        Returns:
            embedding: (batch, embed_dim) — fixed-size opponent embedding
        """
        batch_size, seq_len, _ = action_seq.shape

        # Project input features
        x = self.input_proj(action_seq)  # (batch, seq, embed_dim)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Append aggregation token at the end
        agg = self.agg_token.expand(batch_size, -1, -1)
        x = torch.cat([x, agg], dim=1)  # (batch, seq+1, embed_dim)

        # Build causal mask (each position can only attend to itself and past)
        total_len = seq_len + 1
        causal_mask = torch.triu(
            torch.ones(total_len, total_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )  # True = masked

        # Extend padding mask to include aggregation token (never masked)
        if mask is not None:
            # mask: (batch, seq_len) → (batch, seq+1)
            agg_mask = torch.zeros(batch_size, 1, device=mask.device, dtype=mask.dtype)
            key_padding_mask = torch.cat([mask, agg_mask], dim=1)
        else:
            key_padding_mask = None

        # Run transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        # Extract aggregation token output (last position)
        embedding = x[:, -1, :]  # (batch, embed_dim)
        embedding = self.output_norm(embedding)

        return embedding

    def encode_empty(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Generate a zero-info embedding (new opponent, no history).
        Used when we have no reads — triggers GTO-ish play.
        """
        # Just run the aggregation token through with empty sequence
        return torch.zeros(batch_size, self.embed_dim, device=device)
