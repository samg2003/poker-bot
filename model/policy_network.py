"""
Policy Network — Transformer-based policy + value network.

Takes game state + opponent embeddings + HUD stats and outputs:
  1. Action type distribution (fold/check/call/raise)
  2. Bet sizing (continuous, for raises)
  3. Value estimate (expected bb/hand)

The key architectural feature is CROSS-ATTENTION between the game state
and opponent embeddings, allowing the network to modulate its strategy
based on who it's playing against.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.action_space import NUM_ACTION_TYPES, ActionOutput
from model.stat_tracker import NUM_STAT_FEATURES


class CardEmbedding(nn.Module):
    """
    Embed poker cards (0-51) into a learned representation.

    Separately embeds rank and suit, then combines them.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.rank_embed = nn.Embedding(13, embed_dim)  # 2-A
        self.suit_embed = nn.Embedding(4, embed_dim)   # c/d/h/s
        self.combine = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, cards: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cards: (batch, num_cards) — card indices 0-51, or -1 for absent

        Returns:
            (batch, num_cards, embed_dim)
        """
        # Mask absent cards (index -1)
        valid = cards >= 0
        safe_cards = cards.clamp(min=0)

        ranks = safe_cards // 4
        suits = safe_cards % 4

        r = self.rank_embed(ranks)
        s = self.suit_embed(suits)
        combined = self.combine(torch.cat([r, s], dim=-1))

        # Zero out absent cards
        combined = combined * valid.unsqueeze(-1).float()

        return combined


class CardTransformerBlock(nn.Module):
    """Full transformer block with self-attention + feedforward."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class GameStateEncoder(nn.Module):
    """
    Encode the current game state into a fixed-size representation.

    Cards processed by a 4-layer transformer at embed_dim with learned
    attention-pooling. Full architecture for CUDA training.
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Card embedding at full embed_dim
        self.card_embed = CardEmbedding(embed_dim=embed_dim)
        self.owner_embed = nn.Embedding(2, embed_dim)  # 0 = Hole, 1 = Community

        # 4-layer deep card transformer
        self.card_transformer = nn.ModuleList([
            CardTransformerBlock(embed_dim, num_heads=4)
            for _ in range(4)
        ])

        # Learned attention-pooling
        self.card_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pool_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Numeric features: 23-dim vector per final_state.md §3
        self.numeric_proj = nn.Sequential(
            nn.Linear(23, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Combine cards + numeric
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        hole_cards: torch.Tensor,       # (batch, 2) card indices
        community_cards: torch.Tensor,   # (batch, 5) card indices (-1 if absent)
        numeric_features: torch.Tensor,  # (batch, 23)
    ) -> torch.Tensor:
        batch_size = hole_cards.shape[0]

        # Embed cards at embed_dim and apply ownership tags
        hole_embs = self.card_embed(hole_cards) + self.owner_embed(
            torch.zeros(batch_size, 2, dtype=torch.long, device=hole_cards.device)
        )
        comm_embs = self.card_embed(community_cards) + self.owner_embed(
            torch.ones(batch_size, 5, dtype=torch.long, device=community_cards.device)
        )

        # Zero out absent community cards
        valid_comm = (community_cards >= 0).unsqueeze(-1).float()
        comm_embs = comm_embs * valid_comm

        card_embs = torch.cat([hole_embs, comm_embs], dim=1)  # (batch, 7, embed_dim)

        # 4-layer transformer over all 7 cards
        card_out = card_embs
        for block in self.card_transformer:
            card_out = block(card_out)

        # Learned attention-pooling: query token attends to all card outputs
        query = self.card_query.expand(batch_size, -1, -1)
        card_repr, _ = self.pool_attn(query, card_out, card_out)
        card_repr = card_repr.squeeze(1)  # (B, embed_dim)

        # Project numeric features
        numeric_repr = self.numeric_proj(numeric_features)

        # Combine
        combined = torch.cat([card_repr, numeric_repr], dim=-1)
        return self.combine(combined)


# Per-opponent game state: seat_onehot(9) + stack + bet + pot_committed + active + all_in = 14d
OPP_GAME_STATE_DIM = 14


class PolicyNetwork(nn.Module):
    """
    Full policy + value network with opponent cross-attention.

    Architecture:
        1. Game state encoder → state embedding
        2. Cross-attention: state embedding queries opponent embeddings + stats + game_state
        3. Action head → action type distribution (4-way)
        4. Sizing head → bet size classification (10 discrete buckets)
        5. Value head → expected value

    Handles variable numbers of opponents (2-9 players) via attention masking.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        opponent_embed_dim: int = 128,
        num_cross_attn_heads: int = 4,
        num_cross_attn_layers: int = 2,
        max_opponents: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_opponents = max_opponents

        # Game state encoder
        self.state_encoder = GameStateEncoder(embed_dim=embed_dim)

        # Project opponent embeddings + stats + game_state into the same space
        self.opponent_proj = nn.Linear(
            opponent_embed_dim + NUM_STAT_FEATURES + OPP_GAME_STATE_DIM, embed_dim
        )

        # Cross-attention layers: game state attends to opponent embeddings
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_cross_attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_cross_attn_layers)
        ])
        self.cross_ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_cross_attn_layers)
        ])

        # Own stats projection (so agent is aware of its own image)
        self.own_stats_proj = nn.Linear(NUM_STAT_FEATURES, embed_dim)

        # Final combination before heads
        self.pre_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # state+own_stats combined
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        # Action type head (4-way classification)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, NUM_ACTION_TYPES),
        )

        # Bet sizing head (10 discrete bucket logits)
        self.sizing_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 10),
        )

        # Value head — own trunk + deeper network (separate from policy)
        # V(s) needs high capacity to learn state values for GAE
        self.value_trunk = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # separate projection from combined features
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hole_cards: torch.Tensor,            # (batch, 2)
        community_cards: torch.Tensor,        # (batch, 5) — -1 for absent
        numeric_features: torch.Tensor,       # (batch, 9)
        opponent_embeddings: torch.Tensor,    # (batch, num_opponents, opponent_embed_dim)
        opponent_stats: torch.Tensor,         # (batch, num_opponents, NUM_STAT_FEATURES)
        own_stats: torch.Tensor,              # (batch, NUM_STAT_FEATURES)
        opponent_game_state: Optional[torch.Tensor] = None,  # (batch, num_opponents, 14)
        opponent_mask: Optional[torch.Tensor] = None,  # (batch, num_opponents) True=masked
        action_mask: Optional[torch.Tensor] = None,    # (batch, 4) True=legal
        sizing_mask: Optional[torch.Tensor] = None,    # (batch, 10) True=legal
    ) -> ActionOutput:
        """
        Full forward pass.

        Returns ActionOutput with action logits, probs, sizing, and value.
        """
        batch_size = hole_cards.shape[0]
        num_opps = opponent_embeddings.shape[1]

        # 1. Encode game state
        state = self.state_encoder(
            hole_cards, community_cards, numeric_features
        )  # (batch, embed_dim)

        # 2. Project opponent embeddings + stats + game_state
        if opponent_game_state is None:
            opponent_game_state = torch.zeros(
                batch_size, num_opps, OPP_GAME_STATE_DIM,
                device=opponent_embeddings.device,
            )
        opp_combined = torch.cat([opponent_embeddings, opponent_stats, opponent_game_state], dim=-1)
        opp_projected = self.opponent_proj(opp_combined)  # (batch, num_opp, embed_dim)

        # 3. Cross-attention: game state attends to opponents
        # Handle edge case: if ALL opponents are masked for a sample,
        # cross-attention produces NaN. Detect and skip for those rows.
        if opponent_mask is not None:
            all_masked = opponent_mask.all(dim=1)  # (batch,)
        else:
            all_masked = torch.zeros(batch_size, dtype=torch.bool, device=state.device)

        # State needs sequence dim: (batch, 1, embed_dim)
        query = state.unsqueeze(1)

        if not all_masked.all():  # at least some opponents to attend to
            # For rows where all opponents are masked, temporarily unmask
            # to avoid NaN, then replace with state embedding after
            safe_mask = opponent_mask.clone() if opponent_mask is not None else None
            if safe_mask is not None and all_masked.any():
                safe_mask[all_masked] = False  # unmask to avoid NaN

            for cross_attn, norm, ffn, ffn_norm in zip(
                self.cross_attn_layers, self.cross_norms,
                self.cross_ffns, self.cross_ffn_norms
            ):
                # Cross-attention
                attn_out, _ = cross_attn(
                    query, opp_projected, opp_projected,
                    key_padding_mask=safe_mask,
                )
                query = norm(query + attn_out)

                # FFN
                ffn_out = ffn(query)
                query = ffn_norm(query + ffn_out)

        # Remove sequence dim
        state_with_opponents = query.squeeze(1)  # (batch, embed_dim)

        # For all-masked rows, use raw state embedding (no opponent info = GTO)
        if all_masked.any():
            state_with_opponents[all_masked] = state[all_masked]

        # 4. Add own stats awareness
        own_repr = self.own_stats_proj(own_stats)
        combined = torch.cat([state_with_opponents, own_repr], dim=-1)

        # Policy trunk
        features = self.pre_head(combined)  # (batch, embed_dim)

        # Value trunk (separate from policy to prevent interference)
        value_features = self.value_trunk(combined)  # (batch, embed_dim)

        # 5. Action type head
        action_logits = self.action_head(features)  # (batch, 4)

        # Apply action mask (set illegal actions to -inf)
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))

        action_probs = F.softmax(action_logits, dim=-1)

        # 6. Bet sizing head
        bet_size_logits = self.sizing_head(features)  # (batch, 10)

        # Apply sizing mask (set illegal buckets to -inf)
        if sizing_mask is not None:
            bet_size_logits = bet_size_logits.masked_fill(~sizing_mask, float('-inf'))

        # 7. Value head (uses its own trunk, not shared features)
        value = self.value_head(value_features)  # (batch, 1)

        return ActionOutput(
            action_type_logits=action_logits,
            action_type_probs=action_probs,
            bet_size_logits=bet_size_logits,
            value=value,
        )

    def get_param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
