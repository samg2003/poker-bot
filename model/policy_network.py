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

from model.action_space import NUM_ACTION_TYPES, ActionOutput, ACTION_FEATURE_DIM
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
        safe_cards = cards.clamp(min=0, max=51)

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

        # Combine cards + numeric + hand_story
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        hole_cards: torch.Tensor,         # (batch, 2) card indices
        community_cards: torch.Tensor,    # (batch, 5) card indices (-1 if absent)
        numeric_features: torch.Tensor,   # (batch, 23)
        hand_story: Optional[torch.Tensor] = None,  # (batch, embed_dim)
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

        # Hand story (zeros if not provided — backward compat)
        if hand_story is None:
            hand_story = torch.zeros(
                batch_size, self.combine[0].in_features // 3,
                device=hole_cards.device,
            )

        # Combine cards + numeric + hand_story
        combined = torch.cat([card_repr, numeric_repr, hand_story], dim=-1)
        return self.combine(combined)


# Per-opponent game state: seat_onehot(9) + stack + bet + pot_committed + active + all_in = 14d
OPP_GAME_STATE_DIM = 14

# Cached player profile dimension
PROFILE_DIM = 64

# Hand history token dim: raw_action(13d) + actor_profile(64d) = 77d
HAND_ACTION_DIM = ACTION_FEATURE_DIM + PROFILE_DIM

# Max actions per hand
MAX_HAND_ACTIONS = 40


class ProfileBuilder(nn.Module):
    """Compress [encoder_output + HUD_stats + is_hero] into a 64d cached profile.

    Built once per hand for each player, reused in cross-attention and hand history.
    """

    def __init__(self, encoder_dim: int, stat_features: int = NUM_STAT_FEATURES):
        super().__init__()
        self.stats_proj = nn.Linear(stat_features, 64)
        # encoder_dim + 64 (projected stats) + 1 (is_hero flag)
        self.combine = nn.Sequential(
            nn.Linear(encoder_dim + 64 + 1, PROFILE_DIM),
            nn.GELU(),
            nn.LayerNorm(PROFILE_DIM),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,   # (batch, encoder_dim)
        stats: torch.Tensor,            # (batch, stat_features)
        is_hero: torch.Tensor,          # (batch, 1)
    ) -> torch.Tensor:
        """Returns (batch, 64) cached profile."""
        stats_repr = self.stats_proj(stats)
        combined = torch.cat([encoder_output, stats_repr, is_hero], dim=-1)
        return self.combine(combined)


class HandHistoryEncoder(nn.Module):
    """Transformer encoder for intra-hand action sequences.

    Each token is [raw_action(13d) + actor_cached_profile(64d)] = 77d.
    Uses 2-layer full self-attention (non-causal: all tokens attend to all
    valid tokens) + attention pooling → hand_story (embed_dim).

    Advantages over GRU:
    - Parallelizable: all tokens processed simultaneously (CPU speedup)
    - No recency bias: river decision can directly attend to preflop 3-bet
    - Direct attention over any pair of actions across streets
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, num_layers: int = 2,
                 max_len: int = MAX_HAND_ACTIONS, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Project raw action tokens (77d) into model dimension
        self.input_proj = nn.Linear(HAND_ACTION_DIM, embed_dim)

        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(max_len + 1, embed_dim)

        # Transformer encoder (pre-norm for gradient stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,         # pre-norm: more stable than post-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)


        # Attention pooling: learned scalar per token → weighted sum
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        action_seq: torch.Tensor,     # (batch, max_seq, HAND_ACTION_DIM)
        seq_lengths: torch.Tensor,    # (batch,) — actual lengths, ≥ 1
    ) -> torch.Tensor:
        """Returns (batch, embed_dim) hand_story."""
        B, T, _ = action_seq.shape

        # Project tokens + add positional embeddings
        positions = torch.arange(T, device=action_seq.device).unsqueeze(0)  # (1, T)
        x = self.input_proj(action_seq) + self.pos_embedding(positions)     # (B, T, embed_dim)

        # Padding mask: True = ignore (padded position)
        seq_lengths = seq_lengths.clamp(min=1)
        pad_mask = torch.arange(T, device=action_seq.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)  # (B, T)

        # Full self-attention over valid tokens
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, T, embed_dim)

        # Attention pooling over valid tokens → (B, embed_dim)
        attn_w = self.attn_pool(x).squeeze(-1)          # (B, T)
        attn_w = attn_w.masked_fill(pad_mask, -1e9)     # mask padding
        attn_w = torch.softmax(attn_w, dim=-1)           # (B, T)
        pooled = (attn_w.unsqueeze(-1) * x).sum(dim=1)  # (B, embed_dim)

        return self.output_norm(pooled)



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

        # Phase 6: Profile builder — compresses encoder_output + stats + is_hero → 64d
        self.profile_builder = ProfileBuilder(encoder_dim=opponent_embed_dim)

        # Phase 5: Hand history GRU encoder
        self.hand_history_encoder = HandHistoryEncoder(embed_dim=embed_dim)

        # Project opponent profiles + game_state into cross-attention space
        # When profiles are available: profile(64d) + game_state(14d) = 78d
        # Legacy fallback: opponent_embed + stats + game_state = opp_embed+30+14
        self.opponent_proj = nn.Linear(
            opponent_embed_dim + NUM_STAT_FEATURES + OPP_GAME_STATE_DIM, embed_dim
        )
        self.opponent_proj_v2 = nn.Linear(
            PROFILE_DIM + OPP_GAME_STATE_DIM, embed_dim
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

        # Own stats projection (legacy fallback)
        self.own_stats_proj = nn.Linear(NUM_STAT_FEATURES, embed_dim)
        # Hero profile projection (Phase 6 — used when profiles provided)
        self.hero_proj = nn.Linear(PROFILE_DIM, embed_dim)

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
        # Phase 5+6 new params (backward compatible)
        hand_action_seq: Optional[torch.Tensor] = None,   # (batch, max_seq, 13) raw actions
        hand_action_len: Optional[torch.Tensor] = None,    # (batch,) sequence lengths
        hero_profile: Optional[torch.Tensor] = None,       # (batch, 64) cached hero profile
        opponent_profiles: Optional[torch.Tensor] = None,  # (batch, num_opp, 64) cached opp profiles
        actor_profiles_seq: Optional[torch.Tensor] = None, # (batch, max_seq, 64) per-action actor profile
    ) -> ActionOutput:
        """
        Full forward pass.

        Returns ActionOutput with action logits, probs, sizing, and value.
        """
        batch_size = hole_cards.shape[0]
        num_opps = opponent_embeddings.shape[1]

        # ── Phase 5: Build hand_story ──
        hand_story = None
        if hand_action_seq is not None and actor_profiles_seq is not None:
            # Enrich raw action tokens with actor profiles → 77d
            enriched = torch.cat([hand_action_seq, actor_profiles_seq], dim=-1)  # (B, seq, 77)
            if hand_action_len is None:
                hand_action_len = (hand_action_seq.sum(dim=-1) != 0).sum(dim=-1).long()
            hand_story = self.hand_history_encoder(enriched, hand_action_len)

        # 1. Encode game state (with hand_story when available)
        state = self.state_encoder(
            hole_cards, community_cards, numeric_features, hand_story=hand_story
        )  # (batch, embed_dim)

        # 2. Project opponents for cross-attention
        if opponent_game_state is None:
            opponent_game_state = torch.zeros(
                batch_size, num_opps, OPP_GAME_STATE_DIM,
                device=opponent_embeddings.device,
            )

        # Use v2 projection when profiles are available, else legacy
        if opponent_profiles is not None:
            opp_combined = torch.cat([opponent_profiles, opponent_game_state], dim=-1)
            opp_projected = self.opponent_proj_v2(opp_combined)
        else:
            opp_combined = torch.cat([opponent_embeddings, opponent_stats, opponent_game_state], dim=-1)
            opp_projected = self.opponent_proj(opp_combined)

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

        # 4. Add hero awareness (profile or legacy stats)
        if hero_profile is not None:
            own_repr = self.hero_proj(hero_profile)
        else:
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
        value = self.value_head(value_features)  # V_res: unconstrained — can go negative for reverse implied odds

        return ActionOutput(
            action_type_logits=action_logits,
            action_type_probs=action_probs,
            bet_size_logits=bet_size_logits,
            value=value,
        )

    def get_param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
