"""
Range Estimator — estimates opponent's hole card distribution.

Given opponent embedding + action history this hand, estimates a probability
distribution over the 1326 possible hole card combos (52 choose 2).

Used by the search module to weight opponent responses in the CFR subtree.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Total possible 2-card combos from 52-card deck
NUM_COMBOS = 1326


def _combo_index(c1: int, c2: int) -> int:
    """Map two card indices (0-51) to a combo index (0-1325)."""
    lo, hi = min(c1, c2), max(c1, c2)
    # Index = hi*(hi-1)/2 + lo
    return hi * (hi - 1) // 2 + lo


def _combo_cards(idx: int) -> Tuple[int, int]:
    """Map combo index back to two card indices."""
    # Find hi such that hi*(hi-1)/2 <= idx
    hi = 1
    while hi * (hi + 1) // 2 <= idx:
        hi += 1
    lo = idx - hi * (hi - 1) // 2
    return lo, hi


class RangeEstimator(nn.Module):
    """
    Neural network that estimates an opponent's hand range.

    Input: opponent embedding (from OpponentEncoder) + game context
    Output: probability distribution over 1326 combos

    The network learns which hands an opponent is likely to hold
    given their history and current game state.
    """

    def __init__(
        self,
        opponent_embed_dim: int = 128,
        game_context_dim: int = 9,
        hidden_dim: int = 256,
    ):
        super().__init__()

        input_dim = opponent_embed_dim + game_context_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, NUM_COMBOS),
        )

    def forward(
        self,
        opponent_embedding: torch.Tensor,  # (batch, embed_dim)
        game_context: torch.Tensor,        # (batch, game_context_dim)
        dead_cards: Optional[torch.Tensor] = None,  # (batch, NUM_COMBOS) bool
    ) -> torch.Tensor:
        """
        Estimate opponent's range.

        Args:
            opponent_embedding: learned opponent representation
            game_context: numeric game state features
            dead_cards: mask of impossible combos (cards we can see)

        Returns:
            range_probs: (batch, NUM_COMBOS) — probability over hole card combos
        """
        x = torch.cat([opponent_embedding, game_context], dim=-1)
        logits = self.network(x)

        # Zero out dead cards (combos that include visible cards)
        if dead_cards is not None:
            logits = logits.masked_fill(dead_cards, float('-inf'))

        # Softmax → probabilities
        range_probs = F.softmax(logits, dim=-1)

        return range_probs


def get_dead_card_mask(
    board: List[int],
    own_hand: Tuple[int, int],
) -> torch.Tensor:
    """
    Create a dead card mask for the range estimator.

    Any combo containing a card we can see (our hand + board) is impossible.

    Returns: (NUM_COMBOS,) bool tensor — True = dead (impossible) combo
    """
    visible = set(board) | set(own_hand)
    mask = torch.zeros(NUM_COMBOS, dtype=torch.bool)

    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            idx = _combo_index(c1, c2)
            if c1 in visible or c2 in visible:
                mask[idx] = True

    return mask


def uniform_range(board: List[int], own_hand: Tuple[int, int]) -> torch.Tensor:
    """
    Get a uniform range over all possible combos (prior before any reads).

    Returns: (NUM_COMBOS,) float tensor — uniform probability, dead cards zeroed.
    """
    dead = get_dead_card_mask(board, own_hand)
    probs = torch.ones(NUM_COMBOS)
    probs[dead] = 0.0
    probs = probs / probs.sum()
    return probs
