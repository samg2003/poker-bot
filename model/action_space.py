"""
Hybrid action space for NLHE.

Two-headed output:
  Head 1 (action type):  [fold, check, call, raise]  — 4-way softmax
  Head 2 (bet sizing):   continuous value [0, 1] mapped to [min_raise, all_in]
                         (only used when action_type = raise)

Also handles encoding opponent actions as continuous features.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

import torch
import torch.nn as nn


class ActionIndex(IntEnum):
    """Indices into the action type distribution."""
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE = 3

NUM_ACTION_TYPES = 4


POT_FRACTIONS = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0, -1.0]

class ActionOutput(NamedTuple):
    """Model output for a single decision."""
    action_type_logits: torch.Tensor   # (batch, 4)
    action_type_probs: torch.Tensor    # (batch, 4) — softmaxed
    bet_size_logits: torch.Tensor      # (batch, 10) — discrete unnormalized logits
    value: torch.Tensor                # (batch, 1) — expected value


def encode_action(
    action_type: int,
    bet_size_frac: float,
    pot_size: float,
    stack_size: float,
    street: int,
    relative_position: float = 0.0,
    hand_boundary: float = 0.0,
) -> torch.Tensor:
    """
    Encode a single observed action as a feature vector.

    Used to build action history sequences for the opponent encoder.

    Returns: (13,) tensor:
        [action_one_hot(4), bet_size/pot, pot/100bb, actor_stack/100bb,
         relative_position, street_onehot(4), hand_boundary]
    """
    # One-hot action type
    action_oh = [0.0] * NUM_ACTION_TYPES
    if 0 <= action_type < NUM_ACTION_TYPES:
        action_oh[action_type] = 1.0

    # Street one-hot (4d: preflop, flop, turn, river)
    street_oh = [0.0] * 4
    if 0 <= street <= 3:
        street_oh[street] = 1.0

    features = action_oh + [
        bet_size_frac,                # bet size as fraction of pot
        pot_size / 100.0,             # pot normalized by 100bb
        stack_size / 100.0,           # actor's stack normalized by 100bb
        relative_position,            # (actor - dealer) % num_players / 8.0
    ] + street_oh + [
        hand_boundary,                # 1.0 = first action of a new hand
    ]
    return torch.tensor(features, dtype=torch.float32)


def get_sizing_mask(game_state, spr: float = 0.0) -> torch.Tensor:
    """
    Evaluate which sizing buckets are legal given the current GameState.
    spr: effective stack-to-pot ratio (min(hero, max_active_opp) / pot).
         All-in is masked when spr >= 4 (not committed enough to justify shove).
    Returns: (10,) boolean mask where True = legal.
    """
    from engine.game_state import ActionType
    mask = torch.zeros(10, dtype=torch.bool)
    
    legal_actions = game_state.get_legal_actions()
    if ActionType.RAISE not in legal_actions and ActionType.ALL_IN not in legal_actions:
        return mask

    pot = game_state.pot
    current_bet = game_state.current_bet
    min_raise = game_state.get_min_raise_to()
    max_raise = game_state.get_max_raise_to()

    for i, frac in enumerate(POT_FRACTIONS):
        if frac < 0:
            # All-in: only legal when effective SPR < 4 (committed / short stack)
            if spr < 4.0:
                mask[i] = True
            continue
            
        target_size = frac * pot
        if target_size >= min_raise - 1e-5 and target_size <= max_raise + 1e-5:
            mask[i] = True

    if not mask.any():
        mask[-1] = True  # Fallback to all-in (always allow if nothing else fits)

    return mask

# Dimension of one encoded action token
ACTION_FEATURE_DIM = NUM_ACTION_TYPES + 9  # 4 + 9 = 13

