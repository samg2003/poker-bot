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


class ActionOutput(NamedTuple):
    """Model output for a single decision."""
    action_type_logits: torch.Tensor   # (batch, 4)
    action_type_probs: torch.Tensor    # (batch, 4) — softmaxed
    bet_sizing: torch.Tensor           # (batch, 1) — sigmoid, [0, 1]
    value: torch.Tensor                # (batch, 1) — expected value


def encode_action(
    action_type: int,
    bet_size_frac: float,
    pot_size: float,
    stack_size: float,
    street: int,
) -> torch.Tensor:
    """
    Encode a single observed action as a feature vector.

    Used to build action history sequences for the opponent encoder.

    Returns: (7,) tensor:
        [action_one_hot(4), bet_size/pot, pot/100bb, street/3]
    """
    # One-hot action type
    action_oh = [0.0] * NUM_ACTION_TYPES
    if 0 <= action_type < NUM_ACTION_TYPES:
        action_oh[action_type] = 1.0

    features = action_oh + [
        bet_size_frac,           # bet size as fraction of pot
        pot_size / 100.0,        # pot normalized by 100bb
        street / 3.0,            # street: 0=preflop, 1=flop, 2=turn, 3=river
    ]
    return torch.tensor(features, dtype=torch.float32)


# Dimension of one encoded action token
ACTION_FEATURE_DIM = NUM_ACTION_TYPES + 3  # 4 + 3 = 7
