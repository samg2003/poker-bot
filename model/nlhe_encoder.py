"""
NLHE State Encoder — bridges from the engine's GameState to model inputs.

Converts raw GameState data into the tensor format expected by the
PolicyNetwork: card indices, numeric features, and action masks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from engine.game_state import GameState, Action, ActionType
from model.action_space import ActionIndex, NUM_ACTION_TYPES


class NLHEEncoder:
    """
    Converts a GameState + player perspective → model-ready tensors.

    This is the only layer that knows about the engine's internal representation.
    Everything else works with clean tensors.
    """

    @staticmethod
    def encode_cards(
        game_state: GameState,
        player_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode hole cards and community cards as indices (0-51).

        Returns:
            hole_cards: (2,) long tensor
            community_cards: (5,) long tensor (-1 for absent)
        """
        # Hole cards
        hand = game_state.hands[player_idx]
        if hand and len(hand) == 2:
            hole = torch.tensor(hand, dtype=torch.long)
        else:
            hole = torch.tensor([-1, -1], dtype=torch.long)

        # Community cards (pad to 5 with -1)
        board = list(game_state.board)
        while len(board) < 5:
            board.append(-1)
        community = torch.tensor(board[:5], dtype=torch.long)

        return hole, community

    @staticmethod
    def encode_numeric(
        game_state: GameState,
        player_idx: int,
    ) -> torch.Tensor:
        """
        Encode numeric features as a (9,) float tensor.

        Features (all normalized):
        0. pot_size / 100bb
        1. own_stack / 100bb
        2. own_bet_this_round / 100bb
        3. position (0-1, relative to button)
        4. street (0=preflop, 0.33=flop, 0.67=turn, 1=river)
        5. num_players / 9
        6. num_active / 9
        7. current_bet / 100bb
        8. min_raise / 100bb
        """
        bb = game_state.big_blind
        norm = 100.0 * bb  # normalize by 100bb

        pot = sum(game_state.bets) / norm
        own_stack = game_state.stacks[player_idx] / norm
        own_bet = game_state.bets[player_idx] / norm

        # Position relative to button (0 = button, 1 = UTG)
        num_players = game_state.num_players
        btn_pos = game_state.button
        rel_pos = (player_idx - btn_pos) % num_players
        position = rel_pos / max(num_players - 1, 1)

        street = game_state.street / 3.0  # 0, 0.33, 0.67, 1.0

        num_active = sum(1 for i in range(num_players)
                         if not game_state.folded[i] and game_state.stacks[i] > 0)

        current_bet = max(game_state.bets) / norm
        min_raise_size = game_state.min_raise / norm if hasattr(game_state, 'min_raise') else current_bet * 2

        return torch.tensor([
            pot, own_stack, own_bet, position, street,
            num_players / 9.0, num_active / 9.0,
            current_bet, min_raise_size,
        ], dtype=torch.float32)

    @staticmethod
    def encode_action_mask(game_state: GameState, player_idx: int) -> torch.Tensor:
        """
        Encode legal actions as a (4,) bool tensor.

        Maps engine ActionType → model ActionIndex.
        """
        legal = game_state.get_legal_actions(player_idx)
        mask = torch.zeros(NUM_ACTION_TYPES, dtype=torch.bool)

        for action in legal:
            if action.type == ActionType.FOLD:
                mask[ActionIndex.FOLD] = True
            elif action.type == ActionType.CHECK:
                mask[ActionIndex.CHECK] = True
            elif action.type == ActionType.CALL:
                mask[ActionIndex.CALL] = True
            elif action.type == ActionType.RAISE:
                mask[ActionIndex.RAISE] = True
            elif action.type == ActionType.ALL_IN:
                mask[ActionIndex.RAISE] = True  # all-in is a special raise

        return mask

    @staticmethod
    def decode_action(
        action_type_idx: int,
        bet_sizing: float,
        game_state: GameState,
        player_idx: int,
    ) -> Action:
        """
        Convert model output (action_type + sizing) back to engine Action.

        Args:
            action_type_idx: ActionIndex (0-3)
            bet_sizing: continuous value [0, 1] for raise sizing
            game_state: current game state
            player_idx: which player is acting

        Returns:
            Action that can be applied to the GameState.
        """
        legal = game_state.get_legal_actions(player_idx)

        if action_type_idx == ActionIndex.FOLD:
            for a in legal:
                if a.type == ActionType.FOLD:
                    return a
        elif action_type_idx == ActionIndex.CHECK:
            for a in legal:
                if a.type == ActionType.CHECK:
                    return a
        elif action_type_idx == ActionIndex.CALL:
            for a in legal:
                if a.type == ActionType.CALL:
                    return a
        elif action_type_idx == ActionIndex.RAISE:
            # Map sizing [0, 1] → [min_raise, all_in]
            raise_actions = [a for a in legal
                             if a.type in (ActionType.RAISE, ActionType.ALL_IN)]
            if raise_actions:
                # Find min and max raise amounts
                min_amt = min(a.amount for a in raise_actions if a.amount > 0)
                max_amt = max(a.amount for a in raise_actions)

                if min_amt == max_amt:
                    target = min_amt
                else:
                    target = min_amt + bet_sizing * (max_amt - min_amt)

                # Find closest legal raise
                best = min(raise_actions, key=lambda a: abs(a.amount - target))
                return best

        # Fallback: pick first legal action
        return legal[0] if legal else Action(ActionType.FOLD, 0)

    @classmethod
    def encode_state(
        cls,
        game_state: GameState,
        player_idx: int,
    ) -> dict:
        """
        Full encoding: returns dict of tensors ready for model forward pass.

        Returns dict with keys:
            hole_cards, community_cards, numeric_features, action_mask
        """
        hole, community = cls.encode_cards(game_state, player_idx)
        numeric = cls.encode_numeric(game_state, player_idx)
        action_mask = cls.encode_action_mask(game_state, player_idx)

        return {
            'hole_cards': hole.unsqueeze(0),          # (1, 2)
            'community_cards': community.unsqueeze(0), # (1, 5)
            'numeric_features': numeric.unsqueeze(0),  # (1, 9)
            'action_mask': action_mask.unsqueeze(0),   # (1, 4)
        }
