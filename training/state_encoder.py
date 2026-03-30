"""
State Encoder — stateless functions for encoding game state to model tensors.

Extracted from NLHESelfPlayTrainer. All functions are pure — they depend
only on their inputs and a device parameter.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from engine.game_state import GameState, ActionType, Street
from model.action_space import (
    ActionIndex, NUM_ACTION_TYPES, get_sizing_mask,
)
from model.policy_network import OPP_GAME_STATE_DIM
from model.stat_tracker import StatTracker, NUM_STAT_FEATURES


def encode_action_mask(game_state: GameState, device: torch.device) -> torch.Tensor:
    """Encode legal actions as (1, 4) bool tensor on device."""
    legal_types = game_state.get_legal_actions()
    mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool, device=device)

    for at in legal_types:
        if at == ActionType.FOLD:
            mask[0, ActionIndex.FOLD] = True
        elif at == ActionType.CHECK:
            mask[0, ActionIndex.CHECK] = True
        elif at == ActionType.CALL:
            mask[0, ActionIndex.CALL] = True
        elif at == ActionType.RAISE:
            mask[0, ActionIndex.RAISE] = True
        elif at == ActionType.ALL_IN:
            mask[0, ActionIndex.RAISE] = True

    return mask


def encode_state(game_state: GameState, player_idx: int, device: torch.device) -> dict:
    """Encode game state to device tensors for model input.

    Returns dict with: hole_cards, community_cards, numeric_features, action_mask, sizing_mask
    """
    p = game_state.players[player_idx]

    hole = torch.tensor([list(p.hole_cards)], dtype=torch.long, device=device)
    board = list(game_state.board)
    while len(board) < 5:
        board.append(-1)
    community = torch.tensor([board[:5]], dtype=torch.long, device=device)

    bb = game_state.big_blind
    norm = 100.0 * bb
    pot = game_state.pot / norm
    own_stack = p.stack / norm
    own_bet = p.bet_this_street / norm

    # 9-dim seat one-hot (relative position from BTN)
    rel_pos = (player_idx - game_state.dealer_button) % game_state.num_players
    seat_onehot = [0.0] * 9
    seat_onehot[rel_pos] = 1.0

    # IP flag
    active_positions = []
    for i, pp in enumerate(game_state.players):
        if pp.is_active:
            active_positions.append((i - game_state.dealer_button) % game_state.num_players)
    ip_flag = 1.0 if (active_positions and rel_pos == max(active_positions)) else 0.0

    # 4-dim street one-hot
    street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
    street_idx = street_map.get(game_state.street, 0)
    street_onehot = [0.0] * 4
    street_onehot[street_idx] = 1.0

    num_active = sum(1 for pp in game_state.players if pp.is_active)
    current_bet = game_state.current_bet / norm
    min_raise = game_state.min_raise / norm
    amount_to_call = max(0.0, current_bet - own_bet)
    # Effective SPR
    active_opp_stacks = [
        pp.stack for j, pp in enumerate(game_state.players)
        if j != player_idx and pp.is_active
    ]
    max_opp_stack = max(active_opp_stacks) if active_opp_stacks else p.stack
    effective_stack = min(p.stack, max_opp_stack)
    spr = effective_stack / max(game_state.pot, 0.01)

    numeric = torch.tensor([[
        pot, own_stack, own_bet,
        *seat_onehot,
        ip_flag,
        *street_onehot,
        game_state.num_players / 9.0, num_active / 9.0,
        current_bet, min_raise, amount_to_call,
        spr,
    ]], dtype=torch.float32, device=device)

    action_mask = encode_action_mask(game_state, device)
    sizing_mask = get_sizing_mask(game_state, spr=spr).unsqueeze(0).to(device)

    return {
        'hole_cards': hole,
        'community_cards': community,
        'numeric_features': numeric,
        'action_mask': action_mask,
        'sizing_mask': sizing_mask,
    }


def get_opponent_stats(
    stat_tracker: StatTracker,
    hero_id: int,
    num_players: int,
    device: torch.device,
) -> torch.Tensor:
    """Get HUD stats for all opponents. Returns (1, num_opp, stat_dim)."""
    stats = []
    for pid in range(num_players):
        if pid == hero_id:
            continue
        stats.append(stat_tracker.get_stats(pid))

    if not stats:
        return torch.zeros(1, 1, NUM_STAT_FEATURES, device=device)

    return torch.stack(stats).unsqueeze(0).to(device)


def get_opponent_game_state(
    game_state: GameState,
    hero_id: int,
    num_players: int,
    device: torch.device,
) -> torch.Tensor:
    """Build per-opponent game state tensor: seat_onehot(9) + stack + bet + pot_committed + active + all_in.

    Returns (1, num_opp, OPP_GAME_STATE_DIM).
    """
    bb = max(game_state.big_blind, 1.0)
    pot = max(game_state.pot, 1.0)
    opp_states = []
    for pid in range(num_players):
        if pid == hero_id:
            continue
        p = game_state.players[pid]
        seat_oh = [0.0] * 9
        seat_oh[min(pid, 8)] = 1.0
        opp_states.append(torch.tensor(
            seat_oh + [
                p.stack / (100.0 * bb),
                p.bet_this_street / pot if pot > 0 else 0.0,
                p.bet_total / (100.0 * bb),
                float(p.is_active),
                float(p.is_all_in),
            ], dtype=torch.float32
        ))
    if not opp_states:
        return torch.zeros(1, 1, OPP_GAME_STATE_DIM, device=device)
    return torch.stack(opp_states).unsqueeze(0).to(device)


def compute_hero_ev(game_state: GameState, hero_idx: int, mc_sims: int = 500) -> float:
    """Side-pot-aware hero EV via MC equity per pot layer.

    Uses calculate_side_pots() which works mid-hand. For each pot layer,
    computes MC equity only among eligible players, then sums hero's
    expected share across all layers.
    """
    from engine.hand_evaluator import Eval7Evaluator
    side_pots = game_state.calculate_side_pots()
    hero_ev = 0.0
    for pot_amount, eligible in side_pots:
        if hero_idx not in eligible:
            continue
        hands = [list(game_state.players[i].hole_cards) for i in eligible]
        equities = Eval7Evaluator.get_equity(
            hands, list(game_state.board),
            runouts=mc_sims,
        )
        hero_pos = eligible.index(hero_idx)
        hero_ev += equities[hero_pos] * pot_amount
    return hero_ev - game_state.players[hero_idx].bet_total
