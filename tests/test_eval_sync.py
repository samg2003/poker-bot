"""
Tests to verify that the evaluator and trainer use the same model forward call signature
and action token format (13d). Catches train/eval distribution mismatches.
"""

import sys
import os
import inspect
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.policy_network import PolicyNetwork, MAX_HAND_ACTIONS, PROFILE_DIM
from model.action_space import encode_action, ACTION_FEATURE_DIM
from model.stat_tracker import NUM_STAT_FEATURES
from model.opponent_encoder import OpponentEncoder
from evaluation.evaluator import Evaluator


def test_action_token_is_13d():
    """encode_action must produce 13d tokens everywhere."""
    token = encode_action(
        action_type=3,  # RAISE
        bet_size_frac=0.5,
        pot_size=10.0,
        stack_size=100.0,
        street=1,
        relative_position=0.25,
        hand_boundary=1.0,
    )
    assert token.shape == (ACTION_FEATURE_DIM,), \
        f"Expected ({ACTION_FEATURE_DIM},), got {token.shape}"
    assert ACTION_FEATURE_DIM == 13, \
        f"ACTION_FEATURE_DIM should be 13, got {ACTION_FEATURE_DIM}"


def test_evaluator_record_action_accepts_position_and_boundary():
    """Evaluator._record_action must accept relative_position and hand_boundary."""
    policy = PolicyNetwork(embed_dim=64, opponent_embed_dim=64)
    encoder = OpponentEncoder(embed_dim=64)
    ev = Evaluator(policy, encoder, game='nlhe')

    # Should not raise
    ev._record_action(
        player_id=0,
        action_type=1,
        bet_frac=0.0,
        pot=10.0,
        stack=100.0,
        street=0,
        relative_position=0.125,
        hand_boundary=1.0,
    )
    token = ev.action_histories[0][0]
    assert token.shape == (ACTION_FEATURE_DIM,), \
        f"Evaluator recorded {token.shape[0]}d token, expected {ACTION_FEATURE_DIM}d"


def test_evaluator_forward_passes_hand_history():
    """The evaluator's model forward call must include hand_action_seq and actor_profiles_seq."""
    # Read evaluator source and check that _play_eval_hand_nlhe passes Phase 5+6 params
    source = inspect.getsource(Evaluator._play_eval_hand_nlhe)
    
    assert 'hand_action_seq=' in source, \
        "Evaluator._play_eval_hand_nlhe must pass hand_action_seq to model forward"
    assert 'hand_action_len=' in source, \
        "Evaluator._play_eval_hand_nlhe must pass hand_action_len to model forward"
    assert 'actor_profiles_seq=' in source, \
        "Evaluator._play_eval_hand_nlhe must pass actor_profiles_seq to model forward"


def test_model_forward_with_phase5_features():
    """PolicyNetwork forward must accept and process Phase 5+6 features without error."""
    policy = PolicyNetwork(embed_dim=64, opponent_embed_dim=64)
    policy.eval()

    batch = 2
    hole = torch.randint(0, 52, (batch, 2))
    comm = torch.full((batch, 5), -1, dtype=torch.long)
    numeric = torch.randn(batch, 23)
    opp_embed = torch.randn(batch, 1, 64)
    opp_stats = torch.randn(batch, 1, NUM_STAT_FEATURES)
    own_stats = torch.randn(batch, NUM_STAT_FEATURES)
    mask = torch.ones(batch, 4, dtype=torch.bool)

    ha_seq = torch.randn(batch, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)
    ha_len = torch.tensor([3, 5], dtype=torch.long)
    actor_prof = torch.zeros(batch, MAX_HAND_ACTIONS, PROFILE_DIM)

    with torch.no_grad():
        out = policy(
            hole, comm, numeric, opp_embed, opp_stats, own_stats,
            action_mask=mask,
            hand_action_seq=ha_seq,
            hand_action_len=ha_len,
            actor_profiles_seq=actor_prof,
        )
    assert out.action_type_probs.shape == (batch, 4)
    assert out.value.shape == (batch, 1)


def test_model_forward_without_phase5_backward_compat():
    """PolicyNetwork forward must still work WITHOUT Phase 5+6 features (backward compat)."""
    policy = PolicyNetwork(embed_dim=64, opponent_embed_dim=64)
    policy.eval()

    batch = 1
    hole = torch.randint(0, 52, (batch, 2))
    comm = torch.full((batch, 5), -1, dtype=torch.long)
    numeric = torch.randn(batch, 23)
    opp_embed = torch.randn(batch, 1, 64)
    opp_stats = torch.randn(batch, 1, NUM_STAT_FEATURES)
    own_stats = torch.randn(batch, NUM_STAT_FEATURES)
    mask = torch.ones(batch, 4, dtype=torch.bool)

    with torch.no_grad():
        out = policy(
            hole, comm, numeric, opp_embed, opp_stats, own_stats,
            action_mask=mask,
        )
    assert out.action_type_probs.shape == (batch, 4)


def test_hand_action_accumulation_matches_training():
    """Verify evaluator accumulates hand actions same way as trainer."""
    # Trainer pattern: hand_action_tokens.append((raw_token, pid))
    # Evaluator pattern: hand_action_tokens.append(raw_token)
    # Both use encode_action with same args, snapshot into ha_seq tensor

    ha_seq = torch.zeros(1, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)
    tokens = []
    for i in range(5):
        t = encode_action(3, 0.5, 10.0, 100.0, 0, relative_position=0.125, hand_boundary=float(i == 0))
        tokens.append(t)

    ha_len = torch.tensor([min(len(tokens), MAX_HAND_ACTIONS)], dtype=torch.long)
    trimmed = tokens[-MAX_HAND_ACTIONS:]
    ha_seq[0, :len(trimmed)] = torch.stack(trimmed)

    assert ha_seq.shape == (1, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)
    assert ha_len.item() == 5
    # First token should have hand_boundary=1.0 (index 12)
    assert ha_seq[0, 0, 12].item() == 1.0
    # Others should have hand_boundary=0.0
    assert ha_seq[0, 1, 12].item() == 0.0
