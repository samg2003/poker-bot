"""
Tests for all model components:
- Action space encoding
- Stat tracker
- Opponent encoder (Transformer)
- Policy network (cross-attention + action/sizing/value heads)
"""

import pytest
import torch
import torch.nn as nn

from model.action_space import (
    NUM_ACTION_TYPES, ACTION_FEATURE_DIM, ActionIndex,
    ActionOutput, encode_action,
)
from model.stat_tracker import StatTracker, HandRecord, NUM_STAT_FEATURES
from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork, CardEmbedding, GameStateEncoder


# =============================================================================
# Action Space Tests
# =============================================================================

class TestActionSpace:
    def test_action_indices(self):
        assert ActionIndex.FOLD == 0
        assert ActionIndex.CHECK == 1
        assert ActionIndex.CALL == 2
        assert ActionIndex.RAISE == 3
        assert NUM_ACTION_TYPES == 4

    def test_encode_action(self):
        token = encode_action(
            action_type=ActionIndex.RAISE,
            bet_size_frac=0.75,
            pot_size=10.0,
            stack_size=100.0,
            street=1,
        )
        assert token.shape == (ACTION_FEATURE_DIM,)
        # Check one-hot
        assert token[ActionIndex.RAISE] == 1.0
        assert token[ActionIndex.FOLD] == 0.0
        # Check features
        assert token[4] == 0.75  # bet_size_frac
        assert abs(token[5] - 0.1) < 1e-5  # pot/100
        assert abs(token[6] - 1.0) < 1e-5  # stack/100
        # Check street one-hot (street=1 means flop)
        assert token[8] == 0.0   # preflop
        assert token[9] == 1.0   # flop
        assert token[10] == 0.0  # turn
        assert token[11] == 0.0  # river

    def test_action_feature_dim(self):
        assert ACTION_FEATURE_DIM == 13  # 4 one-hot + 9 (bet, pot, stack, pos, street_oh[4], boundary)


# =============================================================================
# Stat Tracker Tests
# =============================================================================

class TestStatTracker:
    def test_empty_stats_are_zeros(self):
        tracker = StatTracker()
        stats = tracker.get_stats(player_id=0)
        assert stats.shape == (NUM_STAT_FEATURES,)
        assert stats.sum() == 0.0, "No hands recorded → all zeros (GTO mode)"

    def test_vpip_calculation(self):
        tracker = StatTracker()
        for _ in range(10):
            tracker.record_hand(0, HandRecord(vpip=True))
        for _ in range(10):
            tracker.record_hand(0, HandRecord(vpip=False))

        stats = tracker.get_stats(0)
        vpip = stats[0].item()
        assert abs(vpip - 0.5) < 1e-5, f"VPIP should be 50%, got {vpip}"

    def test_pfr_calculation(self):
        tracker = StatTracker()
        for _ in range(6):
            tracker.record_hand(0, HandRecord(vpip=True, pfr=True))
        for _ in range(4):
            tracker.record_hand(0, HandRecord(vpip=True, pfr=False))

        stats = tracker.get_stats(0)
        pfr = stats[1].item()
        assert abs(pfr - 0.6) < 1e-5, f"PFR should be 60%, got {pfr}"

    def test_stats_output_size(self):
        tracker = StatTracker()
        for _ in range(20):
            tracker.record_hand(0, HandRecord(
                vpip=True, pfr=True, saw_flop=True,
                went_to_showdown=True, won_at_showdown=True,
                bet_sizes=[0.5, 0.75], result=5.0,
            ))
        stats = tracker.get_stats(0)
        assert stats.shape == (NUM_STAT_FEATURES,)
        assert not torch.isnan(stats).any(), "Stats should not contain NaN"

    def test_reset(self):
        tracker = StatTracker()
        tracker.record_hand(0, HandRecord(vpip=True))
        tracker.reset(0)
        assert tracker.get_num_hands(0) == 0
        assert tracker.get_stats(0).sum() == 0.0

    def test_multiple_players(self):
        tracker = StatTracker()
        tracker.record_hand(0, HandRecord(vpip=True))
        tracker.record_hand(1, HandRecord(vpip=False))
        assert tracker.get_num_hands(0) == 1
        assert tracker.get_num_hands(1) == 1
        assert tracker.get_stats(0)[0] > 0  # P0 has VPIP
        assert tracker.get_stats(1)[0] == 0  # P1 does not


# =============================================================================
# Opponent Encoder Tests
# =============================================================================

class TestOpponentEncoder:
    def test_output_shape(self):
        encoder = OpponentEncoder(embed_dim=128)
        # Batch of 4, sequence of 20 actions
        x = torch.randn(4, 20, ACTION_FEATURE_DIM)
        out = encoder(x)
        assert out.shape == (4, 128)

    def test_variable_sequence_length(self):
        encoder = OpponentEncoder(embed_dim=64)
        # Different sequence lengths
        short = torch.randn(2, 5, ACTION_FEATURE_DIM)
        long = torch.randn(2, 100, ACTION_FEATURE_DIM)

        out_short = encoder(short)
        out_long = encoder(long)

        assert out_short.shape == (2, 64)
        assert out_long.shape == (2, 64)

    def test_with_padding_mask(self):
        encoder = OpponentEncoder(embed_dim=64)
        x = torch.randn(2, 10, ACTION_FEATURE_DIM)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, 5:] = True  # First batch: only 5 real actions
        mask[1, 8:] = True  # Second batch: 8 real actions

        out = encoder(x, mask=mask)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()

    def test_empty_embedding(self):
        encoder = OpponentEncoder(embed_dim=128)
        out = encoder.encode_empty(batch_size=3)
        assert out.shape == (3, 128)
        assert out.sum() == 0.0

    def test_gradient_flow(self):
        encoder = OpponentEncoder(embed_dim=64, num_layers=2)
        x = torch.randn(2, 10, ACTION_FEATURE_DIM, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow to input"


# =============================================================================
# Policy Network Tests
# =============================================================================

class TestPolicyNetwork:
    @pytest.fixture
    def policy(self):
        return PolicyNetwork(
            embed_dim=64,
            opponent_embed_dim=64,
            num_cross_attn_heads=4,
            num_cross_attn_layers=2,
        )

    @pytest.fixture
    def batch_inputs(self):
        """Create a batch of valid inputs for the policy network."""
        batch = 4
        num_opp = 3
        return {
            'hole_cards': torch.randint(0, 52, (batch, 2)),
            'community_cards': torch.cat([
                torch.randint(0, 52, (batch, 3)),
                torch.full((batch, 2), -1, dtype=torch.long),
            ], dim=1),  # flop only
            'numeric_features': torch.randn(batch, 23),
            'opponent_embeddings': torch.randn(batch, num_opp, 64),
            'opponent_stats': torch.randn(batch, num_opp, NUM_STAT_FEATURES),
            'own_stats': torch.randn(batch, NUM_STAT_FEATURES),
        }

    def test_output_shapes(self, policy, batch_inputs):
        output = policy(**batch_inputs)
        batch = 4
        assert output.action_type_logits.shape == (batch, NUM_ACTION_TYPES)
        assert output.action_type_probs.shape == (batch, NUM_ACTION_TYPES)
        assert output.bet_size_logits.shape == (batch, 10)
        assert output.value.shape == (batch, 1)

    def test_action_probs_sum_to_one(self, policy, batch_inputs):
        output = policy(**batch_inputs)
        sums = output.action_type_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_action_mask(self, policy, batch_inputs):
        # Only FOLD and CALL are legal
        mask = torch.tensor([
            [True, False, True, False],  # fold + call
        ] * 4)
        batch_inputs['action_mask'] = mask
        output = policy(**batch_inputs)

        # Check and call should have zero prob
        assert (output.action_type_probs[:, ActionIndex.CHECK] < 1e-6).all()
        assert (output.action_type_probs[:, ActionIndex.RAISE] < 1e-6).all()

    def test_opponent_mask(self, policy, batch_inputs):
        """Test with some opponents masked (e.g., folded players)."""
        opp_mask = torch.tensor([
            [False, True, True],    # only 1 active opponent
            [False, False, True],   # 2 active
            [False, False, False],  # all 3 active
            [True, True, True],     # all masked (heads-up, no info)
        ])
        batch_inputs['opponent_mask'] = opp_mask
        output = policy(**batch_inputs)
        assert output.action_type_probs.shape == (4, NUM_ACTION_TYPES)
        assert not torch.isnan(output.action_type_probs).any()

    def test_variable_num_opponents(self, policy):
        """Test with different numbers of opponents (2-8)."""
        for num_opp in [1, 3, 5, 8]:
            batch = 2
            inputs = {
                'hole_cards': torch.randint(0, 52, (batch, 2)),
                'community_cards': torch.full((batch, 5), -1, dtype=torch.long),
                'numeric_features': torch.randn(batch, 23),
                'opponent_embeddings': torch.randn(batch, num_opp, 64),
                'opponent_stats': torch.randn(batch, num_opp, NUM_STAT_FEATURES),
                'own_stats': torch.randn(batch, NUM_STAT_FEATURES),
            }
            output = policy(**inputs)
            assert output.action_type_probs.shape == (batch, NUM_ACTION_TYPES)

    def test_gradient_flow_full_model(self, policy, batch_inputs):
        """Verify gradients flow through the entire network."""
        output = policy(**batch_inputs)
        loss = output.value.mean() + output.action_type_logits.sum() + output.bet_size_logits.sum()
        loss.backward()

        # Check key layers have gradients
        assert policy.state_encoder.card_embed.rank_embed.weight.grad is not None
        assert policy.opponent_proj.weight.grad is not None
        assert policy.action_head[-1].weight.grad is not None
        assert policy.sizing_head[-1].weight.grad is not None
        assert policy.value_head[-1].weight.grad is not None

    def test_param_count(self, policy):
        count = policy.get_param_count()
        assert count > 0
        # With embed_dim=64, should be in the ~100K range
        assert count < 5_000_000, f"Model too large: {count} params"


class TestCardEmbedding:
    def test_basic(self):
        embed = CardEmbedding(embed_dim=32)
        cards = torch.tensor([[0, 51]])  # 2c and As
        out = embed(cards)
        assert out.shape == (1, 2, 32)

    def test_absent_cards(self):
        embed = CardEmbedding(embed_dim=32)
        cards = torch.tensor([[0, -1, -1]])  # 2c and two absent
        out = embed(cards)
        assert out.shape == (1, 3, 32)
        # Absent cards should be zeroed
        assert out[0, 1].abs().sum() == 0
        assert out[0, 2].abs().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
