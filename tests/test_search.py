"""
Tests for Phase 4: lightweight search and range estimation.
"""

import pytest
import torch

from search.range_estimator import (
    RangeEstimator, NUM_COMBOS, _combo_index, _combo_cards,
    get_dead_card_mask, uniform_range,
)
from search.search import (
    SearchEngine, SearchConfig, SearchState, SearchNode,
)
from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder


# =============================================================================
# Range Estimator Tests
# =============================================================================

class TestComboIndexing:
    def test_combo_index_symmetry(self):
        """combo_index(a,b) == combo_index(b,a)"""
        assert _combo_index(0, 1) == _combo_index(1, 0)
        assert _combo_index(5, 20) == _combo_index(20, 5)

    def test_combo_roundtrip(self):
        """index → cards → index should be identity."""
        for idx in [0, 100, 500, 1000, 1325]:
            c1, c2 = _combo_cards(idx)
            assert _combo_index(c1, c2) == idx

    def test_total_combos(self):
        """52 choose 2 = 1326."""
        indices = set()
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                indices.add(_combo_index(c1, c2))
        assert len(indices) == NUM_COMBOS

    def test_range_bounds(self):
        """All combo indices should be in [0, 1325]."""
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                idx = _combo_index(c1, c2)
                assert 0 <= idx < NUM_COMBOS


class TestRangeEstimator:
    def test_output_shape(self):
        estimator = RangeEstimator(opponent_embed_dim=64, game_context_dim=10)
        opp_embed = torch.randn(2, 64)
        game_ctx = torch.randn(2, 10).to(torch.float32)
        probs = estimator(opp_embed, game_ctx)
        assert probs.shape == (2, NUM_COMBOS)

    def test_output_is_distribution(self):
        estimator = RangeEstimator(opponent_embed_dim=64, game_context_dim=10)
        opp_embed = torch.randn(1, 64)
        game_ctx = torch.randn(1, 10)
        probs = estimator(opp_embed, game_ctx)
        assert abs(probs.sum().item() - 1.0) < 1e-4
        assert (probs >= 0).all()

    def test_dead_card_masking(self):
        estimator = RangeEstimator(opponent_embed_dim=64, game_context_dim=10)
        opp_embed = torch.randn(1, 64)
        game_ctx = torch.randn(1, 10)

        # Create dead mask (mark ~half as dead)
        dead = torch.zeros(1, NUM_COMBOS, dtype=torch.bool)
        dead[0, :600] = True

        probs = estimator(opp_embed, game_ctx, dead_cards=dead)
        # Dead combos should have near-zero probability
        assert probs[0, :600].sum().item() < 1e-4

    def test_gradient_flow(self):
        estimator = RangeEstimator(opponent_embed_dim=64, game_context_dim=10)
        opp_embed = torch.randn(1, 64, requires_grad=True)
        game_ctx = torch.randn(1, 10).to(torch.float32)
        probs = estimator(opp_embed, game_ctx)
        probs.sum().backward()
        assert opp_embed.grad is not None


class TestDeadCardMask:
    def test_own_hand_is_dead(self):
        mask = get_dead_card_mask(board=[], own_hand=(0, 1))
        # Combo (0, 1) should be dead
        idx = _combo_index(0, 1)
        assert mask[idx]

    def test_board_cards_are_dead(self):
        mask = get_dead_card_mask(board=[10, 20, 30], own_hand=(0, 1))
        # Any combo containing board cards should be dead
        idx = _combo_index(10, 20)
        assert mask[idx]

    def test_uniform_range_sums_to_one(self):
        probs = uniform_range(board=[10, 20, 30], own_hand=(0, 1))
        assert abs(probs.sum().item() - 1.0) < 1e-5
        # Dead combos should be zero
        mask = get_dead_card_mask([10, 20, 30], (0, 1))
        assert (probs[mask] == 0).all()


# =============================================================================
# Search State Tests
# =============================================================================

class TestSearchState:
    def test_initial_actions(self):
        state = SearchState(
            pot=10.0, stacks=[100.0, 100.0], bets=[0.0, 0.0],
            board=[], street=1, current_player=0, num_players=2,
        )
        actions = state.get_actions()
        assert 'check' in actions
        assert 'fold' not in actions  # no bet to fold to

    def test_facing_bet_actions(self):
        state = SearchState(
            pot=10.0, stacks=[100.0, 90.0], bets=[0.0, 10.0],
            board=[], street=1, current_player=0, num_players=2,
        )
        actions = state.get_actions()
        assert 'fold' in actions
        assert 'call' in actions
        assert 'check' not in actions

    def test_fold_ends_hand(self):
        state = SearchState(
            pot=10.0, stacks=[100.0, 90.0], bets=[0.0, 10.0],
            board=[], street=1, current_player=0, num_players=2,
        )
        new_state = state.apply('fold')
        assert new_state.is_terminal

    def test_call_adds_to_pot(self):
        state = SearchState(
            pot=10.0, stacks=[100.0, 90.0], bets=[0.0, 10.0],
            board=[], street=1, current_player=0, num_players=2,
        )
        new_state = state.apply('call')
        assert new_state.pot == 20.0
        assert new_state.stacks[0] == 90.0

    def test_raise_increases_bet(self):
        state = SearchState(
            pot=10.0, stacks=[100.0, 100.0], bets=[0.0, 0.0],
            board=[], street=1, current_player=0, num_players=2,
        )
        new_state = state.apply('raise_0.5')  # half pot
        assert new_state.bets[0] > 0
        assert new_state.pot > 10.0


# =============================================================================
# Search Node Tests
# =============================================================================

class TestSearchNode:
    def test_initial_uniform_strategy(self):
        node = SearchNode(num_actions=3)
        strategy = node.get_strategy()
        assert len(strategy) == 3
        assert abs(sum(strategy) - 1.0) < 1e-6
        assert all(abs(s - 1/3) < 1e-6 for s in strategy)

    def test_regret_matching(self):
        node = SearchNode(num_actions=3)
        node.regret_sum = [10.0, 0.0, 5.0]
        strategy = node.get_strategy()
        # Action 0 should have highest probability
        assert strategy[0] > strategy[1]
        assert strategy[0] > strategy[2]
        assert abs(sum(strategy) - 1.0) < 1e-6

    def test_negative_regrets_ignored(self):
        node = SearchNode(num_actions=3)
        node.regret_sum = [-5.0, 3.0, -2.0]
        strategy = node.get_strategy()
        assert strategy[0] == 0.0
        assert strategy[2] == 0.0
        assert abs(strategy[1] - 1.0) < 1e-6


# =============================================================================
# Search Engine Tests
# =============================================================================

class TestSearchEngine:
    @pytest.fixture
    def engine(self):
        policy = PolicyNetwork(embed_dim=32, opponent_embed_dim=32,
                               num_cross_attn_heads=2, num_cross_attn_layers=1)
        encoder = OpponentEncoder(embed_dim=32, num_layers=1, num_heads=2)
        config = SearchConfig(num_iterations=10, raise_sizes=(0.5, 1.0))
        return SearchEngine(policy, encoder, config=config)

    def test_should_search(self, engine):
        # High entropy, big pot → should search
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert engine.should_search(probs, pot_bb=30.0, street=2)

        # Low entropy → should NOT search
        probs = torch.tensor([0.01, 0.01, 0.01, 0.97])
        assert not engine.should_search(probs, pot_bb=30.0, street=2)

        # Small pot → should NOT search
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        assert not engine.should_search(probs, pot_bb=5.0, street=2)

    def test_search_runs(self, engine):
        """Search completes without errors and returns valid output."""
        actions, probs = engine.search(
            pot=20.0, stacks=[100.0, 100.0],
            board=[10, 20, 30], street=1, hero=0,
            num_iterations=5,
        )
        assert len(actions) > 0
        assert len(probs) == len(actions)
        assert abs(sum(probs) - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_search_creates_nodes(self, engine):
        engine.search(
            pot=20.0, stacks=[100.0, 100.0],
            board=[], street=1, hero=0,
            num_iterations=10,
        )
        stats = engine.get_search_stats()
        assert stats['num_nodes'] > 0
        assert stats['total_visits'] > 0

    def test_search_different_positions(self, engine):
        """Search should work from different player positions."""
        for hero in [0, 1]:
            actions, probs = engine.search(
                pot=15.0, stacks=[80.0, 120.0],
                board=[5, 15, 25], street=1, hero=hero,
                num_iterations=5,
            )
            assert len(actions) > 0
            assert abs(sum(probs) - 1.0) < 1e-5

    def test_evaluate_leaf(self, engine):
        """Leaf evaluation should produce finite values."""
        state = SearchState(
            pot=10.0, stacks=[100.0, 100.0], bets=[0.0, 0.0],
            board=[10, 20, 30], street=1, current_player=0, num_players=2,
        )
        value = engine.evaluate_leaf(state, player=0)
        assert isinstance(value, float)
        assert not (value != value)  # not NaN


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
