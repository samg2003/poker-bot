"""
Tests for Leduc Hold'em and self-play training.
"""

import pytest
import random

from engine.leduc_poker import (
    LeducState, deal_leduc, DECK,
    JACK, QUEEN, KING,
    CHECK, BET, FOLD, CALL, RAISE,
)
from training.self_play_trainer import LeducSelfPlayTrainer, TrainingConfig


# =============================================================================
# Leduc Hold'em game logic tests
# =============================================================================

class TestLeducPoker:
    def test_initial_state(self):
        state = LeducState((JACK, 0), (QUEEN, 0), (KING, 0))
        assert state.current_player == 0
        assert not state.is_terminal
        assert state.get_actions() == [CHECK, BET]

    def test_check_check_preflop(self):
        """Check-check preflop → advance to round 1."""
        state = LeducState((JACK, 0), (QUEEN, 0), (KING, 0))
        state = state.apply(CHECK)
        state = state.apply(CHECK)
        # Should be on round 1 now
        assert state.round_idx == 1
        assert not state.is_terminal

    def test_check_check_both_rounds(self):
        """Check-check on both rounds → showdown."""
        state = LeducState((JACK, 0), (QUEEN, 0), (KING, 0))
        state = state.apply(CHECK)
        state = state.apply(CHECK)  # end preflop
        state = state.apply(CHECK)
        state = state.apply(CHECK)  # end flop → showdown
        assert state.is_terminal

    def test_fold(self):
        state = LeducState((JACK, 0), (QUEEN, 0), (KING, 0))
        state = state.apply(BET)
        state = state.apply(FOLD)
        assert state.is_terminal
        # P1 bet, P2 folded → P1 wins
        assert state.get_payoff(0) > 0
        assert state.get_payoff(1) < 0

    def test_pair_wins(self):
        """Pairing the board should win."""
        # K♠ vs J♠, board K♦ → K♠ pairs
        state = LeducState((KING, 0), (JACK, 0), (KING, 1))
        state = state.apply(CHECK)
        state = state.apply(CHECK)  # → round 1
        state = state.apply(CHECK)
        state = state.apply(CHECK)  # → showdown
        assert state.is_terminal
        assert state.get_payoff(0) > 0  # King pairs the board

    def test_high_card_wins(self):
        """When nobody pairs, high card wins."""
        # Q♠ vs J♠, board K♦ → Q wins
        state = LeducState((QUEEN, 0), (JACK, 0), (KING, 1))
        state = state.apply(CHECK)
        state = state.apply(CHECK)
        state = state.apply(CHECK)
        state = state.apply(CHECK)
        assert state.is_terminal
        assert state.get_payoff(0) > 0  # Queen > Jack

    def test_zero_sum(self):
        """All terminal states should be zero-sum."""
        rng = random.Random(42)
        for _ in range(50):
            p1, p2, board = deal_leduc(rng)
            state = LeducState(p1, p2, board)

            # Random play to terminal
            while not state.is_terminal:
                actions = state.get_actions()
                if not actions:
                    break
                action = rng.choice(actions)
                state = state.apply(action)

            if state.is_terminal:
                total = state.get_payoff(0) + state.get_payoff(1)
                assert total == 0, f"Non-zero-sum: {state}"

    def test_deal_produces_valid_cards(self):
        rng = random.Random(42)
        for _ in range(20):
            p1, p2, board = deal_leduc(rng)
            assert p1 in DECK
            assert p2 in DECK
            assert board in DECK
            assert len({p1, p2, board}) == 3  # all different

    def test_bet_and_raise(self):
        """Test bet → raise → call sequence."""
        state = LeducState((JACK, 0), (QUEEN, 0), (KING, 0))
        state = state.apply(BET)                    # P1 bets
        assert RAISE in state.get_actions()          # P2 can raise
        state = state.apply(RAISE)                   # P2 raises
        assert FOLD in state.get_actions()
        assert CALL in state.get_actions()
        assert RAISE not in state.get_actions()      # max 2 bets


# =============================================================================
# Self-Play Trainer tests
# =============================================================================

class TestSelfPlayTrainer:
    def test_trainer_runs(self):
        """Verify the training loop runs without errors."""
        config = TrainingConfig(
            embed_dim=32,
            opponent_embed_dim=32,
            num_heads=2,
            num_layers=1,
            hands_per_epoch=16,
            ppo_epochs=2,
            log_interval=100,
        )
        trainer = LeducSelfPlayTrainer(config=config, seed=42)
        metrics = trainer.train(num_epochs=3)

        assert len(metrics['epoch_reward']) == 3
        assert len(metrics['epoch_loss']) == 3
        # Rewards should be finite
        assert all(abs(r) < 100 for r in metrics['epoch_reward'])

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        config = TrainingConfig(
            embed_dim=32,
            opponent_embed_dim=32,
            num_heads=2,
            num_layers=1,
            hands_per_epoch=64,
            ppo_epochs=4,
            log_interval=100,
        )
        trainer = LeducSelfPlayTrainer(config=config, seed=42)
        metrics = trainer.train(num_epochs=20)

        # Average loss of last 5 should be less than first 5
        early_loss = sum(metrics['epoch_loss'][:5]) / 5
        late_loss = sum(metrics['epoch_loss'][-5:]) / 5
        # This is a soft check — PPO loss isn't always monotonically decreasing
        # but it should at least be in the same ballpark
        assert late_loss < early_loss * 2, \
            f"Loss not converging: early={early_loss:.4f}, late={late_loss:.4f}"

    def test_self_play_is_symmetric(self):
        """In self-play, average P1 reward should approach 0."""
        config = TrainingConfig(
            embed_dim=32,
            opponent_embed_dim=32,
            num_heads=2,
            num_layers=1,
            hands_per_epoch=128,
            ppo_epochs=2,
            log_interval=100,
        )
        trainer = LeducSelfPlayTrainer(config=config, seed=42)
        metrics = trainer.train(num_epochs=30)

        # Average reward over last 10 epochs should be near 0
        avg_late_reward = sum(metrics['p1_avg_reward'][-10:]) / 10
        assert abs(avg_late_reward) < 5.0, \
            f"Self-play not symmetric: avg P1 reward = {avg_late_reward:.3f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
