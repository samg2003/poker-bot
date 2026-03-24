"""
Tests for NLHE self-play trainer.
"""

import pytest
import torch

from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig


class TestNLHESelfPlayTrainer:
    @pytest.fixture
    def trainer(self):
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=2, starting_bb=20,
            hands_per_epoch=8,
            ppo_epochs=1,
            log_interval=100,  # suppress logging during tests
        )
        return NLHESelfPlayTrainer(config=config, seed=42)

    def test_play_hand(self, trainer):
        """A full hand completes without errors."""
        experiences = trainer._play_hand()
        assert len(experiences) == trainer.config.num_players
        # At least one player should have experience
        total_exp = sum(len(pexp) for pexp in experiences)
        assert total_exp > 0

    def test_rewards_are_zero_sum(self, trainer):
        """In heads-up, rewards should sum to zero (minus rake, which is 0)."""
        experiences = trainer._play_hand()
        rewards = []
        for pexp in experiences:
            if pexp:
                rewards.append(pexp[0].reward)
        if len(rewards) == 2:
            assert abs(rewards[0] + rewards[1]) < 1e-6

    def test_train_runs(self, trainer):
        """Training loop completes without errors."""
        metrics = trainer.train(num_epochs=3)
        assert 'epoch_reward' in metrics
        assert 'epoch_loss' in metrics
        assert len(metrics['epoch_reward']) == 3

    def test_experience_tensors(self, trainer):
        """Experience contains correctly shaped tensors."""
        experiences = trainer._play_hand()
        for pexp in experiences:
            for exp in pexp:
                assert exp.hole_cards.shape == (1, 2)
                assert exp.community_cards.shape == (1, 5)
                assert exp.numeric_features.shape == (1, 9)
                assert exp.action_mask.shape == (1, 4)
                assert 0 <= exp.action_idx < 4

    def test_action_mask_valid(self, trainer):
        """Action mask always has at least one legal action."""
        experiences = trainer._play_hand()
        for pexp in experiences:
            for exp in pexp:
                assert exp.action_mask.sum() > 0

    def test_multiplayer(self):
        """Works with 3+ players."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=3, starting_bb=20,
            hands_per_epoch=4,
            ppo_epochs=1,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        experiences = trainer._play_hand()
        assert len(experiences) == 3

    def test_deep_stacks(self):
        """Works with deep stacks."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=2, starting_bb=200,
            hands_per_epoch=4,
            ppo_epochs=1,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        experiences = trainer._play_hand()
        total = sum(len(p) for p in experiences)
        assert total > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
