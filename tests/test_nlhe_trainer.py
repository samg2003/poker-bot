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
            log_interval=100,
        )
        return NLHESelfPlayTrainer(config=config, seed=42)

    def test_play_hand(self, trainer):
        """A full hand completes without errors."""
        experiences = trainer._play_hand()
        assert len(experiences) == 2  # fixed 2 players
        total_exp = sum(len(pexp) for pexp in experiences)
        assert total_exp > 0

    def test_rewards_are_zero_sum(self, trainer):
        """In heads-up, rewards should sum to zero."""
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
            hands_per_epoch=4, ppo_epochs=1,
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
            hands_per_epoch=4, ppo_epochs=1,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        experiences = trainer._play_hand()
        total = sum(len(p) for p in experiences)
        assert total > 0

    def test_random_players(self):
        """Randomized player count produces varying table sizes."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=0,  # random
            min_players=2, max_players=6,
            starting_bb=20,  # fixed stacks
            hands_per_epoch=4, ppo_epochs=1,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        sizes = set()
        for _ in range(20):
            num_p, stacks = trainer._sample_table()
            assert 2 <= num_p <= 6
            assert len(stacks) == num_p
            sizes.add(num_p)
        # With 20 samples from 2-6, should see at least 2 different sizes
        assert len(sizes) >= 2

    def test_random_stacks(self):
        """Randomized stacks gives per-player variation."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=4,  # fixed
            starting_bb=0,  # random stacks
            min_bb=20, max_bb=200,
            hands_per_epoch=4, ppo_epochs=1,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        num_p, stacks = trainer._sample_table()
        assert num_p == 4
        assert len(stacks) == 4
        # Each stack should be in range [20, 200] * big_blind
        for s in stacks:
            assert 20 * config.big_blind <= s <= 200 * config.big_blind

    def test_fully_random_training(self):
        """Training with fully randomized tables (players + stacks) runs."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=0, starting_bb=0,  # both random
            min_players=2, max_players=4,
            min_bb=20, max_bb=100,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        metrics = trainer.train(num_epochs=2)
        assert len(metrics['epoch_reward']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
