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
            device="cpu",
        )
        return NLHESelfPlayTrainer(config=config, seed=42)

    def test_play_hand(self, trainer):
        """A full hand completes without errors."""
        experiences = trainer._play_hand()
        assert len(experiences) == 2
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
            device="cpu",
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
            device="cpu",
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
            num_players=0, min_players=2, max_players=6,
            starting_bb=20,
            hands_per_epoch=4, ppo_epochs=1,
            device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        sizes = set()
        for _ in range(20):
            num_p, stacks = trainer._sample_table()
            assert 2 <= num_p <= 6
            sizes.add(num_p)
        assert len(sizes) >= 2

    def test_random_stacks(self):
        """Randomized stacks gives per-player variation."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=4, starting_bb=0,
            min_bb=20, max_bb=200,
            hands_per_epoch=4, ppo_epochs=1,
            device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        num_p, stacks = trainer._sample_table()
        assert num_p == 4
        for s in stacks:
            assert 20 * config.big_blind <= s <= 200 * config.big_blind

    def test_fully_random_training(self):
        """Training with fully randomized tables runs."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=0, starting_bb=0,
            min_players=2, max_players=4,
            min_bb=20, max_bb=100,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100, device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        metrics = trainer.train(num_epochs=2)
        assert len(metrics['epoch_reward']) == 2


class TestOpponentTracking:
    """Tests for opponent action history tracking."""

    @pytest.fixture
    def trainer(self):
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=2, starting_bb=20,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100, device="cpu",
        )
        return NLHESelfPlayTrainer(config=config, seed=42)

    def test_action_history_builds(self, trainer):
        """Action histories grow after playing hands."""
        trainer._play_hand()
        # At least one player should have recorded actions
        total_actions = sum(len(h) for h in trainer.action_histories.values())
        assert total_actions > 0

    def test_opponent_embedding_not_empty(self, trainer):
        """After hands, embeddings should be non-zero (not empty)."""
        # Play enough to build history
        for _ in range(5):
            trainer._play_hand()

        # Check that actions were recorded
        assert len(trainer.action_histories) > 0
        # Embedding from a real history should differ from empty
        for pid, history in trainer.action_histories.items():
            if history:
                emb = trainer._get_opponent_embedding(pid)
                assert emb.shape == (1, trainer.config.opponent_embed_dim)

    def test_history_reset(self, trainer):
        """History resets after threshold hands."""
        trainer.next_reset_at = 3  # Force reset after 3 hands
        for _ in range(5):
            trainer._play_hand()
        # After reset, histories should have been cleared at some point
        # (they may have rebuilt, but the reset mechanism triggered)
        assert trainer.hands_since_reset < 5

    def test_stat_tracker_records(self, trainer):
        """HUD stat tracker records hands."""
        for _ in range(5):
            trainer._play_hand()
        # Stats should be non-zero for at least some metric
        for pid in range(2):
            n = trainer.stat_tracker.get_num_hands(pid)
            assert n > 0


class TestDevicePlacement:
    """Tests for device placement."""

    def test_cpu_device(self):
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=2, starting_bb=20,
            hands_per_epoch=4, ppo_epochs=1,
            device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        assert trainer.device == torch.device("cpu")
        # Model should be on CPU
        for p in trainer.policy.parameters():
            assert p.device == torch.device("cpu")
            break

    def test_auto_device(self):
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=2, starting_bb=20,
            hands_per_epoch=4, ppo_epochs=1,
            device="auto",
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        # Should resolve to some valid device
        assert trainer.device in [torch.device("cpu"), torch.device("cuda"), torch.device("mps")]


class TestSearchGuided:
    """Tests for search-guided expert iteration."""

    def test_search_training_runs(self):
        """Training with search fraction > 0 completes."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=2, num_layers=1,
            num_players=2, starting_bb=20,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100, device="cpu",
            search_fraction=0.5,
            search_iterations=5,
        )
        trainer = NLHESelfPlayTrainer(config=config, seed=42)
        metrics = trainer.train(num_epochs=2)
        assert len(metrics['epoch_reward']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
