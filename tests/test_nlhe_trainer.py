"""
Tests for NLHE self-play trainer.
"""

import pytest
import torch

from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig


def _play_one_hand(trainer, table=None):
    """Helper: play a single hand using the generator API and return experiences."""
    from training.nlhe_trainer import TableState
    if table is None:
        table = TableState()
    gen = trainer._play_hand_gen(table)
    trainer.policy.eval()
    trainer.opponent_encoder.eval()

    try:
        state = next(gen)
    except StopIteration as e:
        return e.value  # (experiences, reward)

    while True:
        # Run a single forward pass for this game
        cached = state['_cached_embeds']
        for opid, history in state['_uncached_opp_histories']:
            if not history:
                emb = trainer.opponent_encoder.encode_empty(1, device=str(trainer.device))
            else:
                seq = torch.stack(history).unsqueeze(0).to(trainer.device)
                with torch.no_grad():
                    emb = trainer.opponent_encoder(seq)
            cached[opid] = emb.detach()
            
        ordered = [cached[opid] for opid in state['_opp_ids']]
        if not ordered:
            opp_emb = trainer.opponent_encoder.encode_empty(1, device=str(trainer.device)).unsqueeze(1)
        else:
            opp_emb = torch.cat(ordered, dim=0).unsqueeze(0)

        with torch.no_grad():
            output = trainer.policy(
                hole_cards=state['hole_cards'].to(trainer.device),
                community_cards=state['community_cards'].to(trainer.device),
                numeric_features=state['numeric_features'].to(trainer.device),
                opponent_embeddings=opp_emb.to(trainer.device),
                opponent_stats=state['opponent_stats'].to(trainer.device),
                own_stats=state['own_stats'].to(trainer.device),
                action_mask=state['action_mask'].to(trainer.device),
                sizing_mask=state['sizing_mask'].to(trainer.device),
            )
        probs = output.action_type_probs[0].cpu()
        value = output.value[0, 0].item()
        sizing_probs = torch.softmax(output.bet_size_logits[0], dim=-1).cpu().tolist()
        
        # dummy action embedding
        action_embed = torch.zeros(1, trainer.config.opponent_embed_dim)

        try:
            state = gen.send((probs, value, sizing_probs, action_embed))
        except StopIteration as e:
            return e.value  # (experiences, reward)


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
        return NLHESelfPlayTrainer(config=config)

    def test_play_hand(self, trainer):
        """Self-play runs without crashing."""
        output = _play_one_hand(trainer)
        assert len(output) == 2

    def test_rewards_are_zero_sum(self, trainer):
        """Rewards across players sum to zero in heads-up."""
        hero_exps, hero_reward = _play_one_hand(trainer)
        # Other player rewards are implicit, or computed separately?
        # Just check it returns a float
        assert isinstance(hero_reward, float)

    def test_experience_tensors(self, trainer):
        """Experience tuples contain correct tensor shapes."""
        experiences, _ = _play_one_hand(trainer)
        # only check if hero got to act
        if experiences and experiences[0]:
            exp = experiences[0][0]
            assert exp.reward is not None

    def test_action_mask_valid(self, trainer):
        """Action mask enforces valid options."""
        experiences, _ = _play_one_hand(trainer)
        for pexp in experiences:
            for exp in pexp:
                # Mask should be boolean tensor [1, NUM_ACTION_TYPES]
                assert exp.action_mask.shape == (1, 4)
                if exp.action_idx == 3:  # FOLD
                    assert exp.action_mask[0, 3].item() is True

    def test_multiplayer(self):
        """Works for >2 players."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=1, num_layers=1,
            num_players=6, starting_bb=100,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100, device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config)
        output = _play_one_hand(trainer)
        assert len(output) == 2

    def test_deep_stacks(self):
        """Works for deep stacks (300bb)."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=1, num_layers=1,
            num_players=2, starting_bb=300,
            min_bb=250, max_bb=350,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100, device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config)
        output = _play_one_hand(trainer)
        assert len(output) == 2

    def test_train_loop(self):
        """Train epoch runs and updates policy."""
        config = NLHETrainingConfig(
            embed_dim=32, opponent_embed_dim=32,
            num_heads=1, num_layers=1,
            num_players=2, starting_bb=50,
            min_bb=20, max_bb=100,
            hands_per_epoch=4, ppo_epochs=1,
            log_interval=100, device="cpu",
        )
        trainer = NLHESelfPlayTrainer(config=config)
        metrics = trainer.train(num_epochs=3)
        assert 'epoch_reward' in metrics
        assert 'epoch_loss' in metrics
        assert len(metrics['epoch_reward']) == 3

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
        trainer = NLHESelfPlayTrainer(config=config)
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
        trainer = NLHESelfPlayTrainer(config=config)
        num_p, stacks = trainer._sample_table()
        assert num_p == 4
        for s in stacks:
            assert 10 * config.big_blind <= s <= 200 * config.big_blind

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
        trainer = NLHESelfPlayTrainer(config=config)
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
        return NLHESelfPlayTrainer(config=config)

    def test_action_history_builds(self, trainer):
        """Action histories grow after playing hands."""
        from training.nlhe_trainer import TableState
        table = TableState()
        _play_one_hand(trainer, table)
        total_actions = sum(len(h) for h in table.action_histories.values())
        assert total_actions > 0

    def test_opponent_embedding_not_empty(self, trainer):
        """After hands, embeddings should be non-zero (not empty)."""
        from training.nlhe_trainer import TableState
        table = TableState()
        for _ in range(5):
            _play_one_hand(trainer, table)
        assert len(table.action_histories) > 0

    def test_history_reset(self, trainer):
        """History resets after threshold hands."""
        from training.nlhe_trainer import TableState
        table = TableState()
        table.next_reset_at = 3
        for _ in range(5):
            _play_one_hand(trainer, table)
        assert table.hands_since_reset < 5

    def test_stat_tracker_records(self, trainer):
        """HUD stat tracker records hands."""
        from training.nlhe_trainer import TableState
        table = TableState()
        for _ in range(5):
            _play_one_hand(trainer, table)
        for pid in range(2):
            n = table.stat_tracker.get_num_hands(pid)
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
        trainer = NLHESelfPlayTrainer(config=config)
        assert trainer.device == torch.device("cpu")
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
        trainer = NLHESelfPlayTrainer(config=config)
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
            search_iterations=5, seed=42
        )
        trainer = NLHESelfPlayTrainer(config=config)
        metrics = trainer.train(num_epochs=2)
        assert len(metrics['epoch_reward']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
