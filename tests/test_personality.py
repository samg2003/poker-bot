"""
Tests for Phase 3: personality perturbations, NLHE encoding, and curriculum.
"""

import pytest
import random
import torch

from training.personality import (
    PersonalityModifier, SituationalPersonality, Situation,
    TiltState, detect_situations, sample_table_personalities,
)
from training.curriculum import (
    CurriculumTrainer, CurriculumConfig, CurriculumStage, TrainingMetrics,
)


# =============================================================================
# Personality Modifier Tests
# =============================================================================

class TestPersonalityModifier:
    def test_gto_is_all_ones(self):
        gto = PersonalityModifier.gto()
        assert gto.range_mult == 1.0
        assert gto.aggression_mult == 1.0
        assert gto.fold_pressure == 1.0

    def test_nit_is_tight(self):
        nit = PersonalityModifier.nit()
        assert nit.range_mult < 1.0
        assert nit.fold_pressure > 1.0
        assert nit.bluff_mult < 1.0

    def test_tag_is_tight_aggressive(self):
        tag = PersonalityModifier.tag()
        assert tag.range_mult < 1.0       # tight range
        assert tag.aggression_mult > 1.0  # aggressive
        assert tag.sizing_mult > 1.0      # bets big

    def test_lag_is_loose_aggressive(self):
        lag = PersonalityModifier.lag()
        assert lag.range_mult > 1.0
        assert lag.aggression_mult > 1.0

    def test_maniac_is_aggressive_but_realistic(self):
        maniac = PersonalityModifier.maniac()
        assert maniac.range_mult > 1.0
        assert maniac.aggression_mult > 1.0
        # Key: values are now realistic, not alien-like extremes
        assert maniac.aggression_mult <= 1.8
        assert maniac.range_mult <= 1.8

    def test_calling_station_is_passive(self):
        cs = PersonalityModifier.calling_station()
        assert cs.aggression_mult < 0.5
        assert cs.fold_pressure < 0.5

    def test_fish_is_loose_passive(self):
        fish = PersonalityModifier.fish()
        assert fish.range_mult > 1.0
        assert fish.aggression_mult < 1.0
        assert fish.cbet_mult < 0.5

    def test_random_in_range(self):
        rng = random.Random(42)
        for _ in range(20):
            mod = PersonalityModifier.random(rng)
            assert 0.4 <= mod.range_mult <= 1.7
            assert 0.3 <= mod.aggression_mult <= 1.7
            assert 0.3 <= mod.fold_pressure <= 1.6

    def test_blend(self):
        gto = PersonalityModifier.gto()
        nit = PersonalityModifier.nit()
        half = gto.blend(nit, 0.5)
        # Should be between GTO and Nit
        assert gto.range_mult > half.range_mult > nit.range_mult


# =============================================================================
# Situational Personality Tests
# =============================================================================

class TestSituationalPersonality:
    def test_base_modifier_when_no_overrides(self):
        base = PersonalityModifier.nit()
        sp = SituationalPersonality(base=base)
        result = sp.get_modifier([Situation.PREFLOP])
        assert result.range_mult == base.range_mult

    def test_override_blends_with_base(self):
        base = PersonalityModifier.nit()
        override = PersonalityModifier.lag()
        sp = SituationalPersonality(
            base=base,
            overrides={Situation.WET_BOARD: override}
        )
        result = sp.get_modifier([Situation.WET_BOARD])
        # Should be somewhere between nit and lag
        assert base.range_mult < result.range_mult

    def test_apply_preserves_distribution(self):
        sp = SituationalPersonality(base=PersonalityModifier.nit())
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = sp.apply(probs, [Situation.PREFLOP], hand_strength=0.3)
        assert abs(result.sum().item() - 1.0) < 1e-5
        assert (result >= 0).all()

    def test_nit_folds_more_with_weak_hands(self):
        gto = SituationalPersonality(base=PersonalityModifier.gto())
        nit = SituationalPersonality(base=PersonalityModifier.nit())
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

        gto_result = gto.apply(probs, [], hand_strength=0.2)
        nit_result = nit.apply(probs, [], hand_strength=0.2)

        # Nit should fold more with weak hand
        assert nit_result[0] > gto_result[0]

    def test_nit_folds_less_with_strong_hands(self):
        """Nit with a strong hand should not fold much more than GTO."""
        gto = SituationalPersonality(base=PersonalityModifier.gto())
        nit = SituationalPersonality(base=PersonalityModifier.nit())
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

        gto_result = gto.apply(probs, [], hand_strength=0.9)
        nit_result = nit.apply(probs, [], hand_strength=0.9)

        # With strong hand, nit fold prob should be close to GTO's
        assert abs(nit_result[0].item() - gto_result[0].item()) < 0.1

    def test_lag_raises_more(self):
        gto = SituationalPersonality(base=PersonalityModifier.gto())
        lag = SituationalPersonality(base=PersonalityModifier.lag())
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

        gto_result = gto.apply(probs, [], hand_strength=0.5)
        lag_result = lag.apply(probs, [], hand_strength=0.5)

        # LAG should raise more
        assert lag_result[3] > gto_result[3]

    def test_apply_with_facing_raise(self):
        """Nit facing a raise should fold significantly more."""
        nit = SituationalPersonality(base=PersonalityModifier.nit())
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])

        no_raise = nit.apply(probs, [], hand_strength=0.3, is_facing_raise=False)
        facing_raise = nit.apply(probs, [], hand_strength=0.3, is_facing_raise=True)

        # Nit facing a raise folds more
        assert facing_raise[0] > no_raise[0]

    def test_apply_no_negative_probs(self):
        """Even with extreme personalities, probs should never be negative."""
        maniac = SituationalPersonality(base=PersonalityModifier.maniac())
        probs = torch.tensor([0.1, 0.1, 0.7, 0.1])  # heavy call weight

        result = maniac.apply(probs, [], hand_strength=0.5)
        assert (result >= 0).all()
        assert abs(result.sum().item() - 1.0) < 1e-5

    def test_sizing_perturbation_shifts_distribution(self):
        """apply_sizing with sizing_mult > 1 should shift weight to larger bets."""
        maniac = SituationalPersonality(base=PersonalityModifier.maniac())
        # Uniform sizing probs (10 buckets: 0.1x to all-in)
        sizing = [0.1] * 10

        adjusted = maniac.apply_sizing(sizing, [])
        # Last bucket (all-in) should be weighted higher
        assert adjusted[-1] > adjusted[0]
        # Sum should still be ~1.0
        assert abs(sum(adjusted) - 1.0) < 1e-5

    def test_sizing_nit_prefers_small(self):
        """Nit (sizing_mult < 1) should prefer smaller bet sizes."""
        nit = SituationalPersonality(base=PersonalityModifier.nit())
        sizing = [0.1] * 10

        adjusted = nit.apply_sizing(sizing, [])
        # First bucket (small bet) should be weighted higher
        assert adjusted[0] > adjusted[-1]

    def test_sizing_gto_unchanged(self):
        """GTO (sizing_mult = 1.0) should not change sizing distribution."""
        gto = SituationalPersonality(base=PersonalityModifier.gto())
        sizing = [0.1] * 10

        adjusted = gto.apply_sizing(sizing, [])
        for orig, adj in zip(sizing, adjusted):
            assert abs(orig - adj) < 1e-5


# =============================================================================
# Situation Detection Tests
# =============================================================================

class TestSituationDetection:
    def test_street_detection(self):
        sits = detect_situations(street=0)
        assert Situation.PREFLOP in sits

        sits = detect_situations(street=3)
        assert Situation.RIVER in sits

    def test_position(self):
        sits = detect_situations(street=0, is_in_position=True)
        assert Situation.IN_POSITION in sits

        sits = detect_situations(street=0, is_in_position=False)
        assert Situation.OUT_OF_POSITION in sits

    def test_stack_depth(self):
        sits = detect_situations(street=0, stack_bb=15)
        assert Situation.SHORT_STACK in sits

        sits = detect_situations(street=0, stack_bb=150)
        assert Situation.DEEP_STACK in sits

    def test_facing_action(self):
        sits = detect_situations(street=0, is_facing_raise=True)
        assert Situation.FACING_RAISE in sits

    def test_board_texture(self):
        # Wet: 2h, 3h, 4h (all hearts + connected)
        wet_board = [0, 4, 8]  # same suit pattern
        sits = detect_situations(street=1, board_cards=wet_board)
        assert Situation.WET_BOARD in sits


# =============================================================================
# Tilt State Tests
# =============================================================================

class TestTiltState:
    def test_initial_not_tilting(self):
        tilt = TiltState()
        assert not tilt.is_tilting

    def test_tilt_after_consecutive_losses(self):
        tilt = TiltState()
        for _ in range(3):
            tilt.update(hand_result=-10, pot_size=20, big_blind=1)
        assert tilt.is_tilting

    def test_win_resets_streak(self):
        tilt = TiltState()
        tilt.update(-10, 20, 1)
        tilt.update(-10, 20, 1)
        tilt.update(5, 10, 1)  # win resets
        assert not tilt.is_tilting

    def test_tilt_modifier_is_aggressive(self):
        tilt = TiltState()
        for _ in range(4):
            tilt.update(-20, 40, 1)
        mod = tilt.get_tilt_modifier()
        assert mod.range_mult > 1.0
        assert mod.aggression_mult > 1.0
        assert mod.fold_pressure < 1.0


# =============================================================================
# Table Sampling Tests
# =============================================================================

class TestTableSampling:
    def test_output_count(self):
        players = sample_table_personalities(6, gto_fraction=0.5)
        assert len(players) == 6

    def test_gto_fraction(self):
        rng = random.Random(42)
        gto_count = 0
        total = 100
        for _ in range(total):
            players = sample_table_personalities(1, gto_fraction=0.6, rng=rng)
            if players[0].base.range_mult == 1.0:
                gto_count += 1
        # Should be roughly 60% GTO (with some variance)
        assert 40 < gto_count < 80


# =============================================================================
# Curriculum Trainer Tests
# =============================================================================

class TestCurriculumTrainer:
    def test_trainer_runs(self):
        """Curriculum trainer runs without errors."""
        config = CurriculumConfig(
            embed_dim=32,
            opponent_embed_dim=32,
            num_heads=2,
            num_layers=1,
            hands_per_epoch=16,
            ppo_epochs=2,
            log_interval=100,
            stages=[
                CurriculumStage("Test Stage", 2, 2, 20, 20, 1.0, min_epochs=5),
            ],
        )
        trainer = CurriculumTrainer(config=config, seed=42)
        metrics = trainer.train(max_epochs=3)
        assert len(metrics.epoch_rewards) == 3

    def test_metrics_tracking(self):
        metrics = TrainingMetrics()
        stage = CurriculumStage("Test", 2, 2, 20, 20, 1.0, min_epochs=3, plateau_window=2)

        # Not enough epochs
        metrics.epoch_rewards = [0.1, 0.2]
        assert not metrics.should_advance(stage)

        # Enough epochs but not plateau
        metrics.epoch_rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert not metrics.should_advance(stage)

        # Plateau (reward stabilized)
        metrics.epoch_rewards = [0.1, 0.2, 0.5, 0.5, 0.5, 0.5]
        assert metrics.should_advance(stage)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
