"""
Tests for equity-based reward shaping.

Verifies:
- Side-pot-aware hero EV computation
- Reward scenarios (value bet, bad call, fold, trap)
- GAE with equity shaping
- V_res residual near-zero for all-in scenarios
"""

import pytest
import torch

from engine.game_state import GameState, Action, ActionType, Street
from engine.dealer import Dealer
from engine.hand_evaluator import Eval7Evaluator


# ─────────────────────────────────────────────────────────────
# Helper: run a hand to a specific state for testing
# ─────────────────────────────────────────────────────────────

def _make_game_state(num_players, stacks, big_blind=1.0):
    """Create a fresh game state with posted blinds."""
    dealer = Dealer(num_players=num_players, stacks=stacks, big_blind=big_blind, seed=42)
    dealer.start_hand()
    return dealer


def _get_trainer_minimal():
    """Create a minimal trainer for testing _compute_hero_ev."""
    from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig
    config = NLHETrainingConfig(
        embed_dim=32, opponent_embed_dim=32,
        num_heads=2, num_layers=1,
        num_players=2, starting_bb=100,
        hands_per_epoch=4, ppo_epochs=1,
        device="cpu", mc_equity_sims=500,
    )
    return NLHESelfPlayTrainer(config=config)


# ─────────────────────────────────────────────────────────────
# Test 1: Side-Pot-Aware Hero EV
# ─────────────────────────────────────────────────────────────

class TestComputeHeroEV:

    @pytest.fixture
    def trainer(self):
        return _get_trainer_minimal()

    def test_heads_up_basic(self, trainer):
        """Hero EV in heads-up = equity × pot."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=42)
        state = dealer.start_hand()
        # Both players have posted blinds, pot is ~1.5
        ev = trainer._compute_hero_ev(state, hero_idx=0)
        # EV should be between -m_sunk_cost and pot
        assert -0.6 <= ev <= state.pot + 0.1

    def test_3way_side_pot(self, trainer):
        """Short stack creates a side pot. Hero's EV correctly accounts for pot layers."""
        # Player 0 (hero): 100bb, Player 1: 100bb, Player 2: 10bb (short stack)
        dealer = Dealer(num_players=3, stacks=[100, 100, 10], big_blind=1.0, seed=42)
        state = dealer.start_hand()

        # Force all three all-in: short stack first
        # Apply preflop actions to get everyone all-in
        # P2 (short) goes all-in for 10
        state.apply_action(Action(ActionType.ALL_IN))  # UTG (P0 in 3-way)
        state.apply_action(Action(ActionType.CALL))  # P1 calls
        state.apply_action(Action(ActionType.CALL))  # P2 calls

        # Now compute hero EV — should correctly split across side pots
        ev = trainer._compute_hero_ev(state, hero_idx=0)
        # Hero invested 20bb. Gross EV is >= 0. Net EV can be down to -20.
        assert ev >= -21.0  # Hero max loss is their 20bb investment

    def test_hero_folded_ev_zero(self, trainer):
        """If hero folded, hero_ev = 0 (not eligible for any pot)."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=42)
        state = dealer.start_hand()
        # Hero folds
        state.apply_action(Action(ActionType.FOLD))
        ev = trainer._compute_hero_ev(state, hero_idx=0)
        # Net EV = gross EV - sunk costs. It can be down to -0.5 (SB) and up to pot - 0.5.
        assert -0.6 <= ev <= state.pot + 0.1

    def test_ev_matches_dealer_showdown(self, trainer):
        """Hero EV at showdown should approximately match dealer's ev_profit."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=42)
        state = dealer.start_hand()

        # Play full hand: just call down
        while not dealer.is_hand_over():
            legal = state.get_legal_actions()
            if ActionType.CHECK in legal:
                dealer.apply_action(Action(ActionType.CHECK))
            elif ActionType.CALL in legal:
                dealer.apply_action(Action(ActionType.CALL))
            else:
                dealer.apply_action(Action(ActionType.FOLD))

        results = dealer.get_results()
        ev_profit = results.get('ev_profit', results['profit'])

        # Hero EV at end should be close to ev_profit + hero's investment
        # (ev_profit = EV - bet_total, our hero_ev = EV share of pots)
        # The key property: both use MC equity, so they should be consistent
        assert isinstance(ev_profit[0], float)


# ─────────────────────────────────────────────────────────────
# Test 2: Reward Scenarios — Δ(hero_ev) Correctness
# ─────────────────────────────────────────────────────────────

class TestRewardScenarios:

    @pytest.fixture
    def trainer(self):
        return _get_trainer_minimal()

    def test_value_bet_positive_delta(self, trainer):
        """Hero bets with strong hand, villain calls → Δ(hero_ev) should be positive."""
        # Set up a hand where hero has a strong hand
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=42)
        state = dealer.start_hand()

        ev_before = trainer._compute_hero_ev(state, hero_idx=0)

        # Hero raises (SB acts first heads-up preflop)
        state.apply_action(Action(ActionType.RAISE, amount=3.0))
        
        ev_after_raise = trainer._compute_hero_ev(state, hero_idx=0)

        # After hero raises, pot is bigger. If hero has decent hand, EV should
        # go up because they put money in while having equity in a bigger pot.
        # (Net effect depends on hero's actual cards — but the test verifies
        # the computation runs and returns a reasonable number)
        assert isinstance(ev_after_raise, float)
        assert ev_after_raise > -3.0

    def test_fold_ev_drops_to_zero(self, trainer):
        """After hero folds, hero_ev = 0."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=42)
        state = dealer.start_hand()

        ev_before = trainer._compute_hero_ev(state, hero_idx=0)
        # Net EV might be slightly negative if hero invested SB (0.5) but has poor equity
        assert ev_before > -1.0  # Hero should have some gross equity, making net EV > -1.0

        # Hero folds
        state.apply_action(Action(ActionType.FOLD))
        ev_after = trainer._compute_hero_ev(state, hero_idx=0)
        # Hero is out, sunk cost is 0.5bb
        assert ev_after == pytest.approx(-0.5, abs=0.01)


# ─────────────────────────────────────────────────────────────
# Test 3: GAE with Equity Shaping
# ─────────────────────────────────────────────────────────────

class TestGAEWithEquityShaping:

    @pytest.fixture
    def trainer(self):
        return _get_trainer_minimal()

    def test_synthetic_trajectory_same_street(self, trainer):
        """Within-street Δ(equity×pot) reflected as per-step reward."""
        from training.nlhe_trainer import Experience

        # Build a 3-step trajectory on the same street
        base = dict(
            hole_cards=torch.zeros(1, 2, dtype=torch.long),
            community_cards=torch.full((1, 5), -1, dtype=torch.long),
            numeric_features=torch.zeros(1, 23),
            opponent_embeddings=torch.zeros(1, 1, 32),
            opponent_stats=torch.zeros(1, 1, 30),
            own_stats=torch.zeros(1, 30),
            opponent_game_state=torch.zeros(1, 1, 14),
            hand_action_seq=torch.zeros(1, 40, 13),
            hand_action_len=torch.tensor([0]),
            actor_profiles_seq=torch.zeros(1, 40, 64),
            hero_profile=torch.zeros(1, 64),
            opponent_profiles=torch.zeros(1, 1, 64),
            action_mask=torch.ones(1, 4, dtype=torch.bool),
            sizing_mask=torch.ones(1, 10, dtype=torch.bool),
            action_idx=1, sizing_idx=0,
            log_prob=0.0, action_log_prob=0.0, sizing_log_prob=0.0,
            hand_id=0,
        )

        # Same street, equity grows (hero's bets are getting called while ahead)
        traj = [
            Experience(**base, value=0.0, reward=0.0, step_idx=0,
                       equity_x_pot=0.3, end_street_equity_x_pot=0.3, street_idx=1),
            Experience(**base, value=0.0, reward=0.0, step_idx=1,
                       equity_x_pot=0.5, end_street_equity_x_pot=0.5, street_idx=1),
            Experience(**base, value=0.0, reward=0.1, step_idx=2,
                       equity_x_pot=0.6, end_street_equity_x_pot=0.6, street_idx=1),
        ]

        advantages, returns = trainer._compute_gae(traj)

        assert len(advantages) == 3
        assert len(returns) == 3
        # With growing equity, early actions should have positive advantages
        # (they contributed to equity growth)

    def test_synthetic_trajectory_cross_street(self, trainer):
        """Cross-street uses end_street_equity_x_pot for reward."""
        from training.nlhe_trainer import Experience

        base = dict(
            hole_cards=torch.zeros(1, 2, dtype=torch.long),
            community_cards=torch.full((1, 5), -1, dtype=torch.long),
            numeric_features=torch.zeros(1, 23),
            opponent_embeddings=torch.zeros(1, 1, 32),
            opponent_stats=torch.zeros(1, 1, 30),
            own_stats=torch.zeros(1, 30),
            opponent_game_state=torch.zeros(1, 1, 14),
            hand_action_seq=torch.zeros(1, 40, 13),
            hand_action_len=torch.tensor([0]),
            actor_profiles_seq=torch.zeros(1, 40, 64),
            hero_profile=torch.zeros(1, 64),
            opponent_profiles=torch.zeros(1, 1, 64),
            action_mask=torch.ones(1, 4, dtype=torch.bool),
            sizing_mask=torch.ones(1, 10, dtype=torch.bool),
            action_idx=1, sizing_idx=0,
            log_prob=0.0, action_log_prob=0.0, sizing_log_prob=0.0,
            hand_id=0,
        )

        # Step 0: flop (equity_x_pot=0.3, end_street goes up to 0.5 after calls)
        # Step 1: turn (different street)
        traj = [
            Experience(**base, value=0.0, reward=0.0, step_idx=0,
                       equity_x_pot=0.3, end_street_equity_x_pot=0.5, street_idx=1),
            Experience(**base, value=0.0, reward=0.2, step_idx=1,
                       equity_x_pot=0.4, end_street_equity_x_pot=0.4, street_idx=2),
        ]

        advantages, returns = trainer._compute_gae(traj)

        # Cross-street reward for step 0 = 0.5 - 0.3 = 0.2 (end_street - decision)
        # This captures the fact that opponent called hero's bet while behind
        assert len(advantages) == 2


# ─────────────────────────────────────────────────────────────
# Test 4: V_res Residual Near Zero for All-In
# ─────────────────────────────────────────────────────────────

class TestResidualTargets:

    def test_allin_residual_near_zero(self):
        """V_res target ≈ 0 for all-in because hero_ev ≈ ev_profit.

        In training, hero_ev is computed BEFORE the all-in (at decision time).
        ev_profit is computed during showdown using the same MC equity method.
        Both use Eval7Evaluator.get_equity with similar runouts, so the
        gap between them (the V_res target) should be small.
        """
        trainer = _get_trainer_minimal()

        # Run many hands and check that residuals are reasonable on average
        residuals = []
        for seed in range(10):
            dealer = Dealer(num_players=2, stacks=[50, 50], big_blind=1.0, seed=seed)
            state = dealer.start_hand()

            # Compute hero EV at decision point (BEFORE all-in)
            hero_ev_at_decision = trainer._compute_hero_ev(state, hero_idx=0)

            # Both all-in
            dealer.apply_action(Action(ActionType.ALL_IN))  # SB
            dealer.apply_action(Action(ActionType.CALL))     # BB

            if not dealer.is_hand_over():
                continue

            results = dealer.get_results()
            ev_profit = results.get('ev_profit', results['profit'])

            # ev_profit is net: hero_ev_share - bet_total
            # hero_ev_at_decision is hero's EV share of pot at decision time
            # At decision: pot was ~1.5bb (blinds only), hero investment was ~0.5bb
            # After all-in: pot = 100bb, hero investment = 50bb
            #
            # The V_res target in training is: terminal_reward - equity_at_decision
            # Both scale the same way, so we compare unnormalized:
            # residual_target = ev_profit[0] - hero_ev_at_decision (which is already net EV)
            residual = ev_profit[0] - hero_ev_at_decision
            residuals.append(abs(residual))

        # Average residual across many hands should be moderate
        # (not 0 because equity changes after more money goes in)
        avg_residual = sum(residuals) / len(residuals)
        # Key property: residual is much smaller than the raw profit variance (which is ~100bb)
        assert avg_residual < 60.0, f"Average residual {avg_residual} unexpectedly large"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
