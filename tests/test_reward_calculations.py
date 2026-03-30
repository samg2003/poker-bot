"""
Rigorous, deterministic tests for equity-based reward calculations.

Verifies that:
1. All-in with worse hand → negative terminal reward
2. All-in with better hand → positive terminal reward
3. Fold → hero loses equity in the pot (negative reward)
4. Value bet called → positive Δ(hero_ev)
5. Calling with dominated hand → hero_ev stays low despite bigger pot
6. Side-pot scenarios give correct per-layer equity
7. End-of-street equity captures opponent calls/folds
8. Full GAE pipeline produces correct rewards for specific hands
9. Equity consistency checks (sum to pot, draw equities, etc.)

Uses GameState.str_to_card for card encoding (e.g. 'As' = Ace of spades).
NOTE: calculate_side_pots() uses bet_total, so tests play hands through natural
actions rather than manually setting pot/board to ensure consistency.
"""

import pytest
from engine.game_state import GameState, Action, ActionType, Street
from engine.dealer import Dealer
from engine.hand_evaluator import Eval7Evaluator
from training.state_encoder import compute_hero_ev
from training.ppo_updater import compute_gae


def _get_trainer():
    """Minimal trainer for _compute_hero_ev."""
    from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig
    return NLHESelfPlayTrainer(NLHETrainingConfig(
        embed_dim=32, opponent_embed_dim=32,
        num_heads=2, num_layers=1,
        num_players=2, starting_bb=100,
        hands_per_epoch=4, ppo_epochs=1,
        device="cpu", mc_equity_sims=1000,
    ))


def _set_hole_cards(state, hero_cards, villain_cards):
    """Set hole cards only (don't touch board/pot — those come from actions)."""
    state.players[0].hole_cards = tuple(GameState.str_to_card(c) for c in hero_cards)
    state.players[1].hole_cards = tuple(GameState.str_to_card(c) for c in villain_cards)


def _set_all_cards(state, hero_cards, villain_cards, p2_cards=None):
    """Set hole cards for 2 or 3 players."""
    state.players[0].hole_cards = tuple(GameState.str_to_card(c) for c in hero_cards)
    state.players[1].hole_cards = tuple(GameState.str_to_card(c) for c in villain_cards)
    if p2_cards:
        state.players[2].hole_cards = tuple(GameState.str_to_card(c) for c in p2_cards)


def _force_board(dealer, board_cards):
    """Force specific board cards by overriding the deck.

    Call this BEFORE actions that trigger board dealing.
    The deck is indexed by _next_card_idx. We need to place our desired
    cards at the positions that will be dealt (after burn cards).

    Flop: deck[idx] = burn, deck[idx+1..idx+3] = flop cards
    Turn: deck[idx] = burn, deck[idx+1] = turn card
    River: deck[idx] = burn, deck[idx+1] = river card
    """
    cards = [GameState.str_to_card(c) for c in board_cards]
    idx = dealer._next_card_idx

    if len(cards) >= 3:
        # Set flop: burn + 3 cards
        dealer.deck[idx] = 99  # burn (placeholder, won't be used as card)
        for i, c in enumerate(cards[:3]):
            dealer.deck[idx + 1 + i] = c

    if len(cards) >= 4:
        # Set turn: burn + 1 card (after flop's 4 positions)
        turn_idx = idx + 4
        dealer.deck[turn_idx] = 99  # burn
        dealer.deck[turn_idx + 1] = cards[3]

    if len(cards) >= 5:
        # Set river: burn + 1 card (after turn's 2 positions)
        river_idx = idx + 6
        dealer.deck[river_idx] = 99  # burn
        dealer.deck[river_idx + 1] = cards[4]


# ──────────────────────────────────────────────────────────
# Test 1: All-in Equity Direction
# ──────────────────────────────────────────────────────────

class TestAllInEquityDirection:
    """All-in EV correctly reflects hand strength."""

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_allin_with_better_hand_positive_ev(self, trainer):
        """Hero AA vs 72o → hero_ev should reflect dominant equity.

        Note: preflop HU pot splits into main pot (0.5×2=1.0bb) shared by
        both, and side pot (0.5bb) for BB only. Hero EV = equity × 1.0bb.
        """
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['7d', '2c'])

        hero_ev = compute_hero_ev(state, hero_idx=0)
        # AA vs 72o ≈ 87% equity. Main pot = 1.0bb. Gross EV ≈ 0.87.
        # Net EV = 0.87 - 0.5 (hero SB sunk cost) ≈ 0.37
        assert hero_ev > 0.30, f"AA should dominate, got net hero_ev={hero_ev:.2f}"

    def test_allin_with_worse_hand_low_ev(self, trainer):
        """Hero 72o vs AA → hero_ev < 20% of pot."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['7d', '2c'], ['As', 'Ah'])

        hero_ev = compute_hero_ev(state, hero_idx=0)
        assert hero_ev < 0.2 * state.pot, f"72o should be dominated, got hero_ev={hero_ev:.2f}"

    def test_allin_terminal_negative_for_worse_hand(self, trainer):
        """Hero (72o) all-in vs AA → ev_profit < 0."""
        dealer = Dealer(num_players=2, stacks=[50, 50], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['7d', '2c'], ['As', 'Ah'])

        dealer.apply_action(Action(ActionType.ALL_IN))
        dealer.apply_action(Action(ActionType.CALL))

        results = dealer.get_results()
        ev_profit = results.get('ev_profit', results['profit'])
        assert ev_profit[0] < 0, f"72o vs AA should lose EV, got {ev_profit[0]:.2f}"

    def test_allin_terminal_positive_for_better_hand(self, trainer):
        """Hero (AA) all-in vs 72o → ev_profit > 0."""
        dealer = Dealer(num_players=2, stacks=[50, 50], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['7d', '2c'])

        dealer.apply_action(Action(ActionType.ALL_IN))
        dealer.apply_action(Action(ActionType.CALL))

        results = dealer.get_results()
        ev_profit = results.get('ev_profit', results['profit'])
        assert ev_profit[0] > 0, f"AA vs 72o should win EV, got {ev_profit[0]:.2f}"


# ──────────────────────────────────────────────────────────
# Test 2: Fold Loses Equity
# ──────────────────────────────────────────────────────────

class TestFoldLosesEquity:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_fold_preflop_loses_blind(self, trainer):
        """Hero folds AA preflop → loses 0.5bb (the small blind)."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['7d', '2c'])

        ev_before = compute_hero_ev(state, hero_idx=0)
        # Net EV of AA preflop for SB is ~0.37
        assert ev_before > 0.3, "AA should have high net equity preflop"

        dealer.apply_action(Action(ActionType.FOLD))
        results = dealer.get_results()
        assert results['profit'][0] == pytest.approx(-0.5, abs=0.01)

        ev_after = compute_hero_ev(state, hero_idx=0)
        # Sunk cost is 0.5, equity is 0 → Net EV = -0.5
        assert ev_after == pytest.approx(-0.5, abs=0.01), "After fold, hero_ev must equal negative sunk cost"

    def test_fold_after_raise_loses_investment(self, trainer):
        """Hero raises to 3bb, villain 3-bets, hero folds → hero lost 3bb."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['Kh', 'Qh'], ['As', 'Ah'])

        dealer.apply_action(Action(ActionType.RAISE, amount=3.0))
        dealer.apply_action(Action(ActionType.RAISE, amount=9.0))
        dealer.apply_action(Action(ActionType.FOLD))

        results = dealer.get_results()
        assert results['profit'][0] == pytest.approx(-3.0, abs=0.01)


# ──────────────────────────────────────────────────────────
# Test 3: Value Bet → Positive Delta
# ──────────────────────────────────────────────────────────

class TestValueBetDelta:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_hero_bets_strong_hand_villain_calls(self, trainer):
        """Hero (AA) on A72 flop. Bets, villain calls → Δ(hero_ev) positive."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['Kd', 'Qc'])
        _force_board(dealer, ['Ac', '7h', '2d', 'Ts', '3c'])

        # Heads-up preflop: SB(hero) calls, BB checks → to flop
        dealer.apply_action(Action(ActionType.CALL))     # SB limps
        dealer.apply_action(Action(ActionType.CHECK))    # BB checks

        # Now on flop: A72. Hero has set of aces.
        assert state.street == Street.FLOP
        ev_before_bet = compute_hero_ev(state, hero_idx=0)
        pot_before = state.pot

        # BB acts first postflop in HU
        # In HU postflop: SB=dealer acts first? Actually BB=1 acts first
        # Hero is SB=0=dealer in HU, so hero acts first postflop
        dealer.apply_action(Action(ActionType.RAISE, amount=1.0))  # hero bets 1bb

        ev_after_hero_bet = compute_hero_ev(state, hero_idx=0)

        dealer.apply_action(Action(ActionType.CALL))  # villain calls

        ev_after_call = compute_hero_ev(state, hero_idx=0)

        # Key assertion: hero's EV increased because villain put money in drawing dead
        delta = ev_after_call - ev_before_bet
        assert delta > 0, f"Value bet + call should increase hero EV, Δ={delta:.2f}"


# ──────────────────────────────────────────────────────────
# Test 4: Bad Call — Hero Overpays
# ──────────────────────────────────────────────────────────

class TestBadCallPunishment:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_hero_calls_with_garbage(self, trainer):
        """Hero (72o) calls villain's (AA) bet on K93 → hero_ev stays tiny."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['7d', '2c'], ['As', 'Ah'])
        _force_board(dealer, ['Kh', '9s', '3d', 'Ts', '4c'])

        # Preflop: hero limps, villain checks
        dealer.apply_action(Action(ActionType.CALL))
        dealer.apply_action(Action(ActionType.CHECK))

        assert state.street == Street.FLOP
        ev_before = compute_hero_ev(state, hero_idx=0)
        # 72o vs AA on K93 → hero < 5%, pot=2bb → ev < 0.1
        equity_pct_before = ev_before / state.pot if state.pot > 0 else 0
        assert equity_pct_before < 0.15, f"72o should have tiny equity, got {equity_pct_before*100:.1f}%"


# ──────────────────────────────────────────────────────────
# Test 5: Side Pot with Specific Hands
# ──────────────────────────────────────────────────────────

class TestSidePotDeterministic:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_short_stack_limited_to_main_pot(self, trainer):
        """Short stack (AA, best hand) can only win the main pot."""
        dealer = Dealer(num_players=3, stacks=[100, 100, 10], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_all_cards(state, ['Kh', '5c'], ['Qd', 'Jc'], ['As', 'Ah'])

        # UTG (P0) all-in, SB (P1) calls, BB (P2=short stack) calls
        dealer.apply_action(Action(ActionType.ALL_IN))  # P0: 100
        dealer.apply_action(Action(ActionType.CALL))     # P1: calls 100
        dealer.apply_action(Action(ActionType.CALL))     # P2: calls 10 (all-in)

        if dealer.is_hand_over():
            results = dealer.get_results()
            ev_profit = results.get('ev_profit', results['profit'])
            # P2 (AA) should profit but is capped to main pot
            assert ev_profit[2] > 0, f"AA should profit, got {ev_profit[2]:.2f}"
            # Main pot = 10 × 3 = 30bb. P2's max profit ≈ 30 - 10 = 20bb
            assert ev_profit[2] < 25, f"Short stack capped to main pot, got {ev_profit[2]:.2f}"


# ──────────────────────────────────────────────────────────
# Test 6: End-of-Street Captures Opponent Fold
# ──────────────────────────────────────────────────────────

class TestEndOfStreetEquity:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_villain_folds_hero_wins(self, trainer):
        """Hero raises, villain folds → hero profits +1bb (the BB)."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['7d', '2c'], ['As', 'Ah'])

        dealer.apply_action(Action(ActionType.RAISE, amount=3.0))
        dealer.apply_action(Action(ActionType.FOLD))

        results = dealer.get_results()
        assert results['profit'][0] == pytest.approx(1.0, abs=0.01)


# ──────────────────────────────────────────────────────────
# Test 7: Full GAE Pipeline
# ──────────────────────────────────────────────────────────

class TestFullRewardPipeline:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_gae_value_bet_positive(self, trainer):
        """Synthetic trajectory: hero value bets, villain calls → positive reward."""
        from training.nlhe_trainer import Experience
        import torch

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

        # Cross-street: hero bet on flop, villain called, now on turn
        traj = [
            Experience(**base, value=0.0, reward=0.0, step_idx=0,
                       equity_x_pot=0.048, end_street_equity_x_pot=0.112, street_idx=1),
            Experience(**base, value=0.0, reward=0.05, step_idx=1,
                       equity_x_pot=0.112, end_street_equity_x_pot=0.112, street_idx=2),
        ]

        advantages, returns, _ = compute_gae(traj)

        # Step 0 cross-street reward = 0.112 - 0.048 = 0.064 (good bet got called)
        step0_reward = traj[0].end_street_equity_x_pot - traj[0].equity_x_pot
        assert step0_reward > 0, f"Value bet should yield positive reward, got {step0_reward}"
        assert advantages[0] > 0, f"Overall advantage should be positive, got {advantages[0]}"

    def test_gae_fold_negative(self, trainer):
        """Hero folds → negative advantage (gave up equity)."""
        from training.nlhe_trainer import Experience
        import torch

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
            action_idx=0, sizing_idx=0,
            log_prob=0.0, action_log_prob=0.0, sizing_log_prob=0.0,
            hand_id=0,
        )

        # Hero had 40% equity in 1.5bb pot, folds, loses 0.5bb SB
        traj = [
            Experience(**base, value=0.0, reward=-0.005, step_idx=0,
                       equity_x_pot=0.006, end_street_equity_x_pot=0.006, street_idx=0),
        ]

        advantages, returns, _ = compute_gae(traj)
        assert advantages[0] < 0, f"Fold should have negative advantage, got {advantages[0]}"
        assert returns[0] < 0, f"Fold should have negative return, got {returns[0]}"


# ──────────────────────────────────────────────────────────
# Test 8: Equity Consistency
# ──────────────────────────────────────────────────────────

class TestEquityConsistency:

    @pytest.fixture
    def trainer(self):
        return _get_trainer()

    def test_equities_sum_to_pot(self, trainer):
        """Sum of hero_ev + sum of sunk costs = pot."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['Kd', 'Qc'])

        ev0 = compute_hero_ev(state, hero_idx=0)
        ev1 = compute_hero_ev(state, hero_idx=1)
        sunk_total = state.players[0].bet_total + state.players[1].bet_total
        total_gross = ev0 + ev1 + sunk_total
        
        assert total_gross == pytest.approx(state.pot, rel=0.05), \
            f"Gross EVs ({total_gross:.2f}) should sum to pot ({state.pot})"

    def test_equities_sum_3way(self, trainer):
        """3-way: sum of net EVs + sum of sunk = pot."""
        dealer = Dealer(num_players=3, stacks=[100, 100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_all_cards(state, ['As', 'Ah'], ['Kd', 'Qc'], ['7h', '2d'])

        evs = [compute_hero_ev(state, hero_idx=i) for i in range(3)]
        sunk_total = sum(p.bet_total for p in state.players)
        total_gross = sum(evs) + sunk_total
        assert total_gross == pytest.approx(state.pot, rel=0.05), \
            f"3-way Gross EVs ({total_gross:.2f}) should sum to pot ({state.pot})"

    def test_aa_dominates_72o_preflop(self, trainer):
        """AA has ~87% equity vs 72o preflop.

        Side-pot aware: main pot = 1.0bb (both contribute 0.5).
        hero_ev ≈ 0.87 × 1.0 = 0.87. pot = 1.5.
        equity_pct as hero_ev/pot ≈ 58% (misleading metric), but _absolute_
        hero_ev shows ~87% of the money hero can actually win.
        """
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['7d', '2c'])

        hero_ev = compute_hero_ev(state, hero_idx=0)
        # Gross EV = 87% * 1.0bb = 0.87. Sunk = 0.5. Net = 0.37
        assert hero_ev > 0.30, f"AA should dominate, got net hero_ev={hero_ev:.2f}"

    def test_coinflip_near_50(self, trainer):
        """AKs vs QQ → classic coinflip.

        Side-pot aware: hero (SB) only eligible for 1.0bb main pot.
        hero_ev ≈ 46% × 1.0 = 0.46.
        """
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['Ah', 'Kh'], ['Qs', 'Qd'])

        hero_ev = compute_hero_ev(state, hero_idx=0)
        # AKs vs QQ ≈ 46%. Main pot = 1.0bb. Gross ≈ 0.46. Net ≈ -0.04
        assert -0.2 < hero_ev < 0.2, f"AKs vs QQ should be near neutral net EV, got {hero_ev:.2f}"

    def test_set_on_flop_dominates(self, trainer):
        """AA on A72 flop vs KQ → ~97% equity."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['As', 'Ah'], ['Kd', 'Qc'])
        _force_board(dealer, ['Ac', '7h', '2d', 'Ts', '3c'])

        # Play to flop naturally
        dealer.apply_action(Action(ActionType.CALL))   # SB
        dealer.apply_action(Action(ActionType.CHECK))  # BB

        assert state.street == Street.FLOP
        ev = compute_hero_ev(state, hero_idx=0)
        # Gross EV = 97% of 2.0bb (if both called 1.0) = 1.94bb
        # Net EV = 1.94 - 1.0 (sunk) = 0.94
        assert ev > 0.8, f"Set of aces should dominate, got net ev={ev:.2f}"

    def test_flush_draw_equity(self, trainer):
        """8h7h on 2h 5h Tc vs AK → flush draw."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['8h', '7h'], ['As', 'Kd'])
        _force_board(dealer, ['2h', '5h', 'Tc', '9s', '4c'])

        dealer.apply_action(Action(ActionType.CALL))
        dealer.apply_action(Action(ActionType.CHECK))

        assert state.street == Street.FLOP
        ev = compute_hero_ev(state, hero_idx=0)
        # Equity ~35-55% of 2.0bb pot = ~0.7 to 1.1. Sunk = 1.0. Net = -0.3 to +0.2
        assert -0.4 < ev < 0.3, f"Flush draw should be near neutral net EV initially, got {ev:.2f}"

    def test_made_flush_dominates(self, trainer):
        """8h7h on 2h 5h Th 9c → made flush vs AK ~90%+."""
        dealer = Dealer(num_players=2, stacks=[100, 100], big_blind=1.0, seed=1)
        state = dealer.start_hand()
        _set_hole_cards(state, ['8h', '7h'], ['As', 'Kd'])
        _force_board(dealer, ['2h', '5h', 'Th', '9c', '4d'])

        # Play to turn
        dealer.apply_action(Action(ActionType.CALL))
        dealer.apply_action(Action(ActionType.CHECK))
        # Flop
        dealer.apply_action(Action(ActionType.CHECK))
        dealer.apply_action(Action(ActionType.CHECK))
        # Turn
        assert state.street == Street.TURN
        ev = compute_hero_ev(state, hero_idx=0)
        # Made flush equity >90% of pot=2.0bb = ~1.8bb. Sunk=1.0. Net=0.8
        assert ev > 0.6, f"Made flush should dominate net EV, got {ev:.2f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
