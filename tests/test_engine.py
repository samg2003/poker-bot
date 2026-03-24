"""
Comprehensive tests for the poker engine.

Tests cover:
- Card representation
- Hand evaluation (all categories)
- Game state (legal actions, betting, folding)
- Dealer (full hand playthrough, side pots, showdown)
"""

import pytest
from engine.game_state import GameState, Player, Action, ActionType, Street
from engine.hand_evaluator import (
    HandEvaluator,
    HIGH_CARD, ONE_PAIR, TWO_PAIR, THREE_OF_A_KIND,
    STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH,
)
from engine.dealer import Dealer


# =============================================================================
# Card representation tests
# =============================================================================

class TestCardRepresentation:
    def test_card_to_str(self):
        assert GameState.card_to_str(0) == '2c'   # 0 = rank 0 (2), suit 0 (c)
        assert GameState.card_to_str(1) == '2d'
        assert GameState.card_to_str(4) == '3c'
        assert GameState.card_to_str(48) == 'Ac'
        assert GameState.card_to_str(51) == 'As'

    def test_str_to_card(self):
        assert GameState.str_to_card('2c') == 0
        assert GameState.str_to_card('As') == 51
        assert GameState.str_to_card('Th') == 34  # T = rank 8, h = suit 2

    def test_roundtrip(self):
        for i in range(52):
            assert GameState.str_to_card(GameState.card_to_str(i)) == i


# =============================================================================
# Hand evaluator tests
# =============================================================================

class TestHandEvaluator:
    """Test all hand categories are evaluated correctly."""

    def _cards(self, *names):
        return [GameState.str_to_card(n) for n in names]

    def test_high_card(self):
        cards = self._cards('Ac', 'Kd', '9h', '7s', '3c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == HIGH_CARD

    def test_one_pair(self):
        cards = self._cards('Ac', 'Ad', '9h', '7s', '3c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == ONE_PAIR

    def test_two_pair(self):
        cards = self._cards('Ac', 'Ad', '9h', '9s', '3c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == TWO_PAIR

    def test_three_of_a_kind(self):
        cards = self._cards('Ac', 'Ad', 'Ah', '9s', '3c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == THREE_OF_A_KIND

    def test_straight(self):
        cards = self._cards('9c', '8d', '7h', '6s', '5c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == STRAIGHT

    def test_ace_low_straight(self):
        cards = self._cards('Ac', '2d', '3h', '4s', '5c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == STRAIGHT

    def test_ace_high_straight(self):
        cards = self._cards('Ac', 'Kd', 'Qh', 'Js', 'Tc')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == STRAIGHT

    def test_flush(self):
        cards = self._cards('Ac', 'Kc', '9c', '7c', '3c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == FLUSH

    def test_full_house(self):
        cards = self._cards('Ac', 'Ad', 'Ah', '9s', '9c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == FULL_HOUSE

    def test_four_of_a_kind(self):
        cards = self._cards('Ac', 'Ad', 'Ah', 'As', '3c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == FOUR_OF_A_KIND

    def test_straight_flush(self):
        cards = self._cards('9c', '8c', '7c', '6c', '5c')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == STRAIGHT_FLUSH

    def test_royal_flush(self):
        cards = self._cards('Ac', 'Kc', 'Qc', 'Jc', 'Tc')
        rank = HandEvaluator.evaluate_5(cards)
        assert HandEvaluator.get_category(rank) == STRAIGHT_FLUSH

    def test_hand_ordering(self):
        """Verify hand categories are ranked correctly."""
        hands = [
            self._cards('7c', '5d', '3h', '2s', '9c'),   # high card
            self._cards('Ac', 'Ad', '9h', '7s', '3c'),    # one pair
            self._cards('Ac', 'Ad', '9h', '9s', '3c'),    # two pair
            self._cards('Ac', 'Ad', 'Ah', '9s', '3c'),    # trips
            self._cards('9c', '8d', '7h', '6s', '5c'),    # straight
            self._cards('Ac', 'Kc', '9c', '7c', '3c'),    # flush
            self._cards('Ac', 'Ad', 'Ah', '9s', '9c'),    # full house
            self._cards('Ac', 'Ad', 'Ah', 'As', '3c'),    # quads
            self._cards('9c', '8c', '7c', '6c', '5c'),    # straight flush
        ]
        ranks = [HandEvaluator.evaluate_5(h) for h in hands]
        for i in range(len(ranks) - 1):
            assert ranks[i] < ranks[i + 1], \
                f"{HandEvaluator.hand_to_str(ranks[i])} should rank below " \
                f"{HandEvaluator.hand_to_str(ranks[i + 1])}"

    def test_pair_kickers(self):
        """Higher kicker wins with same pair."""
        pair_K = self._cards('Ac', 'Ad', 'Kh', '7s', '3c')
        pair_Q = self._cards('Ac', 'Ad', 'Qh', '7s', '3c')
        assert HandEvaluator.evaluate_5(pair_K) > HandEvaluator.evaluate_5(pair_Q)

    def test_7_card_evaluation(self):
        """Best 5 from 7 cards."""
        cards = self._cards('Ac', 'Ad', 'Ah', 'Kc', 'Kd', '7s', '3c')
        rank = HandEvaluator.evaluate_7(cards)
        assert HandEvaluator.get_category(rank) == FULL_HOUSE

    def test_7_card_finds_flush(self):
        """Flush hidden in 7 cards."""
        cards = self._cards('Ac', 'Kc', '9c', '7c', '3c', 'Td', '2h')
        rank = HandEvaluator.evaluate_7(cards)
        assert HandEvaluator.get_category(rank) == FLUSH

    def test_ace_low_straight_in_7(self):
        """Ace-low straight in 7 cards."""
        cards = self._cards('Ac', '2d', '3h', '4s', '5c', 'Kd', 'Qh')
        rank = HandEvaluator.evaluate_7(cards)
        assert HandEvaluator.get_category(rank) == STRAIGHT


# =============================================================================
# Game State tests
# =============================================================================

class TestGameState:
    def test_init(self):
        gs = GameState(num_players=6, stacks=[100] * 6)
        assert gs.num_players == 6
        assert all(p.stack == 100 for p in gs.players)
        assert gs.street == Street.PREFLOP

    def test_invalid_players(self):
        with pytest.raises(AssertionError):
            GameState(num_players=1, stacks=[100])
        with pytest.raises(AssertionError):
            GameState(num_players=10, stacks=[100] * 10)

    def test_post_blinds_6_players(self):
        gs = GameState(num_players=6, stacks=[100] * 6, dealer_button=0)
        gs.post_blinds()
        # SB = seat 1, BB = seat 2
        assert gs.players[1].bet_this_street == 0.5
        assert gs.players[2].bet_this_street == 1.0
        assert gs.players[1].stack == 99.5
        assert gs.players[2].stack == 99.0
        assert gs.pot == 1.5
        # First to act = UTG = seat 3
        assert gs.current_player_idx == 3

    def test_post_blinds_heads_up(self):
        gs = GameState(num_players=2, stacks=[100, 100], dealer_button=0)
        gs.post_blinds()
        # HU: dealer is SB (seat 0), other is BB (seat 1)
        assert gs.players[0].bet_this_street == 0.5
        assert gs.players[1].bet_this_street == 1.0
        # HU: SB acts first preflop
        assert gs.current_player_idx == 0

    def test_legal_actions_facing_bet(self):
        gs = GameState(num_players=2, stacks=[100, 100])
        gs.post_blinds()
        # SB facing BB — can fold, call, raise, all-in
        actions = gs.get_legal_actions()
        assert ActionType.FOLD in actions
        assert ActionType.CALL in actions
        assert ActionType.RAISE in actions
        assert ActionType.CHECK not in actions

    def test_fold(self):
        gs = GameState(num_players=2, stacks=[100, 100])
        gs.post_blinds()
        gs.apply_action(Action(ActionType.FOLD))
        assert gs.is_hand_over
        assert gs.winners == [1]  # BB wins

    def test_call_and_check(self):
        gs = GameState(num_players=2, stacks=[100, 100])
        gs.post_blinds()
        # SB calls
        gs.apply_action(Action(ActionType.CALL))
        # BB can check (no additional bet)
        actions = gs.get_legal_actions()
        assert ActionType.CHECK in actions

    def test_raise(self):
        gs = GameState(num_players=2, stacks=[100, 100])
        gs.post_blinds()
        # SB raises to 3
        gs.apply_action(Action(ActionType.RAISE, amount=3.0))
        assert gs.current_bet == 3.0
        # BB can fold, call, raise, all-in
        actions = gs.get_legal_actions()
        assert ActionType.FOLD in actions
        assert ActionType.CALL in actions
        assert ActionType.RAISE in actions

    def test_all_in_short_stack(self):
        gs = GameState(num_players=2, stacks=[5, 100])
        gs.post_blinds()
        # SB (5 chips) goes all-in
        gs.apply_action(Action(ActionType.ALL_IN))
        assert gs.players[0].is_all_in
        assert gs.players[0].stack == 0


# =============================================================================
# Dealer integration tests
# =============================================================================

class TestDealer:
    def test_full_hand_fold_preflop(self):
        """Simple hand: SB folds preflop."""
        dealer = Dealer(num_players=2, stacks=[100, 100], seed=42)
        dealer.start_hand()

        # SB folds
        dealer.apply_action(Action(ActionType.FOLD))
        assert dealer.is_hand_over()

        results = dealer.get_results()
        assert 1 in results['winners']
        # BB wins the 1.5 pot (SB posted 0.5)
        assert results['profit'][1] == 0.5  # BB gains SB's 0.5
        assert results['profit'][0] == -0.5  # SB loses 0.5

    def test_full_hand_to_showdown(self):
        """Hand goes to showdown with checks on every street."""
        dealer = Dealer(num_players=2, stacks=[100, 100], seed=42)
        dealer.start_hand()

        state = dealer.get_state()

        # Preflop: SB calls, BB checks
        dealer.apply_action(Action(ActionType.CALL))
        dealer.apply_action(Action(ActionType.CHECK))

        assert state.street == Street.FLOP
        assert len(state.board) == 3

        # Flop: check, check
        dealer.apply_action(Action(ActionType.CHECK))
        dealer.apply_action(Action(ActionType.CHECK))

        assert state.street == Street.TURN
        assert len(state.board) == 4

        # Turn: check, check
        dealer.apply_action(Action(ActionType.CHECK))
        dealer.apply_action(Action(ActionType.CHECK))

        assert state.street == Street.RIVER
        assert len(state.board) == 5

        # River: check, check
        dealer.apply_action(Action(ActionType.CHECK))
        dealer.apply_action(Action(ActionType.CHECK))

        assert dealer.is_hand_over()
        results = dealer.get_results()
        assert len(results['winners']) >= 1
        # Total profit should be zero-sum
        assert abs(sum(results['profit'])) < 1e-9

    def test_side_pot_3_players(self):
        """Test side pot with 3 players, one short-stacked."""
        dealer = Dealer(
            num_players=3,
            stacks=[10, 50, 50],  # Player 0 is short
            big_blind=1.0,
            dealer_button=0,
            seed=42,
        )
        dealer.start_hand()
        state = dealer.get_state()

        # P0 is dealer, P1 is SB (0.5), P2 is BB (1.0)
        # UTG = P0 (wraps around) → goes all-in for 10
        dealer.apply_action(Action(ActionType.ALL_IN))
        # P1 calls 10
        dealer.apply_action(Action(ActionType.CALL))
        # P2 calls 10
        dealer.apply_action(Action(ActionType.CALL))

        # Check side pots
        pots = state.calculate_side_pots()
        assert len(pots) >= 1  # at least main pot

        # Main pot should be 30 (10 from each), eligible: all 3 in-hand players
        main_pot = pots[0]
        assert abs(main_pot[0] - 30.0) < 1e-9

        # If there's a side pot, it should be between P1 and P2 only
        # (but in this case they both put in 10, so no side pot)

    def test_zero_sum(self):
        """Verify all hands are zero-sum."""
        for seed in range(20):
            dealer = Dealer(num_players=4, stacks=[100] * 4, seed=seed)
            dealer.start_hand()
            state = dealer.get_state()

            # Play a simple hand: everyone calls preflop, then checks to showdown
            # First, get through preflop
            while state.street == Street.PREFLOP and not state.is_hand_over:
                actions = state.get_legal_actions()
                if ActionType.CALL in actions:
                    dealer.apply_action(Action(ActionType.CALL))
                elif ActionType.CHECK in actions:
                    dealer.apply_action(Action(ActionType.CHECK))
                else:
                    dealer.apply_action(Action(ActionType.FOLD))

            # Check through remaining streets
            while not dealer.is_hand_over():
                dealer.apply_action(Action(ActionType.CHECK))

            results = dealer.get_results()
            total_profit = sum(results['profit'])
            assert abs(total_profit) < 1e-9, \
                f"Seed {seed}: Non-zero-sum profit: {results['profit']}"

    def test_button_advance(self):
        dealer = Dealer(num_players=3, stacks=[100] * 3, dealer_button=0)
        dealer.start_hand()
        dealer.apply_action(Action(ActionType.FOLD))
        dealer.apply_action(Action(ActionType.FOLD))
        dealer.advance_button()
        assert dealer.dealer_button == 1

    def test_9_player_table(self):
        """Verify 9-player table works."""
        dealer = Dealer(num_players=9, stacks=[100] * 9, seed=42)
        dealer.start_hand()
        state = dealer.get_state()

        # Fold until hand is over — everyone except BB folds
        while not dealer.is_hand_over():
            dealer.apply_action(Action(ActionType.FOLD))

        assert dealer.is_hand_over()

    def test_variable_stacks(self):
        """Test with different stack sizes (1-350 BB)."""
        stacks = [1, 5, 20, 100, 200, 350]
        dealer = Dealer(num_players=6, stacks=stacks, seed=42)
        dealer.start_hand()
        state = dealer.get_state()

        # Should initialize correctly
        for i, s in enumerate(stacks):
            # Stacks should be original minus any blinds posted
            assert state.players[i].stack <= s


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
