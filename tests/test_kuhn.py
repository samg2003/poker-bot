"""
Tests for Kuhn Poker game logic and CFR convergence to Nash equilibrium.

The known Nash equilibrium for Kuhn Poker has:
- Game value for P1: -1/18 ≈ -0.0556
- P1 with Jack: never bet (or bet as bluff with α ∈ [0, 1/3])
- P1 with Queen: check, call P2's bet with prob 1/3
- P1 with King: bet ~3α of the time
- P2 with Jack: if P1 checks, bet 1/3 of the time; if P1 bets, fold
- P2 with Queen: if P1 checks, check; if P1 bets, call
- P2 with King: always bet or call
"""

import pytest
from engine.kuhn_poker import KuhnState, JACK, QUEEN, KING, CHECK, BET, FOLD, CALL
from training.cfr import KuhnCFR


# =============================================================================
# Kuhn Poker game logic tests
# =============================================================================

class TestKuhnPoker:
    def test_initial_state(self):
        state = KuhnState((JACK, QUEEN))
        assert state.current_player == 0
        assert not state.is_terminal
        assert state.get_actions() == [CHECK, BET]

    def test_check_check_showdown(self):
        """Both check → showdown, higher card wins."""
        state = KuhnState((JACK, QUEEN), 'cc')
        assert state.is_terminal
        assert state.get_payoff(0) == -1  # Jack loses
        assert state.get_payoff(1) == 1   # Queen wins

    def test_king_beats_queen(self):
        state = KuhnState((KING, QUEEN), 'cc')
        assert state.get_payoff(0) == 1

    def test_bet_fold(self):
        """P1 bets, P2 folds → P1 wins ante."""
        state = KuhnState((JACK, KING), 'bf')
        assert state.is_terminal
        assert state.get_payoff(0) == 1   # P1 wins despite worse card
        assert state.get_payoff(1) == -1

    def test_bet_call_showdown(self):
        """P1 bets, P2 calls → showdown with bigger pot."""
        state = KuhnState((QUEEN, KING), 'bk')
        assert state.is_terminal
        assert state.get_payoff(0) == -2  # Queen loses 2
        assert state.get_payoff(1) == 2   # King wins 2

    def test_check_bet_fold(self):
        """P1 checks, P2 bets, P1 folds."""
        state = KuhnState((JACK, QUEEN), 'cbf')
        assert state.is_terminal
        assert state.get_payoff(0) == -1
        assert state.get_payoff(1) == 1

    def test_check_bet_call(self):
        """P1 checks, P2 bets, P1 calls → showdown."""
        state = KuhnState((KING, JACK), 'cbk')
        assert state.is_terminal
        assert state.get_payoff(0) == 2   # King wins big pot
        assert state.get_payoff(1) == -2

    def test_info_set_keys(self):
        """Information sets should include only own card + history."""
        s1 = KuhnState((JACK, QUEEN), '')
        assert s1.info_set_key() == 'J:'

        s2 = KuhnState((JACK, QUEEN), 'c')
        assert s2.info_set_key() == 'Q:c'  # P2's turn, sees Queen

        s3 = KuhnState((KING, JACK), 'cb')
        assert s3.info_set_key() == 'K:cb'  # P1's turn, sees King

    def test_all_terminal_states(self):
        """Verify all 5 terminal states are correctly identified."""
        terminals = ['cc', 'cbf', 'cbk', 'bf', 'bk']
        non_terminals = ['', 'c', 'b', 'cb']

        for h in terminals:
            assert KuhnState((0, 1), h).is_terminal, f"'{h}' should be terminal"
        for h in non_terminals:
            assert not KuhnState((0, 1), h).is_terminal, f"'{h}' should NOT be terminal"

    def test_payoffs_zero_sum(self):
        """All payoffs should be zero-sum."""
        from itertools import permutations
        cards = [JACK, QUEEN, KING]
        terminals = ['cc', 'cbf', 'cbk', 'bf', 'bk']

        for c1, c2 in permutations(cards, 2):
            for h in terminals:
                state = KuhnState((c1, c2), h)
                assert state.get_payoff(0) + state.get_payoff(1) == 0, \
                    f"Non-zero-sum: cards=({c1},{c2}), history='{h}'"


# =============================================================================
# CFR convergence tests
# =============================================================================

class TestCFRConvergence:
    """Test that CFR converges to the known Nash equilibrium."""

    @pytest.fixture(scope='class')
    def trained_cfr(self):
        """Train CFR once, reuse for all tests."""
        cfr = KuhnCFR()
        cfr.train(iterations=50000)
        return cfr

    def test_game_value(self, trained_cfr):
        """Game value should converge to -1/18 ≈ -0.0556."""
        value = trained_cfr.get_game_value()
        assert abs(value - (-1/18)) < 0.01, \
            f"Game value {value:.4f} not close to {-1/18:.4f}"

    def test_p1_jack_never_calls_bet(self, trained_cfr):
        """P1 with Jack facing P2's bet should always fold."""
        strategy = trained_cfr.get_final_strategy()
        # J:cb → P1 has Jack, checked, P2 bet, P1 decides
        actions, probs = strategy['J:cb']
        fold_idx = actions.index(FOLD)
        assert probs[fold_idx] > 0.95, \
            f"P1 should fold Jack vs bet, but fold prob = {probs[fold_idx]:.3f}"

    def test_p2_jack_never_calls(self, trained_cfr):
        """P2 with Jack facing P1's bet should always fold."""
        strategy = trained_cfr.get_final_strategy()
        # J:b → P2 has Jack, P1 bet, P2 decides
        actions, probs = strategy['J:b']
        fold_idx = actions.index(FOLD)
        assert probs[fold_idx] > 0.95, \
            f"P2 should fold Jack vs bet, but fold prob = {probs[fold_idx]:.3f}"

    def test_p2_king_always_calls_or_bets(self, trained_cfr):
        """P2 with King should always call bets and bet when checked to."""
        strategy = trained_cfr.get_final_strategy()

        # K:b → P2 has King, P1 bet → should call
        actions, probs = strategy['K:b']
        call_idx = actions.index(CALL)
        assert probs[call_idx] > 0.95, \
            f"P2 should call with King, but call prob = {probs[call_idx]:.3f}"

        # K:c → P2 has King, P1 checked → should bet
        actions, probs = strategy['K:c']
        bet_idx = actions.index(BET)
        assert probs[bet_idx] > 0.95, \
            f"P2 should bet with King, but bet prob = {probs[bet_idx]:.3f}"

    def test_p2_queen_checks_when_checked_to(self, trained_cfr):
        """P2 with Queen should mostly check when P1 checks."""
        strategy = trained_cfr.get_final_strategy()
        # Q:c → P2 has Queen, P1 checked
        actions, probs = strategy['Q:c']
        check_idx = actions.index(CHECK)
        assert probs[check_idx] > 0.90, \
            f"P2 should check Queen, but check prob = {probs[check_idx]:.3f}"

    def test_p2_jack_bluffs_one_third(self, trained_cfr):
        """P2 with Jack should bluff ~1/3 when P1 checks."""
        strategy = trained_cfr.get_final_strategy()
        # J:c → P2 has Jack, P1 checked → bet as bluff ~1/3
        actions, probs = strategy['J:c']
        bet_idx = actions.index(BET)
        assert abs(probs[bet_idx] - 1/3) < 0.05, \
            f"P2 should bluff Jack ~33%, but bet prob = {probs[bet_idx]:.3f}"

    def test_strategies_are_valid_distributions(self, trained_cfr):
        """All strategies should be valid probability distributions."""
        strategy = trained_cfr.get_final_strategy()
        for key, (actions, probs) in strategy.items():
            assert abs(sum(probs) - 1.0) < 1e-6, \
                f"{key}: probs sum to {sum(probs)}"
            assert all(p >= -1e-6 for p in probs), \
                f"{key}: negative probability {probs}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
