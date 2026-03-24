"""
Kuhn Poker — a minimal poker game with a known Nash equilibrium.

Rules:
- 3 cards: J(0), Q(1), K(2)
- 2 players, each antes 1 chip
- Each dealt 1 card
- Player 1 acts first: CHECK or BET(1)
- If CHECK: Player 2 can CHECK (showdown) or BET(1)
  - If P2 BETs: P1 can FOLD or CALL(1)
- If BET: Player 2 can FOLD or CALL(1)
- Higher card wins at showdown

Known Nash equilibrium value: -1/18 ≈ -0.0556 for Player 1.

This is used to validate our CFR implementation converges correctly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# Cards
JACK, QUEEN, KING = 0, 1, 2
CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}

# Actions
CHECK, BET, FOLD, CALL = 'c', 'b', 'f', 'k'


class KuhnState:
    """
    A state in Kuhn Poker.

    History is a string of actions, e.g., 'cb' means P1 checked, P2 bet.
    """

    def __init__(self, cards: Tuple[int, int], history: str = ''):
        self.cards = cards    # (p1_card, p2_card)
        self.history = history

    @property
    def current_player(self) -> int:
        """0 = Player 1, 1 = Player 2."""
        if len(self.history) == 0:
            return 0
        if len(self.history) == 1:
            return 1
        # history length 2: only possible if P1 checked then P2 bet → P1's turn
        return 0

    @property
    def is_terminal(self) -> bool:
        """Is this a terminal state?"""
        h = self.history
        if h in ('bc', 'bp'):  # P1 bet, P2 folded/called → oops wrong chars
            pass
        # Terminal states:
        # 'cc'  → both checked, showdown
        # 'cbf' → P1 check, P2 bet, P1 fold
        # 'cbk' → P1 check, P2 bet, P1 call → showdown
        # 'bf'  → P1 bet, P2 fold
        # 'bk'  → P1 bet, P2 call → showdown
        return h in ('cc', 'cbf', 'cbk', 'bf', 'bk')

    def get_payoff(self, player: int) -> float:
        """
        Get payoff for the given player. Only valid at terminal states.
        Payoffs are in terms of chips won/lost (ante is 1 each).
        """
        assert self.is_terminal
        h = self.history
        winner_card = max(self.cards[0], self.cards[1])

        if h == 'cc':
            # Both checked, showdown. Pot = 2 (1 ante each).
            if self.cards[0] > self.cards[1]:
                return 1 if player == 0 else -1
            else:
                return -1 if player == 0 else 1

        elif h == 'cbf':
            # P1 checked, P2 bet, P1 folded. P2 wins pot of 2 (1 ante + nothing).
            # Wait: P2 bet 1 into pot of 2, P1 folded. P2 wins P1's ante (1).
            return -1 if player == 0 else 1

        elif h == 'cbk':
            # P1 checked, P2 bet, P1 called. Showdown. Pot = 4.
            if self.cards[0] > self.cards[1]:
                return 2 if player == 0 else -2
            else:
                return -2 if player == 0 else 2

        elif h == 'bf':
            # P1 bet, P2 folded. P1 wins P2's ante (1).
            return 1 if player == 0 else -1

        elif h == 'bk':
            # P1 bet, P2 called. Showdown. Pot = 4.
            if self.cards[0] > self.cards[1]:
                return 2 if player == 0 else -2
            else:
                return -2 if player == 0 else 2

        raise ValueError(f"Unknown terminal history: {h}")

    def get_actions(self) -> List[str]:
        """Legal actions at this state."""
        h = self.history
        if h == '':
            return [CHECK, BET]      # P1: check or bet
        elif h == 'c':
            return [CHECK, BET]      # P2: check or bet
        elif h == 'cb':
            return [FOLD, CALL]      # P1: fold or call P2's bet
        elif h == 'b':
            return [FOLD, CALL]      # P2: fold or call P1's bet
        return []

    def apply(self, action: str) -> 'KuhnState':
        """Return new state after applying action."""
        return KuhnState(self.cards, self.history + action)

    def info_set_key(self) -> str:
        """
        Information set key — what the current player can see.
        This is their card + the action history.
        """
        card = self.cards[self.current_player]
        return f"{CARD_NAMES[card]}:{self.history}"

    def __repr__(self) -> str:
        return (f"KuhnState(cards=({CARD_NAMES[self.cards[0]]},"
                f"{CARD_NAMES[self.cards[1]]}), history='{self.history}')")
