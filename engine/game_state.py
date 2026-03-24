"""
Core game state for No-Limit Texas Hold'em.

Supports 2-9 players, arbitrary stack depths (1-350 BB),
side pots, and all standard NLHE actions.

GameState objects are designed to be cheaply copyable for search/rollouts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


class Street(Enum):
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()


class ActionType(Enum):
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()   # includes bet (raise when no prior bet)
    ALL_IN = auto()  # treated as raise to full stack


@dataclass
class Action:
    """A single player action."""
    action_type: ActionType
    amount: float = 0.0  # total chips committed this action (0 for fold/check)
    player_idx: int = -1

    def __repr__(self) -> str:
        if self.action_type in (ActionType.FOLD, ActionType.CHECK):
            return f"P{self.player_idx}:{self.action_type.name}"
        return f"P{self.player_idx}:{self.action_type.name}({self.amount:.1f})"


@dataclass
class Player:
    """A player at the table."""
    stack: float          # current chip count
    hole_cards: Tuple[int, int] = (-1, -1)  # card indices 0-51
    is_folded: bool = False
    is_all_in: bool = False
    bet_this_street: float = 0.0  # chips put in this street
    bet_total: float = 0.0        # chips put in this hand total
    seat: int = 0

    @property
    def is_active(self) -> bool:
        """Can this player still act?"""
        return not self.is_folded and not self.is_all_in

    @property
    def is_in_hand(self) -> bool:
        """Is this player still contesting the pot?"""
        return not self.is_folded


class GameState:
    """
    Full state of a NLHE hand.

    Cards are represented as integers 0-51:
        card_index = rank * 4 + suit
        rank: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
        suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
    """

    def __init__(
        self,
        num_players: int,
        stacks: List[float],
        small_blind: float = 0.5,
        big_blind: float = 1.0,
        dealer_button: int = 0,
    ):
        assert 2 <= num_players <= 9, f"num_players must be 2-9, got {num_players}"
        assert len(stacks) == num_players, "stacks length must match num_players"
        assert all(s > 0 for s in stacks), "all stacks must be positive"

        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_button = dealer_button

        # Players
        self.players = [
            Player(stack=stacks[i], seat=i) for i in range(num_players)
        ]

        # Community cards
        self.board: List[int] = []

        # Pot and betting
        self.pot: float = 0.0
        self.street: Street = Street.PREFLOP
        self.current_bet: float = 0.0    # highest bet this street
        self.min_raise: float = big_blind  # minimum raise increment
        self.last_raiser: int = -1

        # Action tracking
        self.action_history: List[Action] = []
        self.street_actions: List[Action] = []  # actions this street only

        # Turn management
        self.current_player_idx: int = -1  # set by _init_preflop
        self.num_actions_this_street: int = 0

        # Game status
        self.is_hand_over: bool = False
        self.winners: List[int] = []  # indices of winners

    def copy(self) -> GameState:
        """Create a deep copy for search/rollouts."""
        return copy.deepcopy(self)

    # -------------------------------------------------------------------------
    # Blind posting and street initialization
    # -------------------------------------------------------------------------

    def _sb_idx(self) -> int:
        """Small blind position."""
        if self.num_players == 2:
            return self.dealer_button  # heads-up: dealer is SB
        return (self.dealer_button + 1) % self.num_players

    def _bb_idx(self) -> int:
        """Big blind position."""
        if self.num_players == 2:
            return (self.dealer_button + 1) % self.num_players
        return (self.dealer_button + 2) % self.num_players

    def post_blinds(self) -> None:
        """Post small and big blinds."""
        sb_idx = self._sb_idx()
        bb_idx = self._bb_idx()

        sb_amount = min(self.small_blind, self.players[sb_idx].stack)
        bb_amount = min(self.big_blind, self.players[bb_idx].stack)

        self._place_bet(sb_idx, sb_amount)
        self._place_bet(bb_idx, bb_amount)

        self.current_bet = bb_amount
        self.min_raise = self.big_blind  # min raise is 1 BB preflop

        # First to act preflop: UTG (after BB)
        if self.num_players == 2:
            self.current_player_idx = sb_idx  # heads-up: SB acts first preflop
        else:
            self.current_player_idx = (bb_idx + 1) % self.num_players
        self._advance_to_active_player()

    def _place_bet(self, player_idx: int, amount: float) -> None:
        """Place a bet for a player (internal helper)."""
        p = self.players[player_idx]
        actual = min(amount, p.stack)
        p.stack -= actual
        p.bet_this_street += actual
        p.bet_total += actual
        self.pot += actual
        if p.stack == 0:
            p.is_all_in = True

    # -------------------------------------------------------------------------
    # Legal actions
    # -------------------------------------------------------------------------

    def get_legal_actions(self) -> List[ActionType]:
        """Return legal action types for current player."""
        if self.is_hand_over:
            return []

        p = self.players[self.current_player_idx]
        if not p.is_active:
            return []

        actions = [ActionType.FOLD]
        to_call = self.current_bet - p.bet_this_street

        if to_call <= 0:
            actions.append(ActionType.CHECK)
        else:
            actions.append(ActionType.CALL)

        # Can raise if we have more than just calling
        if p.stack > to_call:
            actions.append(ActionType.RAISE)

        # All-in is always available (as a special raise/call)
        if p.stack > 0:
            actions.append(ActionType.ALL_IN)

        return actions

    def get_min_raise_to(self) -> float:
        """Minimum total raise-to amount."""
        p = self.players[self.current_player_idx]
        min_to = self.current_bet + self.min_raise
        # Can't raise more than our stack allows
        max_to = p.bet_this_street + p.stack
        return min(min_to, max_to)

    def get_max_raise_to(self) -> float:
        """Maximum raise-to amount (all-in)."""
        p = self.players[self.current_player_idx]
        return p.bet_this_street + p.stack

    # -------------------------------------------------------------------------
    # Apply action
    # -------------------------------------------------------------------------

    def apply_action(self, action: Action) -> None:
        """Apply an action and advance the game state."""
        assert not self.is_hand_over, "Hand is already over"

        p = self.players[self.current_player_idx]
        assert p.is_active, f"Player {self.current_player_idx} cannot act"

        action.player_idx = self.current_player_idx

        if action.action_type == ActionType.FOLD:
            p.is_folded = True

        elif action.action_type == ActionType.CHECK:
            assert self.current_bet - p.bet_this_street <= 0, \
                "Cannot check when facing a bet"

        elif action.action_type == ActionType.CALL:
            to_call = self.current_bet - p.bet_this_street
            call_amount = min(to_call, p.stack)
            self._place_bet(self.current_player_idx, call_amount)
            action.amount = call_amount

        elif action.action_type == ActionType.RAISE:
            raise_to = action.amount  # total raise-to amount
            assert raise_to >= self.get_min_raise_to() - 1e-9, \
                f"Raise to {raise_to} below minimum {self.get_min_raise_to()}"
            assert raise_to <= self.get_max_raise_to() + 1e-9, \
                f"Raise to {raise_to} above maximum {self.get_max_raise_to()}"

            raise_to = min(raise_to, self.get_max_raise_to())
            additional = raise_to - p.bet_this_street
            raise_increment = raise_to - self.current_bet
            self.min_raise = max(self.min_raise, raise_increment)
            self.current_bet = raise_to
            self.last_raiser = self.current_player_idx
            self._place_bet(self.current_player_idx, additional)
            action.amount = raise_to

        elif action.action_type == ActionType.ALL_IN:
            all_in_to = p.bet_this_street + p.stack
            additional = p.stack
            if all_in_to > self.current_bet:
                # This is a raise
                raise_increment = all_in_to - self.current_bet
                if raise_increment >= self.min_raise:
                    self.min_raise = raise_increment
                self.current_bet = all_in_to
                self.last_raiser = self.current_player_idx
            self._place_bet(self.current_player_idx, additional)
            action.amount = all_in_to

        # Record action
        self.action_history.append(action)
        self.street_actions.append(action)
        self.num_actions_this_street += 1

        # Check if hand is over or street is over
        self._advance_game()

    # -------------------------------------------------------------------------
    # Game flow
    # -------------------------------------------------------------------------

    def _count_active(self) -> int:
        """Players who can still act."""
        return sum(1 for p in self.players if p.is_active)

    def _count_in_hand(self) -> int:
        """Players who haven't folded."""
        return sum(1 for p in self.players if p.is_in_hand)

    def _advance_to_active_player(self) -> None:
        """Move current_player_idx to next active player."""
        for _ in range(self.num_players):
            if self.players[self.current_player_idx].is_active:
                return
            self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def _is_street_over(self) -> bool:
        """Check if betting for the current street is complete."""
        if self._count_active() == 0:
            return True

        if self.num_actions_this_street == 0:
            return False

        # Check if all active players have acted and matched the current bet
        for p in self.players:
            if p.is_active and p.bet_this_street < self.current_bet:
                return False

        # Everyone has had a chance to act since the last raise
        # Count how many active players still need to act
        if self.last_raiser == -1:
            # No raises — street over when everyone has acted
            return self.num_actions_this_street >= self._count_active()
        else:
            # Action must have gone around to the last raiser
            return self._all_have_acted_since_last_raise()

    def _all_have_acted_since_last_raise(self) -> bool:
        """Check if all active players have acted since the last raise."""
        if self.last_raiser == -1:
            return True

        # Find the last raise in street_actions
        last_raise_idx = -1
        for i, a in enumerate(self.street_actions):
            if a.action_type in (ActionType.RAISE, ActionType.ALL_IN) and \
               a.player_idx == self.last_raiser:
                last_raise_idx = i

        if last_raise_idx == -1:
            return True

        # All active players after the raiser must have acted
        actions_after_raise = self.street_actions[last_raise_idx + 1:]
        players_acted = {a.player_idx for a in actions_after_raise}

        for i, p in enumerate(self.players):
            if i != self.last_raiser and p.is_active and i not in players_acted:
                return False

        return True

    def _advance_game(self) -> None:
        """After an action, determine next player or advance street."""
        # Only one player left — they win
        if self._count_in_hand() == 1:
            self._end_hand_single_winner()
            return

        # Move to next player
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        self._advance_to_active_player()

        # Check if street is over
        if self._is_street_over():
            if self.street == Street.RIVER or self._count_active() == 0:
                # Showdown (or everyone all-in)
                self.street = Street.SHOWDOWN
                self.is_hand_over = True
            else:
                self._next_street()

    def _next_street(self) -> None:
        """Advance to the next street."""
        if self.street == Street.PREFLOP:
            self.street = Street.FLOP
        elif self.street == Street.FLOP:
            self.street = Street.TURN
        elif self.street == Street.TURN:
            self.street = Street.RIVER

        # Reset street state
        for p in self.players:
            p.bet_this_street = 0.0
        self.current_bet = 0.0
        self.min_raise = self.big_blind
        self.last_raiser = -1
        self.street_actions = []
        self.num_actions_this_street = 0

        # First to act post-flop: first active player after dealer
        self.current_player_idx = (self.dealer_button + 1) % self.num_players
        self._advance_to_active_player()

        # If only one active player left (rest all-in), go to showdown
        if self._count_active() <= 1:
            self.street = Street.SHOWDOWN
            self.is_hand_over = True

    def _end_hand_single_winner(self) -> None:
        """End hand when all but one player has folded."""
        self.is_hand_over = True
        for i, p in enumerate(self.players):
            if p.is_in_hand:
                self.winners = [i]
                p.stack += self.pot
                self.pot = 0
                break

    # -------------------------------------------------------------------------
    # Side pots
    # -------------------------------------------------------------------------

    def calculate_side_pots(self) -> List[Tuple[float, List[int]]]:
        """
        Calculate side pots.

        Returns list of (pot_amount, [eligible_player_indices]).
        Sorted from main pot to side pots.
        """
        # Gather all-in amounts
        contributions = []
        for i, p in enumerate(self.players):
            if p.is_in_hand:
                contributions.append((p.bet_total, i))

        if not contributions:
            return []

        # Sort by contribution (ascending)
        contributions.sort(key=lambda x: x[0])

        pots: List[Tuple[float, List[int]]] = []
        prev_level = 0.0

        for level, _ in contributions:
            if level <= prev_level:
                continue

            increment = level - prev_level
            pot_amount = 0.0
            eligible = []

            for contrib, idx in contributions:
                pot_contribution = min(contrib, level) - min(contrib, prev_level)
                pot_amount += pot_contribution
                if self.players[idx].is_in_hand and contrib >= level:
                    eligible.append(idx)

            # Also add contributions from folded players at this level
            for i, p in enumerate(self.players):
                if p.is_folded:
                    pot_contribution = min(p.bet_total, level) - min(p.bet_total, prev_level)
                    pot_amount += pot_contribution

            if pot_amount > 0:
                pots.append((pot_amount, eligible))

            prev_level = level

        return pots

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    def card_to_str(card: int) -> str:
        """Convert card index (0-51) to string like 'As', 'Th', '2c'."""
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        return ranks[card // 4] + suits[card % 4]

    @staticmethod
    def str_to_card(s: str) -> int:
        """Convert string like 'As' to card index (0-51)."""
        ranks = '23456789TJQKA'
        suits = 'cdhs'
        return ranks.index(s[0]) * 4 + suits.index(s[1])

    def __repr__(self) -> str:
        active = [i for i, p in enumerate(self.players) if p.is_active]
        return (
            f"GameState(street={self.street.name}, pot={self.pot:.1f}, "
            f"board={[self.card_to_str(c) for c in self.board]}, "
            f"current_player={self.current_player_idx}, "
            f"active={active})"
        )
