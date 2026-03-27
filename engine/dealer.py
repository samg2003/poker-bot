"""
Dealer module — manages the game loop for a full hand of NLHE.

Handles: shuffling, dealing, blind posting, street progression,
board dealing, and showdown with side pot distribution.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

from engine.game_state import (
    Action, ActionType, GameState, Player, Street,
)
from engine.hand_evaluator import HandEvaluator


class Dealer:
    """
    Manages a complete hand of poker.

    Usage:
        dealer = Dealer(num_players=6, stacks=[100]*6, big_blind=1.0)
        dealer.start_hand()

        while not dealer.is_hand_over():
            state = dealer.get_state()
            action = agent.get_action(state)  # your AI decides
            dealer.apply_action(action)

        results = dealer.get_results()
    """

    def __init__(
        self,
        num_players: int,
        stacks: List[float],
        small_blind: float = 0.5,
        big_blind: float = 1.0,
        dealer_button: int = 0,
        seed: Optional[int] = None,
    ):
        self.num_players = num_players
        self.initial_stacks = list(stacks)
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_button = dealer_button
        self.rng = random.Random(seed)

        self.state: Optional[GameState] = None
        self.deck: List[int] = []
        self.evaluator = HandEvaluator()

    def start_hand(self) -> GameState:
        """Initialize and deal a new hand."""
        # Create fresh game state
        self.state = GameState(
            num_players=self.num_players,
            stacks=list(self.initial_stacks),
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            dealer_button=self.dealer_button,
        )

        # Shuffle deck
        self.deck = list(range(52))
        self.rng.shuffle(self.deck)

        # Deal hole cards
        card_idx = 0
        for i in range(self.num_players):
            c1 = self.deck[card_idx]
            c2 = self.deck[card_idx + 1]
            self.state.players[i].hole_cards = (c1, c2)
            card_idx += 2

        self._next_card_idx = card_idx

        # Post blinds
        self.state.post_blinds()

        return self.state

    def is_hand_over(self) -> bool:
        """Check if the hand is complete."""
        return self.state is not None and self.state.is_hand_over

    def get_state(self) -> GameState:
        """Get current game state."""
        assert self.state is not None, "Hand not started"
        return self.state

    def apply_action(self, action: Action) -> None:
        """Apply a player action and handle street transitions."""
        assert self.state is not None, "Hand not started"

        prev_street = self.state.street
        self.state.apply_action(action)

        # Deal board cards if street advanced
        if self.state.street != prev_street and not self.state.is_hand_over:
            self._deal_board()

        # Handle showdown
        if self.state.street == Street.SHOWDOWN and self.state.is_hand_over:
            self._showdown()

    def _deal_board(self) -> None:
        """Deal community cards for the new street."""
        assert self.state is not None

        if self.state.street == Street.FLOP:
            # Burn one, deal three
            self._next_card_idx += 1  # burn
            for _ in range(3):
                self.state.board.append(self.deck[self._next_card_idx])
                self._next_card_idx += 1
        elif self.state.street in (Street.TURN, Street.RIVER):
            # Burn one, deal one
            self._next_card_idx += 1  # burn
            self.state.board.append(self.deck[self._next_card_idx])
            self._next_card_idx += 1

    def _showdown(self) -> None:
        """Evaluate hands and distribute pot(s)."""
        assert self.state is not None

        from engine.hand_evaluator import Eval7Evaluator

        side_pots = self.state.calculate_side_pots()
        if not side_pots:
            return

        # 1. Secretly calculate the A.I.'s EV *before* the RNG cards are dealt 
        #    so we can feed it to the Trainer, without ruining the real game physics!
        self.ev_profits = [-p.bet_total for p in self.state.players]
        use_ev = len(self.state.board) < 5
        
        for pot_amount, eligible in side_pots:
            eligible_hands = [list(self.state.players[i].hole_cards) for i in eligible]
            if use_ev:
                equities = Eval7Evaluator.get_equity(eligible_hands, self.state.board, runouts=1000)
                for idx, e_idx in enumerate(eligible):
                    self.ev_profits[e_idx] += pot_amount * equities[idx]
            else:
                winners_idx = Eval7Evaluator.get_showdown_winners(eligible_hands, self.state.board)
                for w_idx in winners_idx:
                    self.ev_profits[eligible[w_idx]] += pot_amount / len(winners_idx)

        # 2. Complete the physical board (The Casino RNG)
        while len(self.state.board) < 5:
            self._next_card_idx += 1  # burn
            self.state.board.append(self.deck[self._next_card_idx])
            self._next_card_idx += 1

        # 3. Distribute the ACTUAL, PHYSICAL chips to the players based purely 
        #    on who actually won the dealt cards (Strict 100% Poker Rules)
        winners_all = []
        for pot_amount, eligible in side_pots:
            eligible_hands = [list(self.state.players[i].hole_cards) for i in eligible]
            winners_idx = Eval7Evaluator.get_showdown_winners(eligible_hands, self.state.board)
            share = pot_amount / len(winners_idx)
            for w_idx in winners_idx:
                actual_player = eligible[w_idx]
                self.state.players[actual_player].stack += share
                if actual_player not in winners_all:
                    winners_all.append(actual_player)

        self.state.pot = 0
        self.state.winners = winners_all

    def get_results(self) -> Dict:
        """Get hand results after showdown."""
        assert self.state is not None and self.state.is_hand_over

        # Absolute physical profit
        profit = [
            self.state.players[i].stack - self.initial_stacks[i]
            for i in range(self.num_players)
        ]

        results = {
            'winners': self.state.winners,
            'final_stacks': [p.stack for p in self.state.players],
            'profit': profit,
            # Pass the secret EV mathematically calculated during showdown.
            # If the hand never reached showdown (someone folded), EV is exactly equal to profit! 
            'ev_profit': getattr(self, 'ev_profits', profit),
            'board': [GameState.card_to_str(c) for c in self.state.board],
        }

        # Add hand info for players who went to showdown
        if self.state.street == Street.SHOWDOWN:
            results['hands'] = {}
            for i, p in enumerate(self.state.players):
                if p.is_in_hand and p.hole_cards != (-1, -1):
                    cards = list(p.hole_cards) + self.state.board
                    rank = self.evaluator.evaluate_7(cards)
                    results['hands'][i] = {
                        'hole_cards': (
                            GameState.card_to_str(p.hole_cards[0]),
                            GameState.card_to_str(p.hole_cards[1]),
                        ),
                        'rank': rank,
                        'category': HandEvaluator.get_category_name(rank),
                    }

        return results

    def advance_button(self) -> None:
        """Move the dealer button for the next hand."""
        self.dealer_button = (self.dealer_button + 1) % self.num_players
        # Update stacks from current state
        if self.state:
            self.initial_stacks = [p.stack for p in self.state.players]
