"""
Leduc Hold'em — a medium-complexity poker game for validating
the neural network self-play pipeline before scaling to NLHE.

Rules:
- 6-card deck: J♠, J♦, Q♠, Q♦, K♠, K♦ (2 suits × 3 ranks)
- 2 players, ante 1 each
- Each dealt 1 card
- 2 betting rounds: preflop + "flop" (1 community card)
- Max 2 raises per round, fixed bet sizes (2 preflop, 4 postflop)
- Pairs beat high card; higher pair/card wins ties

This game has ~1,000 information sets — small enough to verify
convergence but complex enough to need real learning.
"""

from __future__ import annotations

import random
from itertools import product
from typing import List, Optional, Tuple


# Cards: (rank, suit) → rank: 0=J, 1=Q, 2=K; suit: 0, 1
JACK, QUEEN, KING = 0, 1, 2
CARD_NAMES = {0: 'J', 1: 'Q', 2: 'K'}
SUIT_NAMES = {0: '♠', 1: '♦'}
DECK = list(product([JACK, QUEEN, KING], [0, 1]))  # 6 cards

# Actions
CHECK, BET, FOLD, CALL, RAISE = 'c', 'b', 'f', 'k', 'r'


class LeducState:
    """
    A state in Leduc Hold'em.

    Tracks:
    - Player cards and community card
    - Action history per round
    - Current round (0 = preflop, 1 = flop)
    """

    def __init__(
        self,
        p1_card: Tuple[int, int],
        p2_card: Tuple[int, int],
        board_card: Optional[Tuple[int, int]] = None,
        history: Tuple[str, ...] = (),
        round_idx: int = 0,
    ):
        self.p1_card = p1_card
        self.p2_card = p2_card
        self.board_card = board_card
        self.history = history
        self.round_idx = round_idx

        # Parse betting state
        self._parse_state()

    def _parse_state(self):
        """Parse history to determine current betting state."""
        self.num_bets = [0, 0]   # bets per round
        self.round_histories = [[], []]
        current_round = 0

        for action in self.history:
            if action == '|':
                current_round = 1
                continue
            self.round_histories[current_round].append(action)
            if action in (BET, RAISE):
                self.num_bets[current_round] += 1

    @property
    def current_player(self) -> int:
        """0 or 1."""
        round_actions = self.round_histories[self.round_idx]
        return len(round_actions) % 2

    @property
    def is_terminal(self) -> bool:
        """Is the game over?"""
        rh = self.round_histories[self.round_idx]

        # Fold
        if rh and rh[-1] == FOLD:
            return True

        # Check-check or call ending a round
        if len(rh) >= 2:
            if rh[-2:] == [CHECK, CHECK]:
                return self.round_idx == 1
            if rh[-1] == CALL:
                return self.round_idx == 1

        return False

    @property
    def _round_over(self) -> bool:
        """Is the current round over (but game may continue)?"""
        rh = self.round_histories[self.round_idx]
        if not rh:
            return False
        if rh[-1] == FOLD:
            return True
        if len(rh) >= 2:
            if rh[-2:] == [CHECK, CHECK]:
                return True
            if rh[-1] == CALL:
                return True
        return False

    def get_payoff(self, player: int) -> float:
        """Get payoff for the given player at terminal state."""
        assert self.is_terminal

        # Calculate pot
        pot = 2  # antes
        bet_sizes = [2, 4]  # preflop bets = 2, flop bets = 4

        for round_i in range(2):
            rh = self.round_histories[round_i]
            for action in rh:
                if action in (BET, RAISE):
                    pot += bet_sizes[round_i]
                elif action == CALL:
                    pot += bet_sizes[round_i]

        # Check for fold
        all_actions = []
        for rh in self.round_histories:
            all_actions.extend(rh)

        if all_actions and all_actions[-1] == FOLD:
            # Player who folded loses
            folder = len(all_actions) % 2  # whose turn when fold happened
            # Actually, the player who folds is the one who TOOK the fold action
            # We need to track whose turn it is at fold time
            folder = self._get_folder()
            winner = 1 - folder
            # Calculate how much the folder put in
            folder_invested = self._player_invested(folder)
            if player == winner:
                return folder_invested
            else:
                return -folder_invested

        # Showdown
        winner = self._get_showdown_winner()
        half_pot = pot // 2
        if winner == -1:  # tie
            return 0
        if player == winner:
            return half_pot
        return -half_pot

    def _get_folder(self) -> int:
        """Determine which player folded."""
        action_count = 0
        for round_i in range(2):
            for action in self.round_histories[round_i]:
                if action == FOLD:
                    return action_count % 2
                action_count += 1
        return -1

    def _player_invested(self, player: int) -> int:
        """How much has a player put into the pot (excluding ante)?"""
        invested = 1  # ante
        bet_sizes = [2, 4]
        action_count = 0
        for round_i in range(2):
            for action in self.round_histories[round_i]:
                if action_count % 2 == player:
                    if action in (BET, RAISE, CALL):
                        invested += bet_sizes[round_i]
                action_count += 1
        return invested

    def _get_showdown_winner(self) -> int:
        """Determine winner at showdown. -1 = tie."""
        p1_rank, p1_suit = self.p1_card
        p2_rank, p2_suit = self.p2_card

        # Pair with board
        if self.board_card is not None:
            board_rank = self.board_card[0]
            p1_pair = p1_rank == board_rank
            p2_pair = p2_rank == board_rank

            if p1_pair and not p2_pair:
                return 0
            if p2_pair and not p1_pair:
                return 1
            if p1_pair and p2_pair:
                return -1  # both pair (same rank)

        # High card
        if p1_rank > p2_rank:
            return 0
        if p2_rank > p1_rank:
            return 1
        return -1  # tie

    def get_actions(self) -> List[str]:
        """Legal actions."""
        if self.is_terminal or self._round_over:
            return []

        rh = self.round_histories[self.round_idx]

        if not rh:
            return [CHECK, BET]

        last = rh[-1]
        if last == CHECK:
            return [CHECK, BET]
        if last == BET:
            if self.num_bets[self.round_idx] < 2:
                return [FOLD, CALL, RAISE]
            return [FOLD, CALL]
        if last == RAISE:
            return [FOLD, CALL]  # max 2 bets per round

        return [CHECK, BET]

    def apply(self, action: str) -> 'LeducState':
        """Return new state after action."""
        new_history = self.history + (action,)
        new_round = self.round_idx

        # Check if round ends after this action
        new_state = LeducState(
            self.p1_card, self.p2_card, self.board_card,
            new_history, new_round,
        )

        # If current round is over and game isn't terminal, advance
        if new_state._round_over and not new_state.is_terminal and new_round == 0:
            new_history = new_history + ('|',)
            new_state = LeducState(
                self.p1_card, self.p2_card, self.board_card,
                new_history, 1,
            )

        return new_state

    def info_set_key(self) -> str:
        """Information set key for current player."""
        player = self.current_player
        card = self.p1_card if player == 0 else self.p2_card
        card_str = f"{CARD_NAMES[card[0]]}{SUIT_NAMES[card[1]]}"

        board_str = ''
        if self.board_card is not None and self.round_idx > 0:
            board_str = f"|{CARD_NAMES[self.board_card[0]]}{SUIT_NAMES[self.board_card[1]]}"

        history_str = ''.join(self.history)
        return f"{card_str}{board_str}:{history_str}"

    def __repr__(self) -> str:
        p1 = f"{CARD_NAMES[self.p1_card[0]]}{SUIT_NAMES[self.p1_card[1]]}"
        p2 = f"{CARD_NAMES[self.p2_card[0]]}{SUIT_NAMES[self.p2_card[1]]}"
        board = ''
        if self.board_card:
            board = f" board={CARD_NAMES[self.board_card[0]]}{SUIT_NAMES[self.board_card[1]]}"
        return f"LeducState({p1} vs {p2}{board} h={''.join(self.history)})"


def deal_leduc(rng: Optional[random.Random] = None) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """Deal a random Leduc hand. Returns (p1_card, p2_card, board_card)."""
    if rng is None:
        rng = random.Random()
    cards = list(DECK)
    rng.shuffle(cards)
    return cards[0], cards[1], cards[2]
