"""
Fast hand evaluator for 5-7 card poker hands.

Uses a lookup-table-free approach with direct ranking.
Hand ranks are comparable integers — higher = better hand.

Rank encoding (32-bit):
    [hand_category (4 bits)] [tiebreaker kickers (remaining bits)]
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Tuple
import random

try:
    import eval7
    HAS_EVAL7 = True
except ImportError:
    HAS_EVAL7 = False


# Hand categories (higher = better)
HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush"
]


class HandEvaluator:
    """Evaluate poker hands and compare them."""

    @staticmethod
    def card_rank(card: int) -> int:
        """Get rank (0=2 ... 12=A) from card index."""
        return card // 4

    @staticmethod
    def card_suit(card: int) -> int:
        """Get suit (0=c, 1=d, 2=h, 3=s) from card index."""
        return card % 4

    @classmethod
    def evaluate_5(cls, cards: List[int]) -> int:
        """
        Evaluate exactly 5 cards. Returns an integer rank.
        Higher rank = better hand.
        """
        assert len(cards) == 5

        ranks = sorted([cls.card_rank(c) for c in cards], reverse=True)
        suits = [cls.card_suit(c) for c in cards]

        is_flush = len(set(suits)) == 1

        # Check for straight
        is_straight = False
        straight_high = -1
        if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
            is_straight = True
            straight_high = ranks[0]
        # Ace-low straight (A-2-3-4-5)
        elif ranks == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3  # 5-high straight

        # Count rank frequencies
        rank_counts: dict = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1

        counts = sorted(rank_counts.values(), reverse=True)
        # Ranks sorted by (count desc, rank desc)
        sorted_ranks = sorted(rank_counts.keys(),
                              key=lambda r: (rank_counts[r], r), reverse=True)

        if is_straight and is_flush:
            return cls._make_rank(STRAIGHT_FLUSH, [straight_high])

        if counts == [4, 1]:
            return cls._make_rank(FOUR_OF_A_KIND, sorted_ranks)

        if counts == [3, 2]:
            return cls._make_rank(FULL_HOUSE, sorted_ranks)

        if is_flush:
            return cls._make_rank(FLUSH, ranks)

        if is_straight:
            return cls._make_rank(STRAIGHT, [straight_high])

        if counts == [3, 1, 1]:
            return cls._make_rank(THREE_OF_A_KIND, sorted_ranks)

        if counts == [2, 2, 1]:
            # Two pair: higher pair first, then lower pair, then kicker
            pairs = [r for r, c in rank_counts.items() if c == 2]
            pairs.sort(reverse=True)
            kicker = [r for r, c in rank_counts.items() if c == 1]
            return cls._make_rank(TWO_PAIR, pairs + kicker)

        if counts == [2, 1, 1, 1]:
            return cls._make_rank(ONE_PAIR, sorted_ranks)

        return cls._make_rank(HIGH_CARD, ranks)

    @classmethod
    def evaluate_7(cls, cards: List[int]) -> int:
        """
        Evaluate the best 5-card hand from 7 cards.
        Returns the highest rank among all C(7,5)=21 combos.
        """
        assert len(cards) == 7
        best = -1
        for combo in combinations(cards, 5):
            rank = cls.evaluate_5(list(combo))
            if rank > best:
                best = rank
        return best

    @classmethod
    def evaluate(cls, hole_cards: List[int], board: List[int]) -> int:
        """
        Evaluate a player's hand given hole cards and board.
        Works with 3-7 total cards.
        """
        all_cards = list(hole_cards) + list(board)
        n = len(all_cards)
        if n < 5:
            # Pad evaluation for partial boards (useful for equity calcs)
            # Just evaluate what we have — not a full hand
            return cls.evaluate_5(all_cards + [0] * (5 - n))
        elif n == 5:
            return cls.evaluate_5(all_cards)
        elif n == 6:
            best = -1
            for combo in combinations(all_cards, 5):
                rank = cls.evaluate_5(list(combo))
                if rank > best:
                    best = rank
            return best
        else:
            return cls.evaluate_7(all_cards)

    @staticmethod
    def _make_rank(category: int, kickers: List[int]) -> int:
        """
        Encode hand category + kickers into a single comparable integer.
        Category takes the top 4 bits, kickers fill the rest.
        """
        rank = category << 20
        for i, k in enumerate(kickers[:5]):
            rank |= k << (4 * (4 - i))
        return rank

    @staticmethod
    def get_category(rank: int) -> int:
        """Extract hand category from rank integer."""
        return rank >> 20

    @staticmethod
    def get_category_name(rank: int) -> str:
        """Get human-readable hand category name."""
        cat = rank >> 20
        if 0 <= cat < len(CATEGORY_NAMES):
            return CATEGORY_NAMES[cat]
        return "Unknown"

    @classmethod
    def hand_to_str(cls, rank: int) -> str:
        """Human-readable hand description."""
        return cls.get_category_name(rank)

class Eval7Evaluator:
    """Fast C++ EV and Showdown calculator binding."""
    
    RANK_CHARS = "23456789TJQKA"
    SUIT_CHARS = "cdhs"

    @classmethod
    def int_to_eval7(cls, card_int: int):
        """Convert engine integer 0-51 (2c...As) to eval7 Card."""
        if card_int < 0 or card_int > 51:
            return None
        r = card_int // 4
        s = card_int % 4
        return eval7.Card(f"{cls.RANK_CHARS[r]}{cls.SUIT_CHARS[s]}")

    @classmethod
    def ints_to_eval7_list(cls, card_ints: List[int]):
        return [cls.int_to_eval7(c) for c in card_ints if c >= 0]

    @classmethod
    def get_equity(cls, known_hands: List[List[int]], board: List[int], runouts: int = 1000) -> List[float]:
        """
        Calculates multi-way Monte Carlo EV dynamically.
        Pass lists of integers.
        """
        if not HAS_EVAL7:
            raise ImportError("eval7 not installed. Please run `pip install eval7`.")
            
        e7_hands = [cls.ints_to_eval7_list(h) for h in known_hands]
        e7_board = cls.ints_to_eval7_list(board)
        
        wins = [0.0] * len(known_hands)
        deck = eval7.Deck()
        
        # Remove dead cards
        dead_cards = e7_board.copy()
        for h in e7_hands:
            dead_cards.extend(h)
            
        for c in dead_cards:
            deck.cards.remove(c)
            
        cards_needed = 5 - len(e7_board)
        deck_list = deck.cards
        
        if cards_needed > 0:
            import numpy as np
            deck_arr = np.array(deck_list, dtype=object)
            # Fast vectorized without-replacement sampling
            rand_vals = np.random.rand(runouts, len(deck_list))
            sample_indices = np.argsort(rand_vals, axis=1)[:, :cards_needed]
            sampled_cards = deck_arr[sample_indices]

            for r_idx in range(runouts):
                sim_board = e7_board + sampled_cards[r_idx].tolist()
                
                best_score = -1
                winners = []
                for i, h in enumerate(e7_hands):
                    score = eval7.evaluate(sim_board + h)
                    if score > best_score:
                        best_score = score
                        winners = [i]
                    elif score == best_score:
                        winners.append(i)
                        
                for w in winners:
                    wins[w] += 1.0 / len(winners)
                    
        else:
            for _ in range(runouts):
                sim_board = e7_board
                
                best_score = -1
                winners = []
                for i, h in enumerate(e7_hands):
                    score = eval7.evaluate(sim_board + h)
                    if score > best_score:
                        best_score = score
                        winners = [i]
                    elif score == best_score:
                        winners.append(i)
                        
                for w in winners:
                    wins[w] += 1.0 / len(winners)
                
        return [w / runouts for w in wins]

    @classmethod
    def get_showdown_winners(cls, known_hands: List[List[int]], board: List[int]) -> List[int]:
        """Instant exact evaluation for a fully dealt board. Returns indices of winners (handles splits)."""
        if not HAS_EVAL7:
            raise ImportError("eval7 not installed.")
            
        e7_hands = [cls.ints_to_eval7_list(h) for h in known_hands]
        e7_board = cls.ints_to_eval7_list(board)
        
        best_score = -1
        winners = []
        for i, h in enumerate(e7_hands):
            score = eval7.evaluate(e7_board + h)
            if score > best_score:
                best_score = score
                winners = [i]
            elif score == best_score:
                winners.append(i)
                
        return winners


class RunoutCache:
    """Cache pre-simulated runout scores for a fixed board + set of hands.

    Simulate N runouts once when the street/board changes. For each runout,
    store every player's eval7 hand score. Then equity for any subset of
    eligible players is derived instantly by finding the max score among
    the eligible set per runout — no re-sampling or re-evaluation needed.

    Usage:
        cache = RunoutCache()
        # Called once per street (when board changes):
        cache.update(all_hands, board, runouts=500)
        # Called many times per street (nearly free):
        equities = cache.get_equity(eligible=[0, 1, 2])
        equities = cache.get_equity(eligible=[0, 2])  # player 1 folded
    """

    def __init__(self):
        self._board_key = None       # tuple of board cards (cache invalidation key)
        self._hands_key = None       # tuple of hole cards (cache invalidation key)
        self._scores = None          # list[list[int]]: scores[runout_idx][player_idx]
        self._num_players = 0
        self._runouts = 0

    def update(self, hands: List[List[int]], board: List[int], runouts: int = 500):
        """Simulate runouts and cache per-player scores.

        Only re-simulates if board or hands have changed since last call.
        """
        board_key = tuple(sorted(c for c in board if c >= 0))
        hands_key = tuple(tuple(h) for h in hands)

        if board_key == self._board_key and hands_key == self._hands_key:
            return  # Cache hit — board and hands unchanged

        if not HAS_EVAL7:
            raise ImportError("eval7 not installed.")

        self._board_key = board_key
        self._hands_key = hands_key
        self._num_players = len(hands)
        self._runouts = runouts

        e7_hands = [Eval7Evaluator.ints_to_eval7_list(h) for h in hands]
        e7_board = Eval7Evaluator.ints_to_eval7_list(board)

        # Build deck minus dead cards
        deck = eval7.Deck()
        dead_cards = e7_board.copy()
        for h in e7_hands:
            dead_cards.extend(h)
        for c in dead_cards:
            deck.cards.remove(c)

        cards_needed = 5 - len(e7_board)
        deck_list = deck.cards

        scores = []

        if cards_needed > 0:
            import numpy as np
            deck_arr = np.array(deck_list, dtype=object)
            rand_vals = np.random.rand(runouts, len(deck_list))
            sample_indices = np.argsort(rand_vals, axis=1)[:, :cards_needed]
            sampled_cards = deck_arr[sample_indices]

            for r_idx in range(runouts):
                sim_board = e7_board + sampled_cards[r_idx].tolist()
                row = [eval7.evaluate(sim_board + h) for h in e7_hands]
                scores.append(row)
        else:
            # River — single evaluation, repeat for consistency
            row = [eval7.evaluate(e7_board + h) for h in e7_hands]
            scores = [row]  # Only 1 unique evaluation needed
            self._runouts = 1

        self._scores = scores

    def get_equity(self, eligible: List[int]) -> List[float]:
        """Compute equity for a subset of eligible players from cached scores.

        Returns equities indexed by position in `eligible` list.
        Nearly free — just integer comparisons over cached scores.
        """
        if self._scores is None:
            return [1.0 / len(eligible)] * len(eligible)

        n = len(eligible)
        wins = [0.0] * n
        num_runouts = len(self._scores)

        for scores_row in self._scores:
            best_score = -1
            winners = []
            for local_idx, player_idx in enumerate(eligible):
                s = scores_row[player_idx]
                if s > best_score:
                    best_score = s
                    winners = [local_idx]
                elif s == best_score:
                    winners.append(local_idx)

            share = 1.0 / len(winners)
            for w in winners:
                wins[w] += share

        return [w / num_runouts for w in wins]

    def invalidate(self):
        """Force re-simulation on next update call."""
        self._board_key = None
        self._hands_key = None
        self._scores = None
