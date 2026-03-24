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
