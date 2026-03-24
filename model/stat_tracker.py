"""
HUD-style stat tracker for opponent profiling.

Computes ~30 interpretable features per opponent from observed actions.
These are concatenated alongside the learned opponent embedding to give
the policy network both learned and hand-crafted signals.

Features are organized into 4 categories:
1. General preflop stats (VPIP, PFR, 3-Bet, etc.)
2. Per-street postflop stats (C-Bet%, Fold-to-CBet%, etc.)
3. Showdown & sizing stats (WTSD%, AF, avg bet size)
4. Tilt / behavioral signals (recent vs overall patterns)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class HandRecord:
    """Record of one completed hand for a player."""
    saw_flop: bool = False
    vpip: bool = False          # voluntarily put money in
    pfr: bool = False           # preflop raise
    three_bet: bool = False
    cold_call: bool = False
    squeeze: bool = False
    limp: bool = False

    # Per-street actions (flop=0, turn=1, river=2)
    was_pf_aggressor: bool = False
    cbet: List[bool] = field(default_factory=lambda: [False, False, False])
    fold_to_cbet: List[Optional[bool]] = field(default_factory=lambda: [None, None, None])
    check_raise: List[bool] = field(default_factory=lambda: [False, False, False])

    went_to_showdown: bool = False
    won_at_showdown: bool = False

    # Sizing
    bet_sizes: List[float] = field(default_factory=list)  # as fractions of pot
    total_wagered: float = 0.0
    result: float = 0.0  # chips won/lost


# Total number of stat features output per opponent
NUM_STAT_FEATURES = 30


class StatTracker:
    """
    Tracks and computes HUD-style statistics for each player.

    Usage:
        tracker = StatTracker()
        tracker.record_hand(player_id=3, hand_record=record)
        stats = tracker.get_stats(player_id=3)  # returns (30,) tensor
    """

    def __init__(self):
        self.hands: Dict[int, List[HandRecord]] = defaultdict(list)

    def record_hand(self, player_id: int, record: HandRecord) -> None:
        """Record a completed hand for a player."""
        self.hands[player_id].append(record)

    def reset(self, player_id: Optional[int] = None) -> None:
        """Reset stats (simulates sitting at new table)."""
        if player_id is not None:
            self.hands[player_id] = []
        else:
            self.hands.clear()

    def get_num_hands(self, player_id: int) -> int:
        """Number of hands recorded for this player."""
        return len(self.hands.get(player_id, []))

    def get_stats(self, player_id: int) -> torch.Tensor:
        """
        Compute all HUD stats for a player.

        Returns: (NUM_STAT_FEATURES,) tensor with values in [0, 1].
        Returns zeros if no hands recorded (fresh opponent → GTO mode).
        """
        records = self.hands.get(player_id, [])
        n = len(records)

        if n == 0:
            return torch.zeros(NUM_STAT_FEATURES, dtype=torch.float32)

        # ---------------------------------------------------------------
        # 1. General preflop stats
        # ---------------------------------------------------------------
        vpip = self._pct(records, lambda r: r.vpip)
        pfr = self._pct(records, lambda r: r.pfr)
        three_bet = self._pct(records, lambda r: r.three_bet)
        cold_call = self._pct(records, lambda r: r.cold_call)
        squeeze = self._pct(records, lambda r: r.squeeze)
        limp = self._pct(records, lambda r: r.limp)
        pfr_over_vpip = pfr / max(vpip, 0.01)  # aggression ratio preflop

        # ---------------------------------------------------------------
        # 2. Post-flop stats (per street, then averaged)
        # ---------------------------------------------------------------
        flop_records = [r for r in records if r.saw_flop]

        cbet_by_street = []
        fold_to_cbet_by_street = []
        check_raise_by_street = []

        for street_idx in range(3):  # flop, turn, river
            # C-Bet: only count when was preflop aggressor
            aggressor_records = [r for r in flop_records if r.was_pf_aggressor]
            cbet = self._pct(aggressor_records, lambda r, s=street_idx: r.cbet[s]) if aggressor_records else 0.0
            cbet_by_street.append(cbet)

            # Fold to C-Bet: only when facing a cbet
            facing_cbet = [r for r in flop_records if r.fold_to_cbet[street_idx] is not None]
            ftcb = self._pct(facing_cbet, lambda r, s=street_idx: r.fold_to_cbet[s]) if facing_cbet else 0.5
            fold_to_cbet_by_street.append(ftcb)

            # Check-raise
            cr = self._pct(flop_records, lambda r, s=street_idx: r.check_raise[s]) if flop_records else 0.0
            check_raise_by_street.append(cr)

        # ---------------------------------------------------------------
        # 3. Showdown & sizing stats
        # ---------------------------------------------------------------
        wtsd = self._pct(flop_records, lambda r: r.went_to_showdown) if flop_records else 0.0
        wsd = self._pct(
            [r for r in flop_records if r.went_to_showdown],
            lambda r: r.won_at_showdown
        ) if any(r.went_to_showdown for r in flop_records) else 0.5

        # Aggression factor and bet sizing
        all_bets = []
        for r in records:
            all_bets.extend(r.bet_sizes)
        avg_bet_size = sum(all_bets) / max(len(all_bets), 1)
        overbet_pct = sum(1 for b in all_bets if b > 1.0) / max(len(all_bets), 1)

        # ---------------------------------------------------------------
        # 4. Tilt / behavioral signals (last 10 vs overall)
        # ---------------------------------------------------------------
        recent = records[-10:] if n >= 10 else records
        vpip_recent = self._pct(recent, lambda r: r.vpip)
        vpip_delta = vpip_recent - vpip  # positive = loosening up (possible tilt)

        recent_results = [r.result for r in recent]
        stack_trajectory = sum(recent_results) / max(len(recent_results), 1) / 100.0  # normalized

        # ---------------------------------------------------------------
        # Assemble feature vector (30 features)
        # ---------------------------------------------------------------
        features = [
            # General preflop (7)
            vpip, pfr, three_bet, cold_call, squeeze, limp, pfr_over_vpip,
            # Post-flop per street (9): cbet, fold_to_cbet, check_raise × 3 streets
            *cbet_by_street, *fold_to_cbet_by_street, *check_raise_by_street,
            # Showdown & sizing (6)
            wtsd, wsd, avg_bet_size, overbet_pct,
            min(n / 500.0, 1.0),  # sample size confidence (0-1, saturates at 500 hands)
            min(len(all_bets) / 200.0, 1.0),  # bet sample confidence
            # Tilt signals (2)
            vpip_delta + 0.5,     # centered at 0.5 (0.5 = no change)
            stack_trajectory + 0.5,  # centered at 0.5
            # Padding to ensure exactly 30
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

        return torch.tensor(features[:NUM_STAT_FEATURES], dtype=torch.float32)

    @staticmethod
    def _pct(records: list, predicate) -> float:
        """Percentage of records matching predicate."""
        if not records:
            return 0.0
        return sum(1 for r in records if predicate(r)) / len(records)
