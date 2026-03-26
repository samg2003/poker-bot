"""
Personality Perturbation System.

Instead of scripted bots, we create realistic opponents by applying continuous
modifiers to the base GTO policy. A "Nit" isn't a decision tree — it's a
competent player who happens to play too tight and fold too much.

Two levels of personality:
1. Global modifiers — overall tendencies (range_mult, aggression_mult, etc.)
2. Situational overrides — context-dependent adjustments (e.g., tight preflop
   but loose on wet boards, passive until river then overbets)

Usage:
    modifier = PersonalityModifier.nit()
    personality = SituationalPersonality(base=modifier, overrides={...})
    adjusted_probs = personality.apply(action_probs, context)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch


# =============================================================================
# Situation tags — auto-detected from game state
# =============================================================================

class Situation(Enum):
    """Context tags for situational personality overrides."""
    # Street
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()

    # Board texture
    WET_BOARD = auto()
    DRY_BOARD = auto()

    # Position
    IN_POSITION = auto()
    OUT_OF_POSITION = auto()

    # Action context
    FACING_BET = auto()
    FACING_RAISE = auto()
    CHECKED_TO = auto()

    # Stack depth
    SHORT_STACK = auto()    # <20bb
    MEDIUM_STACK = auto()   # 20-100bb
    DEEP_STACK = auto()     # >100bb

    # Behavioral
    RECENTLY_LOST_BIG_POT = auto()


# =============================================================================
# Personality Modifier
# =============================================================================

@dataclass
class PersonalityModifier:
    """
    Continuous modifiers that reshape a base policy's action distribution.

    All values default to 1.0 (no modification, i.e., GTO).
    Values >1 amplify the tendency, <1 suppress it.
    """
    range_mult: float = 1.0        # <1 = tighter, >1 = looser
    aggression_mult: float = 1.0   # <1 = passive, >1 = aggressive
    fold_pressure: float = 1.0     # >1 = folds more to raises, <1 = calls down
    bluff_mult: float = 1.0        # >1 = bluffs more, <1 = value-heavy
    sizing_mult: float = 1.0       # >1 = bigger bets, <1 = smaller bets
    cbet_mult: float = 1.0         # >1 = c-bets more, <1 = c-bets less

    def blend(self, other: 'PersonalityModifier', alpha: float) -> 'PersonalityModifier':
        """Linearly interpolate between two modifiers."""
        alpha = max(0.0, min(1.0, alpha))
        return PersonalityModifier(
            range_mult=self.range_mult * (1 - alpha) + other.range_mult * alpha,
            aggression_mult=self.aggression_mult * (1 - alpha) + other.aggression_mult * alpha,
            fold_pressure=self.fold_pressure * (1 - alpha) + other.fold_pressure * alpha,
            bluff_mult=self.bluff_mult * (1 - alpha) + other.bluff_mult * alpha,
            sizing_mult=self.sizing_mult * (1 - alpha) + other.sizing_mult * alpha,
            cbet_mult=self.cbet_mult * (1 - alpha) + other.cbet_mult * alpha,
        )

    # Pre-built archetypes (starting points for continuous sampling)
    # Values calibrated against typical online poker stats (VPIP/PFR/AF).
    # All multipliers stay in [0.3, 1.6] — the human-observable range.

    @classmethod
    def gto(cls) -> 'PersonalityModifier':
        """Unmodified base policy."""
        return cls()

    @classmethod
    def nit(cls) -> 'PersonalityModifier':
        """Tight-passive: plays few hands, folds to pressure. ~12/9 VPIP/PFR."""
        return cls(range_mult=0.6, aggression_mult=0.85, fold_pressure=1.35,
                   bluff_mult=0.35, sizing_mult=0.85, cbet_mult=0.6)

    @classmethod
    def tag(cls) -> 'PersonalityModifier':
        """Tight-aggressive: selective range, bets big when in. ~20/17 VPIP/PFR."""
        return cls(range_mult=0.75, aggression_mult=1.3, fold_pressure=0.8,
                   bluff_mult=0.8, sizing_mult=1.15, cbet_mult=1.2)

    @classmethod
    def lag(cls) -> 'PersonalityModifier':
        """Loose-aggressive: wide range, lots of raises. ~28/22 VPIP/PFR."""
        return cls(range_mult=1.25, aggression_mult=1.35, fold_pressure=0.7,
                   bluff_mult=1.25, sizing_mult=1.15, cbet_mult=1.3)

    @classmethod
    def maniac(cls) -> 'PersonalityModifier':
        """Hyper-aggressive: plays too many hands, raises too much. ~45/35 VPIP/PFR."""
        return cls(range_mult=1.5, aggression_mult=1.6, fold_pressure=0.55,
                   bluff_mult=1.5, sizing_mult=1.35, cbet_mult=1.5)

    @classmethod
    def calling_station(cls) -> 'PersonalityModifier':
        """Loose-passive: calls everything, rarely raises or folds. ~40/8 VPIP/PFR."""
        return cls(range_mult=1.35, aggression_mult=0.4, fold_pressure=0.35,
                   bluff_mult=0.3, sizing_mult=0.8, cbet_mult=0.45)

    @classmethod
    def fish(cls) -> 'PersonalityModifier':
        """Recreational player: limps too much, calls too much pre. ~50/10 VPIP/PFR."""
        return cls(range_mult=1.45, aggression_mult=0.5, fold_pressure=0.5,
                   bluff_mult=0.4, sizing_mult=0.7, cbet_mult=0.35)

    @classmethod
    def random(cls, rng: Optional[random.Random] = None) -> 'PersonalityModifier':
        """Sample a random personality from continuous distributions.

        Ranges are narrowed to [0.5, 1.6] to stay within human-like behavior.
        """
        rng = rng or random.Random()
        return cls(
            range_mult=rng.uniform(0.5, 1.6),
            aggression_mult=rng.uniform(0.35, 1.6),
            fold_pressure=rng.uniform(0.35, 1.5),
            bluff_mult=rng.uniform(0.25, 1.6),
            sizing_mult=rng.uniform(0.6, 1.5),
            cbet_mult=rng.uniform(0.35, 1.5),
        )


# =============================================================================
# Situational Personality
# =============================================================================

@dataclass
class SituationalPersonality:
    """
    A personality with context-dependent overrides.

    The base modifier is the default. When specific situations are detected,
    the corresponding override is blended in.
    """
    base: PersonalityModifier = field(default_factory=PersonalityModifier)
    overrides: Dict[Situation, PersonalityModifier] = field(default_factory=dict)

    def get_modifier(self, active_situations: List[Situation]) -> PersonalityModifier:
        """
        Get the effective modifier given current game context.

        If multiple situations match, overrides are averaged
        (each blended 50% with the base).
        """
        if not active_situations or not self.overrides:
            return self.base

        matching = [self.overrides[s] for s in active_situations if s in self.overrides]
        if not matching:
            return self.base

        # Blend each matching override with base, then average
        result = self.base
        blend_weight = 0.5 / len(matching)
        for override in matching:
            result = result.blend(override, blend_weight)

        return result

    def apply(
        self,
        action_probs: torch.Tensor,
        active_situations: List[Situation],
        hand_strength: float = 0.5,
        is_facing_raise: bool = False,
    ) -> torch.Tensor:
        """
        Apply personality to an action distribution.

        Args:
            action_probs: (4,) tensor [fold, check, call, raise]
            active_situations: list of current Situation tags
            hand_strength: 0-1 (0=worst, 1=nuts)
            is_facing_raise: whether currently facing a raise

        Returns:
            Modified action_probs (4,) tensor, re-normalized.
        """
        mod = self.get_modifier(active_situations)
        probs = action_probs.clone()

        # --- Range adjustment (smooth function of hand strength) ---
        # Tight players fold more with weak hands; loose players fold less.
        # fold_factor smoothly increases for weaker hands when range_mult < 1.
        # At hand_strength=0 and range_mult=0.6: fold_factor ≈ 1.4
        # At hand_strength=1: fold_factor ≈ 1.0 regardless of range_mult
        weakness = max(0.0, 1.0 - hand_strength)  # 0 for nuts, 1 for air
        fold_factor = 1.0 + weakness * (1.0 - mod.range_mult)  # range<1 → fold more
        probs[0] *= max(fold_factor, 0.1)

        # --- Aggression adjustment (Call/Check -> Raise) ---
        # If aggressive, shift weight from BOTH Call and Check to Raise
        aggression_shift = (mod.aggression_mult - 1.0) * 0.35  # scaled

        if aggression_shift > 0:
            # Shift from Call
            call_shift = probs[2].item() * min(aggression_shift, 0.9)
            probs[3] += call_shift
            probs[2] -= call_shift
            
            # Shift from Check (initiate bets!)
            check_shift = probs[1].item() * min(aggression_shift, 0.9)
            probs[3] += check_shift
            probs[1] -= check_shift
        elif aggression_shift < 0:
            # Passive: shift from Raise to Call
            raise_shift = probs[3].item() * min(-aggression_shift, 0.9)
            probs[2] += raise_shift
            probs[3] -= raise_shift

        # --- Fold pressure ---
        if is_facing_raise:
            probs[0] *= mod.fold_pressure

        # --- Bluff adjustment (Absolute + Relative) ---
        # If the base bot never bluffs (0% raise), relative scaling doesn't work.
        # We need to inject absolute bluffing probability for Maniacs!
        bluff_scale = 1.0 + (mod.bluff_mult - 1.0) * max(0.0, 1.0 - hand_strength * 2.0)
        probs[3] *= max(bluff_scale, 0.1)
        
        # Absolute bluff injection for weak hands if bluff_mult > 1
        if mod.bluff_mult > 1.0 and weakness > 0.5:
            abs_bluff = (mod.bluff_mult - 1.0) * 0.15 * weakness
            probs[3] += abs_bluff
            # Steal from fold/check
            probs[0] *= (1.0 - abs_bluff)
            probs[1] *= (1.0 - abs_bluff)

        # --- Clamp and re-normalize ---
        probs = probs.clamp(min=1e-6)
        probs = probs / probs.sum()

        return probs

    def apply_sizing(
        self,
        sizing_probs: List[float],
        active_situations: List[Situation],
    ) -> List[float]:
        """
        Apply personality to bet sizing distribution.

        sizing_mult > 1 shifts weight towards larger bets.
        sizing_mult < 1 shifts weight towards smaller bets.

        Args:
            sizing_probs: list of 10 floats (pot fraction buckets)
            active_situations: current situation tags

        Returns:
            Modified sizing_probs list, re-normalized.
        """
        mod = self.get_modifier(active_situations)
        n = len(sizing_probs)
        if n == 0:
            return sizing_probs

        adjusted = list(sizing_probs)
        sm = mod.sizing_mult

        # Apply exponential weighting: later indices = bigger bets
        # sizing_mult > 1 → weight larger sizes more; < 1 → weight smaller sizes
        for i in range(n):
            # Position weight: 0.0 for smallest, 1.0 for largest (all-in)
            pos = i / max(n - 1, 1)
            # Shift: sizing_mult=1.3 and pos=1.0 → factor ≈ 1.3
            # sizing_mult=0.7 and pos=1.0 → factor ≈ 0.7
            factor = 1.0 + (sm - 1.0) * (2.0 * pos - 1.0)
            adjusted[i] *= max(factor, 0.05)

        total = sum(adjusted)
        if total > 0:
            adjusted = [p / total for p in adjusted]

        return adjusted


# =============================================================================
# Situation detector
# =============================================================================

def detect_situations(
    street: int,
    board_cards: Optional[List[int]] = None,
    is_in_position: bool = False,
    is_facing_bet: bool = False,
    is_facing_raise: bool = False,
    stack_bb: float = 100.0,
    recent_loss: bool = False,
) -> List[Situation]:
    """
    Detect active situation tags from game state.

    Returns list of Situation enums for personality override lookup.
    """
    situations = []

    # Street
    street_map = {0: Situation.PREFLOP, 1: Situation.FLOP,
                  2: Situation.TURN, 3: Situation.RIVER}
    if street in street_map:
        situations.append(street_map[street])

    # Board texture (simplified — count connected/suited cards)
    if board_cards and len(board_cards) >= 3:
        suits = [c % 4 for c in board_cards[:3]]
        ranks = sorted([c // 4 for c in board_cards[:3]])
        flush_draw = len(set(suits)) <= 2
        connected = (ranks[2] - ranks[0]) <= 4
        if flush_draw or connected:
            situations.append(Situation.WET_BOARD)
        else:
            situations.append(Situation.DRY_BOARD)

    # Position
    situations.append(Situation.IN_POSITION if is_in_position else Situation.OUT_OF_POSITION)

    # Action context
    if is_facing_raise:
        situations.append(Situation.FACING_RAISE)
    elif is_facing_bet:
        situations.append(Situation.FACING_BET)
    else:
        situations.append(Situation.CHECKED_TO)

    # Stack depth
    if stack_bb < 20:
        situations.append(Situation.SHORT_STACK)
    elif stack_bb > 100:
        situations.append(Situation.DEEP_STACK)
    else:
        situations.append(Situation.MEDIUM_STACK)

    # Behavioral
    if recent_loss:
        situations.append(Situation.RECENTLY_LOST_BIG_POT)

    return situations


# =============================================================================
# Personality sampling for training
# =============================================================================

def sample_table_personalities(
    num_seats: int,
    gto_fraction: float = 0.9,
    rng: Optional[random.Random] = None,
) -> List[SituationalPersonality]:
    """
    Sample personalities for a training table.

    Args:
        num_seats: number of players
        gto_fraction: fraction of seats that are unmodified GTO

    Returns:
        List of SituationalPersonality, one per seat.
    """
    rng = rng or random.Random()
    personalities = []

    for _ in range(num_seats):
        if rng.random() < gto_fraction:
            # GTO player (no modification)
            personalities.append(SituationalPersonality())
        else:
            # Sample archetype — weighted to match $1/$2 online player pool
            archetypes = [
                PersonalityModifier.fish,             # 30%
                PersonalityModifier.fish,
                PersonalityModifier.fish,
                PersonalityModifier.fish,
                PersonalityModifier.fish,
                PersonalityModifier.fish,
                PersonalityModifier.tag,              # 25%
                PersonalityModifier.tag,
                PersonalityModifier.tag,
                PersonalityModifier.tag,
                PersonalityModifier.tag,
                PersonalityModifier.calling_station,  # 20%
                PersonalityModifier.calling_station,
                PersonalityModifier.calling_station,
                PersonalityModifier.calling_station,
                PersonalityModifier.lag,              # 10%
                PersonalityModifier.lag,
                PersonalityModifier.nit,              # 10%
                PersonalityModifier.nit,
                PersonalityModifier.maniac,           # 5%
            ]
            base = rng.choice(archetypes)()
            personalities.append(SituationalPersonality(base=base))

    return personalities


# =============================================================================
# Tilt model
# =============================================================================

@dataclass
class TiltState:
    """Track tilt-related state for a player during training."""
    consecutive_losses: int = 0
    big_pot_loss_count: int = 0
    total_result: float = 0.0
    hands_played: int = 0

    def update(self, hand_result: float, pot_size: float, big_blind: float) -> None:
        """Update tilt state after a hand."""
        self.hands_played += 1
        self.total_result += hand_result

        if hand_result < 0:
            self.consecutive_losses += 1
            if abs(hand_result) > 10 * big_blind:
                self.big_pot_loss_count += 1
        else:
            self.consecutive_losses = 0

    @property
    def is_tilting(self) -> bool:
        """Detect potential tilt based on recent results."""
        return self.consecutive_losses >= 3 or self.big_pot_loss_count >= 2

    def get_tilt_modifier(self) -> PersonalityModifier:
        """Get tilt-adjusted personality modifier."""
        if not self.is_tilting:
            return PersonalityModifier.gto()

        # Tilting → loosen up, get more aggressive, fold less
        tilt_intensity = min(self.consecutive_losses / 5.0, 1.0)
        return PersonalityModifier(
            range_mult=1.0 + tilt_intensity * 0.8,   # play more hands
            aggression_mult=1.0 + tilt_intensity * 1.0,  # raise more
            fold_pressure=1.0 - tilt_intensity * 0.4,    # call more raises
            bluff_mult=1.0 + tilt_intensity * 0.6,       # bluff more
            sizing_mult=1.0 + tilt_intensity * 0.5,      # bet bigger
        )

    def reset(self) -> None:
        """Reset tilt state (e.g., after a break)."""
        self.consecutive_losses = 0
        self.big_pot_loss_count = 0
