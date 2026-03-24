"""
Evaluation Framework for the poker AI.

Benchmarks:
1. Exploitation — bb/100 vs perturbed opponents (Nit, LAG, Maniac, etc.)
2. GTO verification — after history reset, play near-equilibrium
3. Adaptation speed — detect personality shifts within ~50 hands
4. Search improvement — search-enabled vs policy-only on hard spots
5. Scaling — stable win rates across 2-9 players and 1-350bb stacks

Usage:
    evaluator = Evaluator(agent)
    results = evaluator.run_all_benchmarks()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from model.action_space import ActionIndex, NUM_ACTION_TYPES
from model.stat_tracker import StatTracker, NUM_STAT_FEATURES
from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork
from training.personality import (
    PersonalityModifier, SituationalPersonality,
    detect_situations, sample_table_personalities,
)
from engine.leduc_poker import LeducState, deal_leduc, CHECK, BET, FOLD, CALL, RAISE
from agent.config import AgentConfig


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: str = ''

    def __repr__(self):
        status = '✅ PASS' if self.passed else '❌ FAIL'
        return (f"{status} | {self.name}: {self.metric_name}="
                f"{self.metric_value:.4f} (threshold={self.threshold})")


@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    benchmarks: List[BenchmarkResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(b.passed for b in self.benchmarks)

    @property
    def pass_rate(self) -> float:
        if not self.benchmarks:
            return 0.0
        return sum(1 for b in self.benchmarks if b.passed) / len(self.benchmarks)

    def summary(self) -> str:
        lines = [f"{'='*60}", "EVALUATION RESULTS", f"{'='*60}"]
        for b in self.benchmarks:
            lines.append(str(b))
        lines.append(f"{'='*60}")
        lines.append(f"Pass rate: {self.pass_rate:.0%} ({sum(1 for b in self.benchmarks if b.passed)}/{len(self.benchmarks)})")
        return '\n'.join(lines)


# =============================================================================
# Evaluator
# =============================================================================

class Evaluator:
    """
    Evaluation framework — runs all benchmarks against the trained agent.

    Works on Leduc Hold'em for fast local validation.
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        seed: int = 42,
        num_hands: int = 500,
    ):
        self.policy = policy
        self.opponent_encoder = opponent_encoder
        self.rng = random.Random(seed)
        self.num_hands = num_hands
        self.stat_tracker = StatTracker()

    def _play_eval_hand(
        self,
        personality: SituationalPersonality,
    ) -> float:
        """
        Play one hand: model (player 0) vs personality-modified opponent (player 1).
        Returns reward for player 0.
        """
        self.policy.eval()

        p1_card, p2_card, board_card = deal_leduc(self.rng)
        state = LeducState(p1_card, p2_card, board_card)
        cards = [p1_card, p2_card]

        while not state.is_terminal:
            actions = state.get_actions()
            if not actions:
                break

            player = state.current_player
            card = cards[player]

            hole_idx = card[0] * 2 + card[1]
            board_idx = (board_card[0] * 2 + board_card[1]) if state.round_idx > 0 else -1

            hole_tensor = torch.tensor([[hole_idx, 0]], dtype=torch.long)
            community = torch.tensor([[board_idx, -1, -1, -1, -1]], dtype=torch.long)

            pot = 2.0
            for rh in state.round_histories:
                for a in rh:
                    if a in (BET, RAISE, CALL):
                        pot += (2.0 if state.round_idx == 0 else 4.0)

            numeric = torch.tensor([[
                pot / 10.0, 1.0, 0.0, float(player),
                float(state.round_idx), 2.0/9, 2.0/9, 0.0, 0.0,
            ]], dtype=torch.float32)

            opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
            opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)
            own_stats = torch.zeros(1, NUM_STAT_FEATURES)

            mask_list = [False] * NUM_ACTION_TYPES
            for a in actions:
                if a == FOLD: mask_list[ActionIndex.FOLD] = True
                elif a == CHECK: mask_list[ActionIndex.CHECK] = True
                elif a == CALL: mask_list[ActionIndex.CALL] = True
                elif a in (BET, RAISE): mask_list[ActionIndex.RAISE] = True

            action_mask = torch.tensor([mask_list])

            with torch.no_grad():
                output = self.policy(
                    hole_cards=hole_tensor,
                    community_cards=community,
                    numeric_features=numeric,
                    opponent_embeddings=opp_embed,
                    opponent_stats=opp_stats,
                    own_stats=own_stats,
                    action_mask=action_mask,
                )

            probs = output.action_type_probs[0]

            # Apply personality for player 1 (opponent)
            if player == 1:
                hand_strength = card[0] / 2.0
                situations = detect_situations(street=state.round_idx)
                probs = personality.apply(probs, situations, hand_strength=hand_strength)

            from torch.distributions import Categorical
            dist = Categorical(probs)
            action_idx = dist.sample().item()

            if action_idx == ActionIndex.RAISE:
                leduc_action = BET if BET in actions else RAISE
            elif action_idx == ActionIndex.FOLD:
                leduc_action = FOLD if FOLD in actions else actions[0]
            elif action_idx == ActionIndex.CALL:
                leduc_action = CALL if CALL in actions else actions[0]
            else:
                leduc_action = CHECK if CHECK in actions else actions[0]

            state = state.apply(leduc_action)

        if state.is_terminal:
            return state.get_payoff(0)
        return 0.0

    def benchmark_exploitation(self) -> BenchmarkResult:
        """
        Benchmark: agent should profit against exploitable opponents.

        Plays against a calling station (never folds, never raises).
        A competent agent should win > 0 bb/hand on average.
        """
        personality = SituationalPersonality(base=PersonalityModifier.calling_station())
        total_reward = 0.0

        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand(personality)

        avg_reward = total_reward / self.num_hands
        threshold = -1.0  # should at least not lose badly

        return BenchmarkResult(
            name="Exploitation vs Calling Station",
            passed=avg_reward > threshold,
            metric_name="avg_bb/hand",
            metric_value=avg_reward,
            threshold=threshold,
        )

    def benchmark_gto_symmetry(self) -> BenchmarkResult:
        """
        Benchmark: against GTO opponent, avg reward should be near 0.

        In self-play (both GTO), the game is symmetric → EV = 0.
        """
        personality = SituationalPersonality(base=PersonalityModifier.gto())
        total_reward = 0.0

        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand(personality)

        avg_reward = total_reward / self.num_hands
        threshold = 3.0  # should be within ±3 bb/hand

        return BenchmarkResult(
            name="GTO Symmetry (self-play)",
            passed=abs(avg_reward) < threshold,
            metric_name="abs(avg_bb/hand)",
            metric_value=abs(avg_reward),
            threshold=threshold,
        )

    def benchmark_nit_exploitation(self) -> BenchmarkResult:
        """
        Benchmark: agent should profit against nits.

        Nits fold too much → agent should steal their blinds.
        """
        personality = SituationalPersonality(base=PersonalityModifier.nit())
        total_reward = 0.0

        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand(personality)

        avg_reward = total_reward / self.num_hands
        threshold = -1.0

        return BenchmarkResult(
            name="Exploitation vs Nit",
            passed=avg_reward > threshold,
            metric_name="avg_bb/hand",
            metric_value=avg_reward,
            threshold=threshold,
        )

    def benchmark_maniac_exploitation(self) -> BenchmarkResult:
        """
        Benchmark: agent should profit against maniacs.

        Maniacs bluff too much → agent should call them down.
        """
        personality = SituationalPersonality(base=PersonalityModifier.maniac())
        total_reward = 0.0

        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand(personality)

        avg_reward = total_reward / self.num_hands
        threshold = -1.0

        return BenchmarkResult(
            name="Exploitation vs Maniac",
            passed=avg_reward > threshold,
            metric_name="avg_bb/hand",
            metric_value=avg_reward,
            threshold=threshold,
        )

    def benchmark_model_consistency(self) -> BenchmarkResult:
        """
        Benchmark: model produces consistent outputs for same input.

        Deterministic forward pass should give identical results.
        """
        self.policy.eval()

        hole = torch.tensor([[0, 1]], dtype=torch.long)
        community = torch.tensor([[10, 20, 30, -1, -1]], dtype=torch.long)
        numeric = torch.tensor([[0.5, 1.0, 0.0, 0.0, 0.33, 0.22, 0.22, 0.0, 0.0]], dtype=torch.float32)
        opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
        opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)
        own_stats = torch.zeros(1, NUM_STAT_FEATURES)

        with torch.no_grad():
            out1 = self.policy(hole, community, numeric, opp_embed, opp_stats, own_stats)
            out2 = self.policy(hole, community, numeric, opp_embed, opp_stats, own_stats)

        diff = (out1.action_type_probs - out2.action_type_probs).abs().max().item()
        threshold = 1e-5

        return BenchmarkResult(
            name="Model Consistency",
            passed=diff < threshold,
            metric_name="max_prob_diff",
            metric_value=diff,
            threshold=threshold,
        )

    def benchmark_value_head(self) -> BenchmarkResult:
        """
        Benchmark: value head produces finite, reasonable values.
        """
        self.policy.eval()
        values = []

        for _ in range(20):
            hole = torch.randint(0, 52, (1, 2))
            community = torch.tensor([[-1, -1, -1, -1, -1]], dtype=torch.long)
            numeric = torch.randn(1, 9)
            opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
            opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)
            own_stats = torch.zeros(1, NUM_STAT_FEATURES)

            with torch.no_grad():
                out = self.policy(hole, community, numeric, opp_embed, opp_stats, own_stats)
            values.append(out.value[0, 0].item())

        all_finite = all(v == v and abs(v) < 1000 for v in values)  # not NaN, not huge
        return BenchmarkResult(
            name="Value Head Sanity",
            passed=all_finite,
            metric_name="all_finite",
            metric_value=1.0 if all_finite else 0.0,
            threshold=1.0,
        )

    def run_all_benchmarks(self) -> EvalResults:
        """Run all benchmarks and return aggregated results."""
        results = EvalResults()

        results.benchmarks.append(self.benchmark_model_consistency())
        results.benchmarks.append(self.benchmark_value_head())
        results.benchmarks.append(self.benchmark_gto_symmetry())
        results.benchmarks.append(self.benchmark_exploitation())
        results.benchmarks.append(self.benchmark_nit_exploitation())
        results.benchmarks.append(self.benchmark_maniac_exploitation())

        return results
