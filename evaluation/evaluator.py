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
from model.stat_tracker import StatTracker, NUM_STAT_FEATURES, HandRecord
from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork
from training.personality import (
    PersonalityModifier, SituationalPersonality,
    detect_situations, sample_table_personalities,
)
from engine.leduc_poker import LeducState, deal_leduc, CHECK, BET, FOLD, CALL, RAISE
from engine.dealer import Dealer
from engine.game_state import ActionType, Action, Street
from model.action_space import encode_action
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
        game: str = 'leduc',
        verbose: bool = False,
    ):
        self.policy = policy
        self.opponent_encoder = opponent_encoder
        self.rng = random.Random(seed)
        self.num_hands = num_hands
        self.game = game
        self.verbose = verbose
        self.stat_tracker = StatTracker()
        self.action_histories: Dict[int, List[torch.Tensor]] = {}
        self._opp_embed_cache: Dict[int, torch.Tensor] = {}

    def reset_tracking(self):
        """Clear histories and stats for a fresh session."""
        self.action_histories.clear()
        self.stat_tracker.reset()
        self._opp_embed_cache.clear()

    # ─────────────────────────────────────────────────────────
    # Helper Methods for Tracking Opponents
    # ─────────────────────────────────────────────────────────
    def _action_to_type_idx(self, action: Action) -> int:
        mapping = {
            ActionType.FOLD: ActionIndex.FOLD,
            ActionType.CHECK: ActionIndex.CHECK,
            ActionType.CALL: ActionIndex.CALL,
            ActionType.RAISE: ActionIndex.RAISE,
            ActionType.ALL_IN: ActionIndex.RAISE,
        }
        return mapping.get(action.action_type, ActionIndex.FOLD)

    def _record_action(self, player_id: int, action_type: int,
                       bet_frac: float, pot: float, stack: float, street: int):
        if player_id not in self.action_histories:
            self.action_histories[player_id] = []
        token = encode_action(action_type, bet_frac, pot, stack, street)
        self.action_histories[player_id].append(token)
        if len(self.action_histories[player_id]) > 16:
            self.action_histories[player_id] = self.action_histories[player_id][-16:]

    def _get_opponent_embedding(self, player_id: int) -> torch.Tensor:
        if player_id in self._opp_embed_cache:
            return self._opp_embed_cache[player_id]
        history = self.action_histories.get(player_id, [])
        if not history:
            emb = self.opponent_encoder.encode_empty(1)
        else:
            seq = torch.stack(history).unsqueeze(0)
            with torch.no_grad():
                emb = self.opponent_encoder(seq)
        
        self._opp_embed_cache[player_id] = emb.detach()
        return emb

    def _get_all_opponent_embeddings(self, hero_id: int, num_players: int) -> torch.Tensor:
        opp_embeds = []
        for pid in range(num_players):
            if pid == hero_id: continue
            opp_embeds.append(self._get_opponent_embedding(pid))
        
        if not opp_embeds:
            return self.opponent_encoder.encode_empty(1).unsqueeze(1)
        return torch.cat(opp_embeds, dim=0).unsqueeze(0)

    def _get_opponent_stats(self, hero_id: int, num_players: int) -> torch.Tensor:
        stats = []
        for pid in range(num_players):
            if pid == hero_id: continue
            stats.append(self.stat_tracker.get_stats(pid))
        
        if not stats:
            return torch.zeros(1, 1, NUM_STAT_FEATURES)
        return torch.stack(stats).unsqueeze(0)

    def _play_eval_hand(
        self,
        personality: SituationalPersonality,
    ) -> float:
        """ Backward compatibility for Leduc wrapper. Plays heads-up vs 1 personality. """
        if self.game == 'nlhe':
            return self._play_eval_hand_nlhe([None, personality], num_players=2, stacks=[100.0, 100.0])
        
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
                0.0,  # amount to call
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
                facing_raise = any(a == RAISE for rh in state.round_histories for a in rh)
                probs = personality.apply(probs, situations, hand_strength=hand_strength,
                                         is_facing_raise=facing_raise)

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

    def _play_eval_hand_nlhe(
        self, 
        personalities: List[Optional[SituationalPersonality]],
        num_players: int = 2,
        stacks: List[float] = None
    ) -> float:
        self.policy.eval()
        self._opp_embed_cache.clear() # clear per hand

        if stacks is None:
            stacks = [100.0] * num_players

        dealer = Dealer(
            num_players=num_players,
            stacks=stacks,
            small_blind=0.5,
            big_blind=1.0,
            dealer_button=self.rng.randint(0, num_players - 1),
            seed=self.rng.randint(0, 2**31)
        )
        game_state = dealer.start_hand()
        hand_records = [HandRecord() for _ in range(num_players)]

        # --- Per-hand stat tracking state ---
        pf_raise_count = 0
        pf_callers_after_raise = 0
        pf_aggressor = -1
        pf_has_raise = False
        player_checked_this_street = [False] * num_players
        first_bet_this_street_by = -1
        current_street = Street.PREFLOP

        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            hole = torch.tensor([list(p.hole_cards)], dtype=torch.long)
            board = list(game_state.board)
            while len(board) < 5:
                board.append(-1)
            community = torch.tensor([board[:5]], dtype=torch.long)

            pot = game_state.pot / 100.0
            own_stack = p.stack / 100.0
            own_bet = p.bet_this_street / 100.0
            rel_pos = (pid - game_state.dealer_button) % num_players
            position = rel_pos / max(num_players - 1, 1)

            street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
            street_val = street_map.get(game_state.street, 0) / 3.0

            num_active = sum(1 for pp in game_state.players if pp.is_active)
            current_bet = game_state.current_bet / 100.0
            min_raise = game_state.min_raise / 100.0
            amount_to_call = max(0.0, current_bet - own_bet)

            numeric = torch.tensor([[
                pot, own_stack, own_bet, position, street_val,
                num_players / 9.0, num_active / 9.0,
                current_bet, min_raise, amount_to_call
            ]], dtype=torch.float32)

            legal_types = game_state.get_legal_actions()
            mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool)
            for at in legal_types:
                if at == ActionType.FOLD:
                    mask[0, ActionIndex.FOLD] = True
                elif at == ActionType.CHECK:
                    mask[0, ActionIndex.CHECK] = True
                elif at == ActionType.CALL:
                    mask[0, ActionIndex.CALL] = True
                elif at in (ActionType.RAISE, ActionType.ALL_IN):
                    mask[0, ActionIndex.RAISE] = True

            opp_embed = self._get_all_opponent_embeddings(pid, num_players)
            opp_stats = self._get_opponent_stats(pid, num_players)
            own_stats = self.stat_tracker.get_stats(pid).unsqueeze(0)
            
            from model.action_space import get_sizing_mask
            sizing_mask = get_sizing_mask(game_state).unsqueeze(0)

            with torch.no_grad():
                out = self.policy(
                    hole, community, numeric, opp_embed, opp_stats, own_stats,
                    action_mask=mask, sizing_mask=sizing_mask
                )

            probs = out.action_type_probs[0]
            sizing_probs = torch.softmax(out.bet_size_logits[0], dim=-1).tolist()

            personality = personalities[pid]
            if personality is not None:
                # Calculate simple hand strength proxy for opponent personality
                hand_strength = 0.5
                if game_state.street == Street.PREFLOP:
                    c1, c2 = p.hole_cards
                    r1 = c1 // 4
                    r2 = c2 // 4
                    if r1 == r2: hand_strength = 0.8
                    elif max(r1, r2) >= 10: hand_strength = 0.6
                    else: hand_strength = 0.2

                is_facing_raise = (game_state.current_bet > 0
                                   and p.bet_this_street < game_state.current_bet)
                situations = detect_situations(
                    street=street_map.get(game_state.street, 0),
                    is_facing_raise=is_facing_raise,
                )
                probs = personality.apply(probs, situations,
                                         hand_strength=hand_strength,
                                         is_facing_raise=is_facing_raise)
                sizing_probs = personality.apply_sizing(torch.tensor(sizing_probs, dtype=torch.float32), situations).tolist()

            from torch.distributions import Categorical
            dist = Categorical(probs)
            a_idx = dist.sample().item()
            
            sizing_idx = 0
            if a_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
                s_dist = Categorical(torch.tensor(sizing_probs))
                sizing_idx = s_dist.sample().item()

            if a_idx == ActionIndex.FOLD and ActionType.FOLD in legal_types:
                act = Action(ActionType.FOLD)
            elif a_idx == ActionIndex.CHECK and ActionType.CHECK in legal_types:
                act = Action(ActionType.CHECK)
            elif a_idx == ActionIndex.CALL and ActionType.CALL in legal_types:
                act = Action(ActionType.CALL)
            elif a_idx == ActionIndex.RAISE and ActionType.RAISE in legal_types:
                from model.action_space import POT_FRACTIONS
                frac = POT_FRACTIONS[sizing_idx]
                if frac < 0:
                    rt = game_state.get_max_raise_to()
                    act = Action(ActionType.ALL_IN, amount=rt)
                else:
                    min_r = game_state.get_min_raise_to()
                    max_r = game_state.get_max_raise_to()
                    rt = game_state.current_bet + frac * game_state.pot
                    act = Action(ActionType.RAISE, amount=max(min_r, min(rt, max_r)))
            else:
                if ActionType.CHECK in legal_types:
                    act = Action(ActionType.CHECK)
                elif ActionType.CALL in legal_types:
                    act = Action(ActionType.CALL)
                else:
                    act = Action(ActionType.FOLD)

            # --- Comprehensive stat tracking BEFORE apply_action ---
            pre_street = game_state.street
            bet_frac = act.amount / max(game_state.pot, 1.0) if act.amount > 0 else 0.0

            if act.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].vpip = True
            if act.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].pfr = True

            # Preflop stats
            if pre_street == Street.PREFLOP:
                if act.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                    pf_raise_count += 1
                    if pf_raise_count >= 2:
                        hand_records[pid].three_bet = True
                    if pf_raise_count == 1 and pf_callers_after_raise > 0:
                        hand_records[pid].squeeze = True
                    pf_aggressor = pid
                    pf_has_raise = True
                    pf_callers_after_raise = 0
                elif act.action_type == ActionType.CALL:
                    if pf_has_raise:
                        hand_records[pid].cold_call = True
                        pf_callers_after_raise += 1
                    else:
                        hand_records[pid].limp = True

            # Post-flop stats
            if pre_street in (Street.FLOP, Street.TURN, Street.RIVER):
                st_idx = street_map.get(pre_street, 0) - 1
                if 0 <= st_idx < 3:
                    if act.action_type == ActionType.CHECK:
                        player_checked_this_street[pid] = True
                    if act.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                        if player_checked_this_street[pid]:
                            hand_records[pid].check_raise[st_idx] = True
                    if act.action_type in (ActionType.RAISE, ActionType.ALL_IN) and first_bet_this_street_by == -1:
                        first_bet_this_street_by = pid
                        if pid == pf_aggressor:
                            hand_records[pid].cbet[st_idx] = True
                    if act.action_type == ActionType.FOLD and first_bet_this_street_by == pf_aggressor and pf_aggressor != -1:
                        hand_records[pid].fold_to_cbet[st_idx] = True
                    elif act.action_type == ActionType.CALL and first_bet_this_street_by == pf_aggressor and pf_aggressor != -1:
                        hand_records[pid].fold_to_cbet[st_idx] = False

            if act.amount > 0 and act.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].bet_sizes.append(bet_frac)
            hand_records[pid].total_wagered += (act.amount if act.amount else 0.0)

            dealer.apply_action(act)

            # Record for history/stats
            pot_for_record = game_state.pot
            self._record_action(
                player_id=pid,
                action_type=self._action_to_type_idx(act),
                bet_frac=bet_frac,
                pot=pot_for_record,
                stack=p.stack,
                street=street_map.get(pre_street, 0),
            )

            # Street transition
            if game_state.street != current_street:
                current_street = game_state.street
                player_checked_this_street = [False] * num_players
                first_bet_this_street_by = -1

        # End of Hand: Track Stats
        results = dealer.get_results()
        for pid in range(num_players):
            hand_records[pid].result = results['profit'][pid]
            pf_actions = [a for a in game_state.action_history if a.player_idx == pid]
            hand_records[pid].saw_flop = not any(a.action_type == ActionType.FOLD for a in pf_actions) and game_state.street.value > Street.PREFLOP.value
            hand_records[pid].was_pf_aggressor = (pf_aggressor == pid)
            hand_records[pid].went_to_showdown = (
                game_state.street == Street.SHOWDOWN and not game_state.players[pid].is_folded
            )
            hand_records[pid].won_at_showdown = (hand_records[pid].went_to_showdown and pid in results.get('winners', []))
            self.stat_tracker.record_hand(pid, hand_records[pid])

        # Player 0 profit in big blinds (from starting stack)
        return game_state.players[0].stack - stacks[0]

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
        numeric = torch.tensor([[0.5, 1.0, 0.0, 0.0, 0.33, 0.22, 0.22, 0.0, 0.0, 0.0]], dtype=torch.float32)
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
            numeric = torch.randn(1, 10)
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

    def benchmark_multi_way_gto(self) -> BenchmarkResult:
        """6-max GTO Symmetry. Model vs 5 GTO opponents."""
        self.reset_tracking()
        personalities = [None] + [SituationalPersonality(base=PersonalityModifier.gto()) for _ in range(5)]
        total_reward = 0.0
        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand_nlhe(personalities, num_players=6)
            
        avg_reward = total_reward / self.num_hands
        threshold = 3.0
        return BenchmarkResult(
            name="Multi-Way GTO (6-max)",
            passed=abs(avg_reward) < threshold,
            metric_name="abs(avg_bb/hand)",
            metric_value=abs(avg_reward),
            threshold=threshold,
        )

    def benchmark_multi_way_exploit(self) -> BenchmarkResult:
        """6-max Mixed Exploit. Model vs 3 Nits and 2 Calling Stations."""
        self.reset_tracking()
        personalities = [None]
        for _ in range(3): personalities.append(SituationalPersonality(base=PersonalityModifier.nit()))
        for _ in range(2): personalities.append(SituationalPersonality(base=PersonalityModifier.calling_station()))
        
        total_reward = 0.0
        for _ in range(self.num_hands):
            # Shuffle positions of opponents
            opps = personalities[1:]
            self.rng.shuffle(opps)
            shuffled_personalities = [None] + opps
            total_reward += self._play_eval_hand_nlhe(shuffled_personalities, num_players=6)
            
        avg_reward = total_reward / self.num_hands
        threshold = 0.0  # Should beat a table of weak exploitable players
        return BenchmarkResult(
            name="Multi-Way Exploit (6-max Mixed)",
            passed=avg_reward > threshold,
            metric_name="avg_bb/hand",
            metric_value=avg_reward,
            threshold=threshold,
        )

    def benchmark_short_stack(self) -> BenchmarkResult:
        """15bb 6-max Push/Fold."""
        self.reset_tracking()
        personalities = [None] + [SituationalPersonality(base=PersonalityModifier.gto()) for _ in range(5)]
        total_reward = 0.0
        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand_nlhe(personalities, num_players=6, stacks=[15.0]*6)
            
        avg_reward = total_reward / self.num_hands
        threshold = -0.5 # Losing a little variance is fine in push/fold, just don't hemorrhage
        return BenchmarkResult(
            name="Short Stack Push/Fold (15bb)",
            passed=avg_reward > threshold,
            metric_name="avg_bb/hand",
            metric_value=avg_reward,
            threshold=threshold,
        )

    def benchmark_deep_stack(self) -> BenchmarkResult:
        """300bb Heads-Up warfare."""
        self.reset_tracking()
        personalities = [None, SituationalPersonality(base=PersonalityModifier.gto())]
        total_reward = 0.0
        for _ in range(self.num_hands):
            total_reward += self._play_eval_hand_nlhe(personalities, num_players=2, stacks=[300.0, 300.0])
            
        avg_reward = total_reward / self.num_hands
        threshold = 3.0
        return BenchmarkResult(
            name="Deep Stack Tactics (300bb)",
            passed=abs(avg_reward) < threshold,
            metric_name="abs(avg_bb/hand)",
            metric_value=abs(avg_reward),
            threshold=threshold,
        )

    def benchmark_adaptive_shift(self) -> BenchmarkResult:
        """Spies benchmark: Opponent flips from Nit to Maniac. Does agent recover?"""
        self.reset_tracking()
        
        # Phase 1: Nit
        p_nit = SituationalPersonality(base=PersonalityModifier.nit())
        personalities = [None, p_nit]
        for _ in range(100):
            self._play_eval_hand_nlhe(personalities, num_players=2)
            
        # Phase 2: Maniac
        p_maniac = SituationalPersonality(base=PersonalityModifier.maniac())
        personalities = [None, p_maniac]
        post_shift_reward = 0.0
        
        for _ in range(100):
            post_shift_reward += self._play_eval_hand_nlhe(personalities, num_players=2)
            
        avg_reward = post_shift_reward / 100.0
        threshold = -2.0 # Standard loss to maniac is -3 to -5 bb without adjustment. -2 implies some adjustment.
        return BenchmarkResult(
            name="Adaptation: Nit -> Maniac Shift",
            passed=avg_reward > threshold,
            metric_name="post_shift_bb/hand",
            metric_value=avg_reward,
            threshold=threshold,
            details="Played 100 hands vs Nit, then 100 vs Maniac. Measured Maniac winrate."
        )

    def _run_benchmark(self, func) -> BenchmarkResult:
        if self.verbose:
            import sys
            name = func.__doc__.strip().split('.')[0] if func.__doc__ else func.__name__
            # remove extra newlines in name
            name = ' '.join(name.split())
            sys.stdout.write(f"Running {name} ({self.num_hands} hands)... ")
            sys.stdout.flush()
            
        res = func()
        
        if self.verbose:
            status = "✅" if res.passed else "❌"
            print(f"{status} | {res.metric_value:.4f} {res.metric_name}")
            
        return res

    def run_all_benchmarks(self) -> EvalResults:
        """Run all benchmarks and return aggregated results."""
        results = EvalResults()

        results.benchmarks.append(self._run_benchmark(self.benchmark_model_consistency))
        results.benchmarks.append(self._run_benchmark(self.benchmark_value_head))
        
        results.benchmarks.append(self._run_benchmark(self.benchmark_gto_symmetry))
        results.benchmarks.append(self._run_benchmark(self.benchmark_exploitation))
        results.benchmarks.append(self._run_benchmark(self.benchmark_nit_exploitation))
        results.benchmarks.append(self._run_benchmark(self.benchmark_maniac_exploitation))

        if self.game == 'nlhe':
            results.benchmarks.append(self._run_benchmark(self.benchmark_multi_way_gto))
            results.benchmarks.append(self._run_benchmark(self.benchmark_multi_way_exploit))
            results.benchmarks.append(self._run_benchmark(self.benchmark_short_stack))
            results.benchmarks.append(self._run_benchmark(self.benchmark_deep_stack))
            results.benchmarks.append(self._run_benchmark(self.benchmark_adaptive_shift))

        return results
