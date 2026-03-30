"""
NLHE Self-Play Trainer — full No-Limit Hold'em training loop.

Improvements over basic trainer:
1. Real opponent embeddings (tracks action history across hands)
2. Auto GPU/MPS/CPU device placement
3. Search-guided expert iteration (optional System 2 during training)

Each hand randomly samples:
- Number of players (min_players to max_players)
- Per-player stack depth (min_bb to max_bb, independently)
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

from engine.game_state import GameState, Action, ActionType, Street
from engine.dealer import Dealer
from model.action_space import (
    ActionIndex, ActionOutput, NUM_ACTION_TYPES,
    ACTION_FEATURE_DIM, encode_action, POT_FRACTIONS,
)
from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork, OPP_GAME_STATE_DIM, PROFILE_DIM, MAX_HAND_ACTIONS
from model.stat_tracker import StatTracker, HandRecord, NUM_STAT_FEATURES
from model.nlhe_encoder import NLHEEncoder
from training.opponent_pool import OpponentPool
from training.ppo_updater import compute_gae, precompute_ppo_data, compute_ppo_loss, count_action_distribution
from training.state_encoder import encode_state, encode_action_mask, get_opponent_stats, get_opponent_game_state, compute_hero_ev


# ─────────────────────────────────────────────────────────────
# Experience tuple
# ─────────────────────────────────────────────────────────────

class Experience(NamedTuple):
    """One decision point during a hand."""
    hole_cards: torch.Tensor          # (1, 2)
    community_cards: torch.Tensor     # (1, 5)
    numeric_features: torch.Tensor    # (1, 9)
    opponent_embeddings: torch.Tensor # (1, num_opp, embed_dim)
    opponent_stats: torch.Tensor      # (1, num_opp, stat_dim)
    own_stats: torch.Tensor           # (1, stat_dim)
    opponent_game_state: torch.Tensor # (1, num_opp, 14)
    hand_action_seq: torch.Tensor     # (1, max_seq, 13) raw action tokens
    hand_action_len: torch.Tensor     # (1,) actual sequence length
    actor_profiles_seq: torch.Tensor  # (1, max_seq, 64) per-action actor profile
    hero_profile: torch.Tensor        # (1, 64) cached hero profile
    opponent_profiles: torch.Tensor   # (1, num_opp, 64) cached opp profiles
    action_mask: torch.Tensor         # (1, 4)
    sizing_mask: torch.Tensor         # (1, 10)
    action_idx: int                   # chosen action
    sizing_idx: int                   # chosen sizing bucket (0 if not raise)
    log_prob: float                   # COMBINED log prob (action + sizing) for backward compat
    value: float                      # value estimate
    reward: float                     # final reward (set after hand ends)
    hand_id: int = 0                  # groups decisions into per-hand trajectories
    step_idx: int = 0                 # position within trajectory
    hero_stack_bb: float = 0.0        # hero's starting stack in bb (for deep all-in tracking)
    effective_stack_bb: float = 1.5   # min(hero_stack, max(opp_stacks)) at hero's first decision; floor=1.5bb
    action_log_prob: float = 0.0      # action-only log prob (for decoupled PPO)
    sizing_log_prob: float = 0.0      # sizing-only log prob (for decoupled PPO, 0 if not raise)
    equity_x_pot: float = 0.0        # side-pot-aware hero EV at decision point
    end_street_equity_x_pot: float = 0.0  # hero EV at end of this street
    street_idx: int = 0              # 0=preflop, 1=flop, 2=turn, 3=river
    v_res_end_of_street: float = 0.0 # V_res at end of street (before new card)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class NLHETrainingConfig:
    """Configuration for NLHE training."""
    # Architecture
    embed_dim: int = 64
    opponent_embed_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2

    # Game setup — ranges for per-hand randomization
    min_players: int = 2
    max_players: int = 6
    min_bb: int = 20
    max_bb: int = 200
    small_blind: float = 0.5
    big_blind: float = 1.0
    uniform_stacks: bool = False

    # Backwards compat — non-zero overrides ranges
    num_players: int = 0
    starting_bb: int = 0

    # Training
    lr: float = 3e-4
    gamma: float = 1.0           # no temporal discount — poker hands are finite episodes
    gae_lambda: float = 0.95     # GAE lambda parameter
    v_res_alpha: float = 1.0     # scales V_res influence on advantages (0=pure equity, 1=full V_res)
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    mini_batch_size: int = 64          # mini-batch size for PPO updates
    entropy_coef: float = 0.005   # was 0.05 — reduced 10x to not drown out policy gradient
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5

    # Self-play
    hands_per_epoch: int = 512   # was 128 — increased for hero-only training
    batch_chunk_size: int = 500  # Max simultaneous games per sub-batch
    num_workers: int = 0         # Number of Gym processes. 0 = sequential python

    # Frozen opponent pool
    frozen_update_interval: int = 20   # sync frozen opponent every N epochs
    max_recent_pool: int = 10           # recent checkpoints (FIFO)
    max_archive_pool: int = 5           # old checkpoints (random replacement)

    # Epsilon-greedy exploration (training only, annealed)
    epsilon: float = 0.15              # probability of random action (start)
    epsilon_end: float = 0.08          # probability of random action (end) — poker needs permanent mixing

    # Opponent modeling
    history_reset_interval: Tuple[int, int] = (300, 500)



    # Equity-based reward decomposition
    mc_equity_sims: int = 500         # MC runouts per equity computation during training

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Logging
    log_interval: int = 10
    verbose: bool = False

    # Seed
    seed: int = 42

    # PPO modifications
    remove_clip: bool = False     # Use KL Penalty instead of hard clipping
    kl_beta: float = 1.0          # Coefficient for KL penalty when clip is removed




@dataclass
class TableState:
    """Persistent state for a single table across multiple hands."""
    action_histories: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    stat_tracker: StatTracker = field(default_factory=StatTracker)
    hands_since_reset: int = 0
    next_reset_at: int = 50

    seat_pool_idx: Dict[int, int] = field(default_factory=dict)
    opp_embed_cache: Dict[int, torch.Tensor] = field(default_factory=dict)


class NLHESelfPlayTrainer:
    """Self-play RL trainer for NLHE."""

    def __init__(self, config: NLHETrainingConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)

        # ── Device ────────────────────────────────────────────
        self.device = self._resolve_device(self.config.device)

        # ── Live models (trained via PPO) ─────────────────────
        self.opponent_encoder = OpponentEncoder(
            embed_dim=self.config.opponent_embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
        ).to(self.device)

        self.policy = PolicyNetwork(
            embed_dim=self.config.embed_dim,
            opponent_embed_dim=self.config.opponent_embed_dim,
            num_cross_attn_heads=self.config.num_heads,
            num_cross_attn_layers=self.config.num_layers,
        ).to(self.device)

        all_params = list(self.policy.parameters()) + list(self.opponent_encoder.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.lr)

        # ── Opponent pool ─────────────────────────────────────
        self.pool = OpponentPool(
            embed_dim=self.config.embed_dim,
            opponent_embed_dim=self.config.opponent_embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_recent=self.config.max_recent_pool,
            max_archive=self.config.max_archive_pool,
            rng=self.rng,
        )
        # Seed pool with initial weights
        self.pool.sync(self.policy.state_dict(), self.opponent_encoder.state_dict())

        # Epoch tracking
        self.current_epoch = 0
        self.total_epochs = 1

        # ── Hand counter for unique hand IDs ──────────────────
        self._hand_counter = 0

        # ── Metrics ───────────────────────────────────────────
        self.epoch_rewards: List[float] = []
        self.epoch_losses: List[float] = []

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Auto-detect best device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device_str)

    def _to(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to training device."""
        return tensor.to(self.device)

    def _sync_frozen(self):
        """Sync current live weights to opponent pool."""
        self.pool.sync(self.policy.state_dict(), self.opponent_encoder.state_dict())

    def save_pool(self, save_dir: str):
        """Save opponent pool to disk."""
        self.pool.save(save_dir)

    def load_pool(self, load_dir: str):
        """Load opponent pool from disk."""
        self.pool.load(load_dir)

    def _get_epsilon(self) -> float:
        """Get the annealed epsilon for the current epoch."""
        c = self.config
        progress = min(self.current_epoch / max(self.total_epochs, 1), 1.0)
        return c.epsilon + (c.epsilon_end - c.epsilon) * progress

    def _get_entropy_coef(self) -> float:
        """Get the annealed entropy coefficient for the current epoch."""
        c = self.config
        progress = min(self.current_epoch / max(self.total_epochs, 1), 1.0)
        return c.entropy_coef + (c.entropy_coef_end - c.entropy_coef) * progress


    def _maybe_reset_histories(self, table: 'TableState'):
        """Periodically reset opponent histories (simulate new table)."""
        table.hands_since_reset += 1

        # Hard reset: total flush of histories and stats
        if table.hands_since_reset >= table.next_reset_at:
            table.action_histories.clear()
            table.stat_tracker.reset()
            table.hands_since_reset = 0
            table.next_reset_at = self.rng.randint(*self.config.history_reset_interval)
            table.seat_pool_idx = {}

    # ─────────────────────────────────────────────────────────
    # Table sampling
    # ─────────────────────────────────────────────────────────

    def _sample_table(self) -> Tuple[int, List[float]]:
        """Sample random player count + per-player stacks with realistic distributions."""
        c = self.config

        # Player count: 90% common sizes (2, 6, 9), 10% uniform random
        if c.num_players > 0:
            num_p = c.num_players
        elif self.rng.random() < 0.9:
            num_p = self.rng.choice([2, 6, 9])
            num_p = max(c.min_players, min(num_p, c.max_players))
        else:
            num_p = self.rng.randint(c.min_players, c.max_players)

        # Stack depth: 70% → 100bb, 20% → random in range, 10% → short (10-50bb)
        if c.starting_bb > 0:
            stacks = [c.starting_bb * c.big_blind] * num_p
        else:
            def _sample_bb():
                roll = self.rng.random()
                if roll < 0.70:
                    return 100  # Standard 100BB
                elif roll < 0.90:
                    return self.rng.randint(c.min_bb, c.max_bb)
                else:
                    return self.rng.randint(max(c.min_bb, 10), min(c.max_bb, 50))

            if c.uniform_stacks:
                bb = _sample_bb()
                stacks = [bb * c.big_blind] * num_p
            else:
                stacks = [_sample_bb() * c.big_blind for _ in range(num_p)]

        return num_p, stacks

    # ─────────────────────────────────────────────────────────
    # Opponent history tracking
    # ─────────────────────────────────────────────────────────

    def _record_action(self, table: TableState, player_id: int, action_type: int,
                       bet_frac: float, pot: float, stack: float, street: int,
                       relative_position: float = 0.0, hand_boundary: float = 0.0):
        """Record an observed action for a player."""
        if player_id not in table.action_histories:
            table.action_histories[player_id] = []

        token = encode_action(action_type, bet_frac, pot, stack, street,
                              relative_position=relative_position,
                              hand_boundary=hand_boundary)
        table.action_histories[player_id].append(token)

        # Cap at 16 actions (rolling window — last ~4 hands of context)
        if len(table.action_histories[player_id]) > 16:
            table.action_histories[player_id] = table.action_histories[player_id][-16:]





    def _decode_action(self, action_idx: int, sizing_idx: int,
                       game_state: GameState) -> Action:
        """Convert model output to engine Action."""
        legal = game_state.get_legal_actions()

        if action_idx == ActionIndex.FOLD and ActionType.FOLD in legal:
            return Action(ActionType.FOLD)
        elif action_idx == ActionIndex.CHECK and ActionType.CHECK in legal:
            return Action(ActionType.CHECK)
        elif action_idx == ActionIndex.CALL and ActionType.CALL in legal:
            return Action(ActionType.CALL)
        elif action_idx == ActionIndex.RAISE and ActionType.RAISE in legal:
            from model.action_space import POT_FRACTIONS
            frac = POT_FRACTIONS[sizing_idx]
            if frac < 0:
                return Action(ActionType.ALL_IN, amount=game_state.get_max_raise_to())
            min_r = game_state.get_min_raise_to()
            max_r = game_state.get_max_raise_to()
            raise_to = frac * game_state.pot
            return Action(ActionType.RAISE, amount=max(min_r, min(raise_to, max_r)))

        # Fallback
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK)
        if ActionType.CALL in legal:
            return Action(ActionType.CALL)
        return Action(ActionType.FOLD)

    # ─────────────────────────────────────────────────────────
    # Hand simulation
    # ─────────────────────────────────────────────────────────

    def _action_to_type_idx(self, action: Action) -> int:
        """Map engine action to model action index."""
        mapping = {
            ActionType.FOLD: ActionIndex.FOLD,
            ActionType.CHECK: ActionIndex.CHECK,
            ActionType.CALL: ActionIndex.CALL,
            ActionType.RAISE: ActionIndex.RAISE,
            ActionType.ALL_IN: ActionIndex.RAISE,
        }
        return mapping.get(action.action_type, ActionIndex.FOLD)

    def _play_hand_gen(self, table: TableState) -> Generator[dict, Tuple[torch.Tensor, float, list], Tuple[List[dict], float]]:
        num_p, stacks = self._sample_table()
        table.opp_embed_cache.clear()  # Clear cache for new hand

        hand_id = self._hand_counter
        self._hand_counter += 1

        dealer = Dealer(
            num_players=num_p,
            stacks=stacks,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            dealer_button=self.rng.randint(0, num_p - 1),
            seed=self.rng.randint(0, 2**31),
        )

        game_state = dealer.start_hand()
        hero_experiences: List[dict] = []
        hero_step_idx = 0
        hero_effective_stack_bb = 1.5  # computed at hero's first decision

        # Per-hand runout cache: simulate once per street, reuse across actions
        from engine.hand_evaluator import RunoutCache
        equity_cache = RunoutCache()

        # Per-seat opponent assignment
        if not table.seat_pool_idx or len(table.seat_pool_idx) != num_p - 1:
            table.seat_pool_idx = {}
            combined_pool = self.pool.get_combined()
            for pid in range(1, num_p):
                if combined_pool:
                    table.seat_pool_idx[pid] = self.rng.randint(0, len(combined_pool) - 1)
                else:
                    table.seat_pool_idx[pid] = 0



        hand_records = [HandRecord() for _ in range(num_p)]
        epsilon = self._get_epsilon()

        # --- Per-hand stat tracking state ---
        pf_raise_count = 0           # number of raises preflop (for 3bet detection)
        pf_callers_after_raise = 0   # callers after first raise (for squeeze detection)
        pf_aggressor = -1            # seat of last preflop raiser
        pf_has_raise = False         # has there been a preflop raise?
        player_checked_this_street = [False] * num_p  # per-player check tracking
        first_bet_this_street_by = -1  # who made the first bet/raise this street
        current_street = Street.PREFLOP
        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3, Street.SHOWDOWN: 3}
        preflop_folders = set()  # players who folded preflop (for saw_flop)

        # Phase 5: Accumulate all raw action tokens during this hand
        hand_action_tokens = []  # list of (raw_token_13d, actor_pid)

        # Phase 6: Pre-compute 64d architectural profiles for the entire hand
        with torch.no_grad():
            embs = []
            stats_list = []
            heroes = []
            
            for p_idx in range(num_p):
                heroes.append([1.0] if p_idx == 0 else [0.0])
                
                # Retrieve the historical opponent trace
                emb = table.opp_embed_cache.get(p_idx, None)
                if emb is None:
                    hist = table.action_histories.get(p_idx, [])
                    if len(hist) > 0:
                        seq = torch.stack(hist)[-self.opponent_encoder.max_seq_len:]
                        enc = self.opponent_encoder(seq.unsqueeze(0).to(self.device))
                        emb = enc.detach()
                        table.opp_embed_cache[p_idx] = emb
                    else:
                        emb = torch.zeros(1, self.config.opponent_embed_dim, device=self.device)
                else:
                    emb = emb.to(self.device).clone().detach()
                
                embs.append(emb.squeeze(0))
                stats_list.append(table.stat_tracker.get_stats(p_idx).to(self.device).squeeze(0))
                
            # Execute impressively parallel GPU kernels for both perspectives
            embs_t = torch.stack(embs, dim=0) # (num_p, 128)
            stats_t = torch.stack(stats_list, dim=0) # (num_p, 30)
            
            heroes_ones = torch.ones(num_p, 1, device=self.device)
            heroes_zeros = torch.zeros(num_p, 1, device=self.device)
            
            profiles_as_hero = self.policy.profile_builder(embs_t, stats_t, heroes_ones) # (num_p, 64)
            profiles_as_opp = self.policy.profile_builder(embs_t, stats_t, heroes_zeros) # (num_p, 64)

        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            if not p.is_active:
                break

            encoded = encode_state(game_state, pid, self.device)
            opp_stats = get_opponent_stats(table.stat_tracker, pid, num_p, self.device)
            own_stats = self._to(table.stat_tracker.get_stats(pid).unsqueeze(0))
            opp_game_state = get_opponent_game_state(game_state, pid, num_p, self.device)

            opp_ids = [opid for opid in range(num_p) if opid != pid]
            uncached = []
            cached = {}
            for opid in opp_ids:
                if opid in table.opp_embed_cache:
                    cached[opid] = table.opp_embed_cache[opid]
                else:
                    uncached.append((opid, table.action_histories.get(opid, [])))

            # Phase 5: Snapshot current hand action history for this decision point
            n_actions = len(hand_action_tokens)
            ha_seq = torch.zeros(1, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)
            ha_len = torch.tensor([min(n_actions, MAX_HAND_ACTIONS)], dtype=torch.long)
            if n_actions > 0:
                tokens = [t for t, _ in hand_action_tokens[-MAX_HAND_ACTIONS:]]
                ha_seq[0, :len(tokens)] = torch.stack(tokens)

            # Construct subjective historical actor profiles
            actor_profiles_seq = torch.zeros(1, MAX_HAND_ACTIONS, PROFILE_DIM, device=self.device)
            for i, (_, apid) in enumerate(hand_action_tokens[-MAX_HAND_ACTIONS:]):
                actor_profiles_seq[0, i] = profiles_as_hero[apid] if apid == pid else profiles_as_opp[apid]

            # Yield for batched inference (both hero and frozen opponents)
            probs, value, sizing_probs, opp_embed_tensor = yield {
                'hole_cards': encoded['hole_cards'],
                'community_cards': encoded['community_cards'],
                'numeric_features': encoded['numeric_features'],
                'opponent_stats': opp_stats,
                'own_stats': own_stats,
                'opponent_game_state': opp_game_state,
                'hand_action_seq': ha_seq,
                'hand_action_len': ha_len,
                'actor_profiles_seq': actor_profiles_seq,
                'hero_profile': profiles_as_hero[pid],
                'opponent_profiles': profiles_as_opp[opp_ids],
                '_hand_action_pids': [apid for _, apid in hand_action_tokens[-MAX_HAND_ACTIONS:]],
                'action_mask': encoded['action_mask'],
                'sizing_mask': encoded['sizing_mask'],
                '_is_hero': pid == 0,
                '_pool_idx': table.seat_pool_idx.get(pid, 0) if pid != 0 else 0,
                '_opp_ids': opp_ids,
                '_uncached_opp_histories': uncached,
                '_cached_embeds': cached,
            }

            if pid == 0:
                # ── HERO ──
                # Compute side-pot-aware hero EV before action
                equity_x_pot = compute_hero_ev(game_state, hero_idx=0, mc_sims=self.config.mc_equity_sims, runout_cache=equity_cache)

                # Compute effective stack at hero's first decision
                if hero_step_idx == 0:
                    bb = max(self.config.big_blind, 1.0)
                    hero_stack_bb = p.stack / bb
                    opp_stacks_bb = [
                        game_state.players[opid].stack / bb
                        for opid in range(num_p)
                        if opid != 0 and not game_state.players[opid].is_folded
                    ]
                    max_opp = max(opp_stacks_bb) if opp_stacks_bb else 0.0
                    hero_effective_stack_bb = max(min(hero_stack_bb, max_opp), 1.5)

                raw_probs = probs.clone()

                action_mask_cpu = encoded['action_mask'].squeeze(0).cpu()
                legal_indices = action_mask_cpu.nonzero(as_tuple=True)[0].tolist()

                # 1. Sample action
                if self.rng.random() < epsilon and len(legal_indices) > 1:
                    action_idx = self.rng.choice(legal_indices)
                else:
                    dist = Categorical(probs)
                    action_idx = dist.sample().item()

                # 2. Compute exact behavior distribution log-prob to prevent PPO ratio explosion
                behavior_probs = probs.clone().cpu()
                if len(legal_indices) > 1:
                    behavior_probs = behavior_probs * (1.0 - epsilon)
                    for li in legal_indices:
                        behavior_probs[li] += epsilon / len(legal_indices)
                
                behavior_dist = Categorical(behavior_probs)
                action_lp = behavior_dist.log_prob(torch.tensor(action_idx)).item()
                sizing_lp = 0.0

                sizing_idx = 0
                if action_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
                    sizing_tensor = torch.tensor(sizing_probs, dtype=torch.float32)
                    sizing_mask_1d = encoded['sizing_mask'].squeeze(0).cpu()
                    legal_sizing = sizing_mask_1d.nonzero(as_tuple=True)[0].tolist()

                    if self.rng.random() < epsilon and len(legal_sizing) > 1:
                        sizing_idx = self.rng.choice(legal_sizing)
                    else:
                        s_dist = Categorical(sizing_tensor)
                        sizing_idx = s_dist.sample().item()

                    # Exact sizing behavior distribution log-prob
                    behavior_sizing = sizing_tensor.clone().cpu()
                    if len(legal_sizing) > 1:
                        behavior_sizing = behavior_sizing * (1.0 - epsilon)
                        for si in legal_sizing:
                            behavior_sizing[si] += epsilon / len(legal_sizing)
                    
                    s_dist_behavior = Categorical(behavior_sizing)
                    sizing_lp = s_dist_behavior.log_prob(torch.tensor(sizing_idx)).item()

                log_prob = action_lp + sizing_lp

                hero_experiences.append({
                    'hole_cards': encoded['hole_cards'].detach().cpu(),
                    'community_cards': encoded['community_cards'].detach().cpu(),
                    'numeric_features': encoded['numeric_features'].detach().cpu(),
                    'opponent_embeddings': opp_embed_tensor.detach().cpu(),
                    'opponent_stats': opp_stats.detach().cpu(),
                    'own_stats': own_stats.detach().cpu(),
                    'opponent_game_state': opp_game_state.detach().cpu(),
                    'hand_action_seq': ha_seq.detach().cpu(),
                    'hand_action_len': ha_len.detach().cpu(),
                    'actor_profiles_seq': actor_profiles_seq.detach().cpu(),
                    'hero_profile': profiles_as_hero[0].detach().cpu(),
                    'opponent_profiles': profiles_as_opp[opp_ids].detach().cpu(),
                    '_hand_action_pids': [apid for _, apid in hand_action_tokens[-MAX_HAND_ACTIONS:]],
                    'action_mask': encoded['action_mask'].detach().cpu(),
                    'sizing_mask': encoded['sizing_mask'].detach().cpu(),
                    'action_idx': action_idx,
                    'sizing_idx': sizing_idx,
                    'log_prob': log_prob,
                    'action_log_prob': action_lp,
                    'sizing_log_prob': sizing_lp,
                    'value': value,
                    'step_idx': hero_step_idx,
                    'equity_x_pot': equity_x_pot,
                    'end_street_equity_x_pot': equity_x_pot,  # default, overwritten at street end
                    'street_idx': street_map.get(game_state.street, 0),
                })
                hero_step_idx += 1

            else:
                # ── OPPONENT ──
                dist = Categorical(probs)
                action_idx = dist.sample().item()
                sizing_idx = 0
                if action_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
                    sizing_tensor = torch.tensor(sizing_probs, dtype=torch.float32)
                    s_dist = Categorical(sizing_tensor)
                    sizing_idx = s_dist.sample().item()

            # Execute action
            action = self._decode_action(action_idx, sizing_idx, game_state)

            # --- Comprehensive stat tracking BEFORE apply_action ---
            street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
            pre_street = game_state.street
            bet_frac = action.amount / max(game_state.pot, 1e-6) if action.amount else 0.0
            pot_for_record = game_state.pot / max(game_state.big_blind, 1.0)

            # Track VPIP / PFR (preflop only)
            if pre_street == Street.PREFLOP:
                if action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
                    hand_records[pid].vpip = True
                if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                    hand_records[pid].pfr = True

            # --- Preflop-specific stats ---
            if pre_street == Street.PREFLOP:
                if action.action_type == ActionType.FOLD:
                    preflop_folders.add(pid)
                elif action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                    pf_raise_count += 1
                    if pf_raise_count >= 2:
                        hand_records[pid].three_bet = True  # 3-bet (or 4-bet+)
                    if pf_raise_count == 1 and pf_callers_after_raise > 0:
                        hand_records[pid].squeeze = True  # raise after raise+call
                    pf_aggressor = pid
                    pf_has_raise = True
                    pf_callers_after_raise = 0  # reset caller count
                elif action.action_type == ActionType.CALL:
                    if pf_has_raise:
                        hand_records[pid].cold_call = True
                        pf_callers_after_raise += 1
                    else:
                        hand_records[pid].limp = True  # call BB with no raise

            # --- Post-flop stats ---
            if pre_street in (Street.FLOP, Street.TURN, Street.RIVER):
                st_idx = street_map.get(pre_street, 0) - 1  # 0=flop, 1=turn, 2=river
                if 0 <= st_idx < 3:
                    if action.action_type == ActionType.CHECK:
                        player_checked_this_street[pid] = True

                    if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                        # Check-raise: player checked earlier this street, now raising
                        if player_checked_this_street[pid]:
                            hand_records[pid].check_raise[st_idx] = True

                    # C-bet: preflop aggressor makes first bet on this street
                    if action.action_type in (ActionType.RAISE, ActionType.ALL_IN) and first_bet_this_street_by == -1:
                        first_bet_this_street_by = pid
                        if pid == pf_aggressor:
                            hand_records[pid].cbet[st_idx] = True

                    # Fold to C-bet: folding when facing a cbet
                    if action.action_type == ActionType.FOLD and first_bet_this_street_by == pf_aggressor and pf_aggressor != -1:
                        hand_records[pid].fold_to_cbet[st_idx] = True
                    elif action.action_type == ActionType.CALL and first_bet_this_street_by == pf_aggressor and pf_aggressor != -1:
                        hand_records[pid].fold_to_cbet[st_idx] = False

            # Bet sizing tracking
            if action.amount > 0 and action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].bet_sizes.append(bet_frac)
            hand_records[pid].total_wagered += (action.amount if action.amount else 0.0)

            # Apply the action
            dealer.apply_action(action)

            # Compute relative position for enriched action tokens
            dealer_btn = game_state.dealer_button
            rel_pos = ((pid - dealer_btn) % num_p) / 8.0
            is_hand_boundary = 1.0 if (len(table.action_histories.get(pid, [])) == 0 or pre_street == Street.PREFLOP and pf_raise_count == 0 and pid == game_state.current_player_idx) else 0.0
            self._record_action(table, pid, self._action_to_type_idx(action), bet_frac, pot_for_record, p.stack, street_map.get(pre_street, 0),
                                relative_position=rel_pos, hand_boundary=is_hand_boundary)

            # Phase 5: Record raw action token for hand history
            raw_token = encode_action(
                self._action_to_type_idx(action), bet_frac, pot_for_record, p.stack,
                street_map.get(pre_street, 0), relative_position=rel_pos,
                hand_boundary=is_hand_boundary,
            )
            hand_action_tokens.append((raw_token, pid))

            # Detect street transitions to reset per-street state
            if game_state.street != current_street:
                # Compute end-of-street hero EV (captures opponent responses)
                if hero_experiences:
                    end_ev = compute_hero_ev(game_state, hero_idx=0, mc_sims=self.config.mc_equity_sims, runout_cache=equity_cache)
                    hero_experiences[-1]['end_street_equity_x_pot'] = end_ev

                    # Compute V_res at end of street (BEFORE new card)
                    # Value-only forward pass — we only use the value head output
                    last_exp = hero_experiences[-1]
                    end_encoded = encode_state(game_state, 0, self.device)
                    end_opp_stats = get_opponent_stats(table.stat_tracker, 0, num_p, self.device)
                    end_own_stats = self._to(table.stat_tracker.get_stats(0).unsqueeze(0))
                    end_opp_gs = get_opponent_game_state(game_state, 0, num_p, self.device)

                    # Reuse tensors from the last hero experience, ensuring correct shapes
                    opp_embed_t = last_exp.get('_opp_embed_tensor', torch.zeros(1, 1, self.config.opponent_embed_dim, device=self.device))
                    if opp_embed_t.dim() == 2:
                        opp_embed_t = opp_embed_t.unsqueeze(0)
                    opp_prof_t = last_exp.get('opponent_profiles', torch.zeros(1, PROFILE_DIM, device=self.device))
                    if opp_prof_t.dim() == 2:
                        opp_prof_t = opp_prof_t.unsqueeze(0)
                    hero_prof_t = last_exp.get('hero_profile', torch.zeros(PROFILE_DIM, device=self.device))
                    if hero_prof_t.dim() == 1:
                        hero_prof_t = hero_prof_t.unsqueeze(0)

                    with torch.no_grad():
                        end_out = self.policy(
                            hole_cards=end_encoded['hole_cards'],
                            community_cards=end_encoded['community_cards'],
                            numeric_features=end_encoded['numeric_features'],
                            opponent_embeddings=opp_embed_t,
                            opponent_stats=end_opp_stats,
                            own_stats=end_own_stats,
                            opponent_game_state=end_opp_gs,
                            action_mask=end_encoded['action_mask'],
                            sizing_mask=last_exp.get('sizing_mask', torch.ones(1, 10, dtype=torch.bool)),
                            hand_action_seq=last_exp.get('hand_action_seq', torch.zeros(1, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)),
                            hand_action_len=last_exp.get('hand_action_len', torch.zeros(1, dtype=torch.long)),
                            actor_profiles_seq=last_exp.get('actor_profiles_seq', torch.zeros(1, MAX_HAND_ACTIONS, PROFILE_DIM, device=self.device)),
                            hero_profile=hero_prof_t,
                            opponent_profiles=opp_prof_t,
                        )
                    hero_experiences[-1]['v_res_end_of_street'] = end_out.value[0, 0].item()

                current_street = game_state.street
                player_checked_this_street = [False] * num_p
                first_bet_this_street_by = -1

        # --- End of hand: populate remaining fields ---
        results = dealer.get_results()
        profits = results['profit']
        
        # The mathematically extracted EV profit computed out of Eval7
        # Falls back to regular profit if the hand ended in a fold.
        ev_profits = results.get('ev_profit', profits)

        for pid in range(num_p):
            hand_records[pid].result = profits[pid]
            # saw_flop: player didn't fold preflop and hand went past preflop
            hand_records[pid].saw_flop = (pid not in preflop_folders) and game_state.street.value > Street.PREFLOP.value
            hand_records[pid].was_pf_aggressor = (pf_aggressor == pid)
            hand_records[pid].went_to_showdown = (game_state.street == Street.SHOWDOWN and not game_state.players[pid].is_folded)
            hand_records[pid].won_at_showdown = (hand_records[pid].went_to_showdown and pid in (results.get('winners', [])))
            table.stat_tracker.record_hand(pid, hand_records[pid])

        self._maybe_reset_histories(table)

        # Train PPO purely on the Expected Value mathematically calculated at showdown!
        hero_reward_bb = ev_profits[0] / max(self.config.big_blind, 1.0)
        # Normalize reward by effective stack: puts everything in "fraction of stack" units
        # so V(s), advantages, and returns are all on the same ~0-1 scale
        hero_reward = hero_reward_bb / max(hero_effective_stack_bb, 1.5)
        hero_exp_list = []
        for exp_dict in hero_experiences:
            hero_exp_list.append(Experience(
                hole_cards=exp_dict['hole_cards'],
                community_cards=exp_dict['community_cards'],
                numeric_features=exp_dict['numeric_features'],
                opponent_embeddings=exp_dict['opponent_embeddings'],
                opponent_stats=exp_dict['opponent_stats'],
                own_stats=exp_dict['own_stats'],
                opponent_game_state=exp_dict['opponent_game_state'],
                hand_action_seq=exp_dict['hand_action_seq'],
                hand_action_len=exp_dict['hand_action_len'],
                actor_profiles_seq=exp_dict['actor_profiles_seq'],
                hero_profile=exp_dict['hero_profile'],
                opponent_profiles=exp_dict['opponent_profiles'],
                action_mask=exp_dict['action_mask'],
                sizing_mask=exp_dict['sizing_mask'],
                action_idx=exp_dict['action_idx'],
                sizing_idx=exp_dict['sizing_idx'],
                log_prob=exp_dict['log_prob'],
                value=exp_dict['value'],
                reward=hero_reward,
                hand_id=hand_id,
                step_idx=exp_dict['step_idx'],
                hero_stack_bb=stacks[0] / self.config.big_blind,
                effective_stack_bb=hero_effective_stack_bb,
                action_log_prob=exp_dict['action_log_prob'],
                sizing_log_prob=exp_dict['sizing_log_prob'],
                equity_x_pot=exp_dict['equity_x_pot'] / max(self.config.big_blind, 1.0) / max(hero_effective_stack_bb, 1.5),
                end_street_equity_x_pot=exp_dict['end_street_equity_x_pot'] / max(self.config.big_blind, 1.0) / max(hero_effective_stack_bb, 1.5),
                street_idx=exp_dict['street_idx'],
                v_res_end_of_street=exp_dict.get('v_res_end_of_street', 0.0),
            ))

        return [hero_exp_list] + [[] for _ in range(num_p - 1)], hero_reward_bb

    def _run_batched_epoch(self) -> Tuple[List[Experience], float]:
        self.policy.eval()
        self.opponent_encoder.eval()

        num_hands = self.config.hands_per_epoch
        all_exp: List[Experience] = []
        total_reward = 0.0
        total_finished = 0

        chunk_size = min(self.config.batch_chunk_size, num_hands)
        epoch_start = time.time()

        # CREATE PERSISTENT TABLES FOR THE BATCH
        tables = [TableState() for _ in range(chunk_size)]

        for chunk_start in range(0, num_hands, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_hands)
            chunk_n = chunk_end - chunk_start

            # Build all frozen models needed for this table if not existing!
            for i in range(chunk_n):
                combined_pool = self.pool.get_combined()
                for pool_idx in tables[i].seat_pool_idx.values():
                    if pool_idx not in self.pool._frozen_models:
                        if combined_pool and pool_idx < len(combined_pool):
                            fm = copy.deepcopy(self.policy).to(self.device)
                            fm.load_state_dict(combined_pool[pool_idx])
                            for p in fm.parameters(): p.requires_grad = False
                            fm.eval()
                            self.pool._frozen_models[pool_idx] = fm

            games = [self._play_hand_gen(tables[i]) for i in range(chunk_n)]
            pending = {}
            for i, g in enumerate(games):
                try:
                    state = next(g)
                    pending[i] = (g, state)
                except StopIteration as e:
                    exps, reward = e.value
                    for pexp in exps: all_exp.extend(pexp)
                    total_reward += reward
                    total_finished += 1

            step_count = 0
            while pending:
                game_indices = list(pending.keys())
                states = [pending[i][1] for i in game_indices]

                # Periodic MPS sync to prevent graph cache exhaustion
                step_count += 1
                if step_count % 25 == 0 and self.device.type == 'mps':
                    torch.mps.synchronize()

                # 1. BATCH OPPONENT EMBEDDING GENERATION
                all_seqs_to_encode = []
                seq_locs = [] # (state_idx, list_of_opid_indices)
                
                # First pass: collect uncached histories
                for s_idx, s in enumerate(states):
                    for opid, history in s['_uncached_opp_histories']:
                        if history:
                            all_seqs_to_encode.append((s_idx, opid, history))
                
                new_embs_flat = []
                if all_seqs_to_encode:
                    seqs = [torch.stack(h) for _, _, h in all_seqs_to_encode]
                    orig_len = max(sq.shape[0] for sq in seqs)
                    
                    # Pad to power of 2 to avoid MPS graph leak!
                    pad_len = 1
                    while pad_len < orig_len: pad_len *= 2
                    
                    orig_b = len(seqs)
                    pad_b = ((orig_b + 31) // 32) * 32
                    
                    feat_dim = seqs[0].shape[1]
                    
                    padded = torch.zeros(pad_b, pad_len, feat_dim, device=self.device)
                    mask = torch.ones(pad_b, pad_len, device=self.device, dtype=torch.bool)
                    for i, sq in enumerate(seqs):
                        sq = self._to(sq)
                        padded[i, :sq.shape[0]] = sq
                        mask[i, :sq.shape[0]] = False
                        
                    with torch.no_grad():
                        out_embs = self.opponent_encoder(padded, mask=mask)
                    
                    # Slice back
                    out_embs = out_embs[:orig_b]
                    
                    new_embs_flat = [out_embs[i:i+1].detach() for i in range(len(seqs))]

                # Distribute back to cache and build tensor
                emb_idx = 0
                for s_idx, s in enumerate(states):
                    table = tables[game_indices[s_idx]]
                    # encode empties and cached
                    cached = s['_cached_embeds']
                    for opid, history in s['_uncached_opp_histories']:
                        if not history:
                            emb = self.opponent_encoder.encode_empty(1, device=str(self.device))
                            table.opp_embed_cache[opid] = emb.detach()
                            cached[opid] = emb
                        else:
                            emb = new_embs_flat[emb_idx]
                            emb_idx += 1
                            table.opp_embed_cache[opid] = emb
                            cached[opid] = emb

                    # Assemble
                    ordered = [cached[opid] for opid in s['_opp_ids']]
                    if not ordered:
                        s['opponent_embeddings'] = self.opponent_encoder.encode_empty(1, device=str(self.device)).unsqueeze(1)
                    else:
                        s['opponent_embeddings'] = torch.cat(ordered, dim=0).unsqueeze(0)

                # 2. GROUP INFERENCES BY MODEL
                groups = {} # model_identifier -> list of (batch_idx, game_idx, state)
                for b_idx, (g_idx, s) in enumerate(zip(game_indices, states)):
                    model_id = 'hero' if s['_is_hero'] else s['_pool_idx']
                    if model_id not in groups: groups[model_id] = []
                    groups[model_id].append((b_idx, g_idx, s))

                # Result array mapped by game_idx
                results_to_yield = {} # game_idx -> (probs, value, sizing_probs, opp_embed)

                for model_id, items in groups.items():
                    orig_b = len(items)
                    pad_b = ((orig_b + 31) // 32) * 32
                    
                    # Duplicate last state to pad batch
                    sub_states = [item[2] for item in items]
                    while len(sub_states) < pad_b:
                        sub_states.append(sub_states[-1])
                    
                    # Always pad to maximum possible opponents to keep shape constant
                    max_opps = self.config.max_players - 1
                    embed_dim = sub_states[0]['opponent_embeddings'].shape[2]
                    stat_dim = sub_states[0]['opponent_stats'].shape[2]

                    batch_hole = torch.cat([s['hole_cards'] for s in sub_states], dim=0).to(self.device)
                    batch_comm = torch.cat([s['community_cards'] for s in sub_states], dim=0).to(self.device)
                    batch_num = torch.cat([s['numeric_features'] for s in sub_states], dim=0).to(self.device)
                    batch_mask = torch.cat([s['action_mask'] for s in sub_states], dim=0).to(self.device)
                    batch_s_mask = torch.cat([s['sizing_mask'] for s in sub_states], dim=0).to(self.device)
                    batch_own = torch.cat([s['own_stats'] for s in sub_states], dim=0).to(self.device)

                    padded_opp_embeds = []
                    padded_opp_stats = []
                    padded_opp_gs = []
                    opp_masks = []
                    for s in sub_states:
                        oe = s['opponent_embeddings']
                        os_t = s['opponent_stats']
                        og = s.get('opponent_game_state', torch.zeros(1, oe.shape[1], OPP_GAME_STATE_DIM, device=oe.device))
                        n_opp = oe.shape[1]
                        if n_opp < max_opps:
                            pad_e = torch.zeros(1, max_opps - n_opp, embed_dim, device=oe.device)
                            oe = torch.cat([oe, pad_e], dim=1)
                            pad_s = torch.zeros(1, max_opps - n_opp, stat_dim, device=os_t.device)
                            os_t = torch.cat([os_t, pad_s], dim=1)
                            pad_g = torch.zeros(1, max_opps - n_opp, OPP_GAME_STATE_DIM, device=og.device)
                            og = torch.cat([og, pad_g], dim=1)
                        padded_opp_embeds.append(oe)
                        padded_opp_stats.append(os_t)
                        padded_opp_gs.append(og)
                        m = torch.ones(1, max_opps, dtype=torch.bool, device=oe.device)
                        m[0, :n_opp] = False
                        opp_masks.append(m)

                    batch_opp_embed = torch.cat(padded_opp_embeds, dim=0).to(self.device)
                    batch_opp_stats = torch.cat(padded_opp_stats, dim=0).to(self.device)
                    batch_opp_gs = torch.cat(padded_opp_gs, dim=0).to(self.device)
                    batch_opp_mask = torch.cat(opp_masks, dim=0).to(self.device)

                    # Phase 5+6: Batch hand action sequences & profiles
                    batch_ha_seq = torch.cat([s.get('hand_action_seq', torch.zeros(1, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)) for s in sub_states], dim=0).to(self.device)
                    batch_ha_len = torch.cat([s.get('hand_action_len', torch.ones(1, dtype=torch.long)) for s in sub_states], dim=0).to(self.device)
                    batch_actor_prof = torch.cat([s.get('actor_profiles_seq', torch.zeros(1, MAX_HAND_ACTIONS, PROFILE_DIM)) for s in sub_states], dim=0).to(self.device)
                    
                    batch_hero_prof = torch.stack([s.get('hero_profile', torch.zeros(PROFILE_DIM)) for s in sub_states], dim=0).to(self.device)
                    
                    padded_opp_profiles = []
                    for s in sub_states:
                        op = s.get('opponent_profiles', torch.zeros(1, PROFILE_DIM))
                        if op.shape[0] < max_opps:
                            op = torch.cat([op, torch.zeros(max_opps - op.shape[0], PROFILE_DIM, device=op.device)], dim=0)
                        padded_opp_profiles.append(op.unsqueeze(0))
                    batch_opp_profiles = torch.cat(padded_opp_profiles, dim=0).to(self.device)

                    target_model = self.policy if model_id == 'hero' else self.pool._frozen_models.get(model_id, self.policy)
                    
                    with torch.no_grad():
                        output = target_model(
                            hole_cards=batch_hole, community_cards=batch_comm, numeric_features=batch_num,
                            opponent_embeddings=batch_opp_embed, opponent_stats=batch_opp_stats, own_stats=batch_own,
                            opponent_game_state=batch_opp_gs,
                            action_mask=batch_mask, sizing_mask=batch_s_mask, opponent_mask=batch_opp_mask,
                            hand_action_seq=batch_ha_seq, hand_action_len=batch_ha_len,
                            actor_profiles_seq=batch_actor_prof,
                            hero_profile=batch_hero_prof,
                            opponent_profiles=batch_opp_profiles,
                        )

                    for loc_idx, (b_idx, g_idx, s) in enumerate(items):
                        probs = output.action_type_probs[loc_idx].cpu()
                        value = output.value[loc_idx, 0].item() if hasattr(output, 'value') else 0.0
                        sizing_logits = output.bet_size_logits[loc_idx]
                        sizing_probs = torch.softmax(sizing_logits, dim=-1).cpu().tolist()
                        results_to_yield[g_idx] = (probs, value, sizing_probs, s['opponent_embeddings'].detach().cpu())

                # Advance generators
                for game_idx in game_indices:
                    g, _ = pending[game_idx]
                    probs, value, sizing_probs, op_emb_tup = results_to_yield[game_idx]
                    try:
                        state = g.send((probs, value, sizing_probs, op_emb_tup))
                        pending[game_idx] = (g, state)
                    except StopIteration as e:
                        exps, reward = e.value
                        for pexp in exps: all_exp.extend(pexp)
                        total_reward += reward
                        total_finished += 1
                        del pending[game_idx]

        return all_exp, total_reward



    def _precompute_ppo_data(self, experiences: List[Experience]):
        """Pre-compute and batch all PPO data: tensors + GAE advantages/returns.
        
        Called ONCE before PPO epochs. Returns dict of batched tensors.
        """
        if not experiences:
            return None

        # 1. Stack scalar/fixed-size features (keep on CPU to prevent MPS source-shape caching leak!)
        hole_cards = torch.cat([e.hole_cards for e in experiences], dim=0)
        community = torch.cat([e.community_cards for e in experiences], dim=0)
        numeric = torch.cat([e.numeric_features for e in experiences], dim=0)
        own_stats = torch.cat([e.own_stats for e in experiences], dim=0)
        action_masks = torch.cat([e.action_mask for e in experiences], dim=0)
        sizing_masks = torch.cat([e.sizing_mask for e in experiences], dim=0)
        
        # 2. Pad opponent sequence arrays
        # Always pad to hardcoded max dimension to lock tensor shapes for MPS!
        max_opps = self.config.max_players - 1
        
        opp_emb_list = []
        opp_stat_list = []
        opp_gs_list = []
        opp_mask_list = []
        opp_prof_list = []
        
        for e in experiences:
            emb = e.opponent_embeddings
            stat = e.opponent_stats
            gs = e.opponent_game_state
            prof = e.opponent_profiles
            
            curr_opps = emb.shape[1]
            pad_len = max_opps - curr_opps
            
            if pad_len > 0:
                emb_pad = torch.zeros(1, pad_len, emb.shape[2])
                stat_pad = torch.zeros(1, pad_len, stat.shape[2])
                gs_pad = torch.zeros(1, pad_len, OPP_GAME_STATE_DIM)
                prof_pad = torch.zeros(pad_len, prof.shape[1] if len(prof.shape) > 1 else PROFILE_DIM)
                
                emb = torch.cat([emb, emb_pad], dim=1)
                stat = torch.cat([stat, stat_pad], dim=1)
                gs = torch.cat([gs, gs_pad], dim=1)
                prof = torch.cat([prof, prof_pad], dim=0)
                
                mask = torch.tensor([[False]*curr_opps + [True]*pad_len], dtype=torch.bool)
            else:
                mask = torch.tensor([[False]*curr_opps], dtype=torch.bool)
                
            opp_emb_list.append(emb)
            opp_stat_list.append(stat)
            opp_gs_list.append(gs)
            opp_mask_list.append(mask)
            opp_prof_list.append(prof.unsqueeze(0))
            
        opp_embeds = torch.cat(opp_emb_list, dim=0)
        opp_stats = torch.cat(opp_stat_list, dim=0)
        opp_gs = torch.cat(opp_gs_list, dim=0)
        opp_masks = torch.cat(opp_mask_list, dim=0)
        opp_profiles = torch.cat(opp_prof_list, dim=0)
        
        # 3. Stack hand action history (already padded to MAX_HAND_ACTIONS)
        hand_action_seqs = torch.cat([e.hand_action_seq for e in experiences], dim=0)
        hand_action_lens = torch.cat([e.hand_action_len for e in experiences], dim=0)
        actor_profiles_seqs = torch.cat([e.actor_profiles_seq for e in experiences], dim=0)
        hero_profiles = torch.stack([e.hero_profile for e in experiences], dim=0)
        # opp_profiles already padded and stacked above

        # 4. Stack targets (keep on CPU)
        action_t = torch.tensor([e.action_idx for e in experiences], dtype=torch.long)
        sizing_t = torch.tensor([e.sizing_idx for e in experiences], dtype=torch.long)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32)
        old_action_log_probs = torch.tensor([e.action_log_prob for e in experiences], dtype=torch.float32)
        old_sizing_log_probs = torch.tensor([e.sizing_log_prob for e in experiences], dtype=torch.float32)
        old_values = torch.tensor([e.value for e in experiences], dtype=torch.float32)
        equity_x_pot = torch.tensor([e.equity_x_pot for e in experiences], dtype=torch.float32)

        # 4. Compute GAE advantages and returns ONCE (using original values)
        hand_groups: Dict[int, List[int]] = {}
        for idx, e in enumerate(experiences):
            hid = e.hand_id
            if hid not in hand_groups:
                hand_groups[hid] = []
            hand_groups[hid].append(idx)

        gae_advantages = torch.zeros(len(experiences))
        gae_returns = torch.zeros(len(experiences))

        for hid, indices in hand_groups.items():
            indices.sort(key=lambda i: experiences[i].step_idx)
            trajectory = [experiences[i] for i in indices]
            advantages, full_returns, _ = compute_gae(trajectory, self.config.gamma, self.config.gae_lambda, self.config.v_res_alpha)
            for local_idx, global_idx in enumerate(indices):
                gae_advantages[global_idx] = advantages[local_idx]
                gae_returns[global_idx] = full_returns[local_idx]

        # Reward is already normalized by effective stack at source (hero_reward /= eff_stack),
        # so advantages and returns are in consistent "fraction of stack" units (~[-1, +1]).
        # Mean-centering ONLY (no std div) removes critic bias while preserving stack-depth severity:
        gae_advantages = gae_advantages - gae_advantages.mean()

        return {
            'hole_cards': hole_cards,
            'community': community,
            'numeric': numeric,
            'own_stats': own_stats,
            'action_masks': action_masks,
            'sizing_masks': sizing_masks,
            'opp_embeds': opp_embeds,
            'opp_stats': opp_stats,
            'opp_gs': opp_gs,
            'opp_masks': opp_masks,
            'hand_action_seqs': hand_action_seqs,
            'hand_action_lens': hand_action_lens,
            'actor_profiles_seqs': actor_profiles_seqs,
            'hero_profiles': hero_profiles,
            'opp_profiles': opp_profiles,
            'action_t': action_t,
            'sizing_t': sizing_t,
            'old_log_probs': old_log_probs,
            'old_action_log_probs': old_action_log_probs,
            'old_sizing_log_probs': old_sizing_log_probs,
            'old_values': old_values,
            'gae_advantages': gae_advantages,
            'gae_returns': gae_returns,
            'equity_x_pot': equity_x_pot,
        }

    def _compute_ppo_loss_minibatch(self, data: dict, indices: List[int]) -> float:
        """Compute PPO loss on a mini-batch."""
        return compute_ppo_loss(
            policy=self.policy,
            data=data,
            indices=indices,
            device=self.device,
            ppo_clip=self.config.ppo_clip,
            value_coef=self.config.value_coef,
            entropy_coef=self._get_entropy_coef(),
            remove_clip=self.config.remove_clip,
            kl_beta=self.config.kl_beta,
        )



    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────

    def train(self, num_epochs: int = 100, epoch_callback=None, start_epoch: int = 0) -> Dict[str, List[float]]:
        """Run balanced self-play training with batched GPU inference.
        
        Args:
            num_epochs: Number of epochs to train.
            epoch_callback: Optional function(trainer, epoch, metrics) called after each epoch.
            start_epoch: Epoch to start from (for resumed training).
        """
        self.total_epochs = num_epochs  # for annealing schedules
        mini_batch_size = self.config.mini_batch_size

        metrics = {
            'epoch_reward': [],
            'epoch_loss': [],
        }

        print(f"Device: {self.device}")
        print(f"Balanced self-play: hero-only training, frozen opponent pool")
        print(f"GAE: gamma={self.config.gamma}, lambda={self.config.gae_lambda}")
        print(f"Entropy: {self.config.entropy_coef} -> {self.config.entropy_coef_end}")
        print(f"Epsilon-greedy: {self.config.epsilon} -> {self.config.epsilon_end}")
        print(f"Mini-batch PPO: batch_size={mini_batch_size}, ppo_epochs={self.config.ppo_epochs}")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            # Sync frozen opponent periodically
            if epoch > 0 and epoch % self.config.frozen_update_interval == 0:
                self._sync_frozen()
                if self.config.verbose:
                    pool_total = len(self.pool.recent) + len(self.pool.archive)
                    print(f"    [Frozen sync] Pool: {len(self.pool.recent)} recent + {len(self.pool.archive)} archive = {pool_total}")

            # Batched simulation: all hands run as generators with batched GPU calls
            if self.config.num_workers > 0 and epoch == start_epoch:
                print("    [Warning] --num-workers > 0 ignored (vectorized env removed), using batched epoch")
            all_exp, total_reward = self._run_batched_epoch()

            avg_reward = total_reward / self.config.hands_per_epoch

            # Action distribution (conditional: % chosen when legal)
            action_pcts = count_action_distribution(all_exp)

            # Pre-compute GAE and batch tensors ONCE
            self.policy.train()
            self.opponent_encoder.train()
            ppo_data = self._precompute_ppo_data(all_exp)

            # Mini-batch PPO update
            total_loss = 0.0
            total_action_loss = 0.0
            total_sizing_loss = 0.0
            total_value_loss = 0.0
            num_updates = 0
            if ppo_data is not None:
                n = len(all_exp)
                all_indices = list(range(n))
                for _ in range(self.config.ppo_epochs):
                    self.rng.shuffle(all_indices)
                    for start in range(0, n, mini_batch_size):
                        if start + mini_batch_size > n:
                            break  # Drop last incomplete batch to avoid MPS graph compilation leak
                            
                        mb_indices = all_indices[start:start + mini_batch_size]
                        self.optimizer.zero_grad()
                        loss_val, a_loss, s_loss, v_loss = self._compute_ppo_loss_minibatch(ppo_data, mb_indices)
                        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                        nn.utils.clip_grad_norm_(self.opponent_encoder.parameters(), 1.0)
                        self.optimizer.step()
                        total_loss += loss_val
                        total_action_loss += a_loss
                        total_sizing_loss += s_loss
                        total_value_loss += v_loss
                        num_updates += 1

            avg_loss = total_loss / max(num_updates, 1)
            avg_a = total_action_loss / max(num_updates, 1)
            avg_s = total_sizing_loss / max(num_updates, 1)
            avg_v = total_value_loss / max(num_updates, 1)

            # Free experience memory and flush device cache
            del all_exp
            del ppo_data
            import gc
            gc.collect()
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

            metrics['epoch_reward'].append(avg_reward)
            metrics['epoch_loss'].append(avg_loss)

            epoch_duration = time.time() - epoch_start
            hands_per_sec = self.config.hands_per_epoch / epoch_duration

            if self.config.verbose or (epoch + 1) % self.config.log_interval == 0:
                deep_ai = action_pcts.get('deep_ai', 0)
                print(
                    f"Epoch {epoch + 1:4d} ({epoch_duration:.1f}s, {hands_per_sec:.1f} hands/s) | "
                    f"Reward: {avg_reward:+.3f} bb | "
                    f"Loss: {avg_loss:.4f} (A:{avg_a:.3f} S:{avg_s:.3f} V:{avg_v:.3f}) | "
                    f"F/Ch/Ca/R/AI: {action_pcts['fold']:.0f}/{action_pcts['check']:.0f}/{action_pcts['call']:.0f}/{action_pcts['raise']:.0f}/{action_pcts['allin']:.0f}% "
                    f"DAI:{deep_ai:.0f}%"
                )

            if epoch_callback:
                epoch_callback(self, epoch + 1, metrics)

        return metrics

