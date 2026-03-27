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
from search.search import SearchEngine, SearchConfig
from training.personality import (
    SituationalPersonality, PersonalityModifier,
    sample_table_personalities, detect_situations,
)


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
    gamma: float = 0.99          # was 1.0 — temporal discount for GAE
    gae_lambda: float = 0.95     # GAE lambda parameter
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    mini_batch_size: int = 64          # mini-batch size for PPO updates
    entropy_coef: float = 0.005   # was 0.05 — reduced 10x to not drown out policy gradient
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5

    # Self-play
    hands_per_epoch: int = 512   # was 128 — increased for hero-only training
    batch_chunk_size: int = 500  # Max simultaneous games per sub-batch

    # Frozen opponent pool
    frozen_update_interval: int = 20   # sync frozen opponent every N epochs
    max_recent_pool: int = 10           # recent checkpoints (FIFO)
    max_archive_pool: int = 5           # old checkpoints (random replacement)

    # Epsilon-greedy exploration (training only, annealed)
    epsilon: float = 0.15              # probability of random action (start)
    epsilon_end: float = 0.08          # probability of random action (end) — poker needs permanent mixing

    # Opponent modeling
    history_reset_interval: Tuple[int, int] = (300, 500)

    # Search-guided training (expert iteration)
    search_fraction: float = 0.0     # fraction of hands to use search (0-1)
    search_iterations: int = 50      # CFR iterations per search call

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Logging
    log_interval: int = 10
    verbose: bool = False

    # Seed
    seed: int = 42


@dataclass
class TableState:
    """Persistent state for a single table across multiple hands."""
    action_histories: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    stat_tracker: StatTracker = field(default_factory=StatTracker)
    hands_since_reset: int = 0
    next_reset_at: int = 50
    hands_since_personality_reset: int = 0
    next_personality_reset_at: int = 50
    table_personalities: List[Optional['SituationalPersonality']] = field(default_factory=list)
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

        # ── Frozen opponent (no gradients, provides stable opposition) ──
        # Template model on CPU for cloning into per-seat frozen models
        self._frozen_template = copy.deepcopy(self.policy) # Now kept on original device or moved dynamically
        self.frozen_opponent_encoder = copy.deepcopy(self.opponent_encoder)
        for p in self._frozen_template.parameters():
            p.requires_grad = False
        for p in self.frozen_opponent_encoder.parameters():
            p.requires_grad = False

        # Two-tier opponent pool: recent (FIFO) + archive (old preserved)
        self.opponent_pool_recent: List[dict] = [
            {k: v.cpu() for k, v in self.policy.state_dict().items()}
        ]
        self.opponent_pool_archive: List[dict] = []
        # Per-table frozen models: {pool_idx: model}
        self._frozen_models: Dict[int, 'PolicyNetwork'] = {}

        # Personality curriculum
        self.current_epoch = 0
        self.total_epochs = 1  # set by train() for annealing schedules

        # ── Hand counter for unique hand IDs ──────────────────
        self._hand_counter = 0

        # ── Search engine ─────────────────────────────────────
        if self.config.search_fraction > 0:
            self.search_engine = SearchEngine(
                policy=self.policy,
                opponent_encoder=self.opponent_encoder,
                config=SearchConfig(num_iterations=self.config.search_iterations),
            )
        else:
            self.search_engine = None

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
        """
        Save current live weights to opponent pool (two-tier: recent + archive).
        """
        cpu_state = {k: v.cpu() for k, v in self.policy.state_dict().items()}

        # Evict oldest recent entry → 30% chance it goes to archive
        if len(self.opponent_pool_recent) >= self.config.max_recent_pool:
            evicted = self.opponent_pool_recent.pop(0)
            if len(self.opponent_pool_archive) < self.config.max_archive_pool:
                # Free slot — always archive
                self.opponent_pool_archive.append(evicted)
            elif self.rng.random() < 0.30:
                # Archive full — 30% chance replace a random entry
                idx = self.rng.randint(0, len(self.opponent_pool_archive) - 1)
                self.opponent_pool_archive[idx] = evicted

        self.opponent_pool_recent.append(cpu_state)

        # Also sync opponent encoder
        cpu_enc_state = {k: v.cpu() for k, v in self.opponent_encoder.state_dict().items()}
        self.frozen_opponent_encoder.load_state_dict(cpu_enc_state)

        # Reset current loaded index
        self._frozen_models = {}  # force rebuild on next table setup

    def _get_combined_pool(self) -> List[dict]:
        """Get combined pool (recent + archive) as a single list."""
        return self.opponent_pool_recent + self.opponent_pool_archive

    def _build_table_models(self, seat_pool_idx: Dict[int, int]):
        """Build frozen models for each unique pool index at the table.
        
        Caches models by pool index — only creates new models for indices
        not already in the cache. Much faster than rebuilding every hand.
        """
        combined_pool = self._get_combined_pool()
        unique_indices = set(seat_pool_idx.values())
        # Only build models for new pool indices (cache hit = skip)
        for idx in unique_indices:
            if idx not in self._frozen_models and idx < len(combined_pool):
                model = self._make_frozen_model(combined_pool[idx])
                self._frozen_models[idx] = model

    def _make_frozen_model(self, state_dict: dict):
        """Create a fresh frozen model from state dict (no deepcopy)."""
        from model.policy_network import PolicyNetwork
        model = PolicyNetwork(
            embed_dim=self.config.embed_dim,
            num_cross_attn_heads=self.config.num_heads,
            num_cross_attn_layers=self.config.num_layers,
            opponent_embed_dim=self.config.opponent_embed_dim,
        ).to('cpu')
        model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def save_pool(self, save_dir: str):
        """Save opponent pool to disk for checkpoint persistence."""
        import os
        pool_dir = os.path.join(save_dir, 'pool')
        os.makedirs(pool_dir, exist_ok=True)
        for i, sd in enumerate(self.opponent_pool_recent):
            torch.save(sd, os.path.join(pool_dir, f'recent_{i:03d}.pt'))
        for i, sd in enumerate(self.opponent_pool_archive):
            torch.save(sd, os.path.join(pool_dir, f'archive_{i:03d}.pt'))

    def load_pool(self, load_dir: str):
        """Load opponent pool from disk."""
        import os, glob
        pool_dir = os.path.join(load_dir, 'pool')
        if not os.path.exists(pool_dir):
            return
        # Load recent entries
        recent_files = sorted(glob.glob(os.path.join(pool_dir, 'recent_*.pt')))
        if recent_files:
            self.opponent_pool_recent = [
                torch.load(f, map_location='cpu', weights_only=True) for f in recent_files
            ]
        # Load archive entries
        archive_files = sorted(glob.glob(os.path.join(pool_dir, 'archive_*.pt')))
        if archive_files:
            self.opponent_pool_archive = [
                torch.load(f, map_location='cpu', weights_only=True) for f in archive_files
            ]
        self._frozen_models = {}  # force rebuild

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

    def _compute_gae(self, trajectory: List[Experience]) -> Tuple[List[float], List[float]]:
        """
        Compute GAE advantages and returns for a single hand trajectory.

        Args:
            trajectory: List of Experience tuples for one hand, ordered by step_idx.

        Returns:
            (advantages, returns) — parallel lists of floats, one per step.
        """
        gamma = self.config.gamma
        lam = self.config.gae_lambda
        n = len(trajectory)
        if n == 0:
            return [], []

        # Collect values and the terminal reward
        values = [exp.value for exp in trajectory]
        terminal_reward = trajectory[-1].reward  # only last step has real reward

        # TD errors: δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        deltas = []
        for t in range(n):
            if t == n - 1:
                # Last step: r = terminal_reward, V(s_{t+1}) = 0 (hand over)
                delta = terminal_reward - values[t]
            else:
                # Intermediate step: r = 0, V(s_{t+1}) = values[t+1]
                delta = gamma * values[t + 1] - values[t]
            deltas.append(delta)

        # GAE: A_t = Σ (γλ)^k · δ_{t+k}, computed backward
        advantages = [0.0] * n
        gae = 0.0
        for t in reversed(range(n)):
            gae = deltas[t] + gamma * lam * gae
            advantages[t] = gae

        # Returns: G_t = A_t + V(s_t) (proper targets for value head)
        returns = [advantages[t] + values[t] for t in range(n)]

        return advantages, returns

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



    def _get_opponent_stats(self, table: TableState, hero_id: int, num_players: int) -> torch.Tensor:
        """Get HUD stats for all opponents. Returns (1, num_opp, stat_dim)."""
        stats = []
        for pid in range(num_players):
            if pid == hero_id:
                continue
            stats.append(table.stat_tracker.get_stats(pid))

        if not stats:
            return torch.zeros(1, 1, NUM_STAT_FEATURES, device=self.device)

        return torch.stack(stats).unsqueeze(0).to(self.device)  # (1, num_opp, stat_dim)

    def _get_opponent_game_state(self, game_state: GameState, hero_id: int, num_players: int) -> torch.Tensor:
        """Build per-opponent game state: seat_onehot(9) + stack + bet + pot_committed + active + all_in.
        Returns (1, num_opp, 14)."""
        bb = max(game_state.big_blind, 1.0)
        pot = max(game_state.pot, 1.0)
        opp_states = []
        for pid in range(num_players):
            if pid == hero_id:
                continue
            p = game_state.players[pid]
            seat_oh = [0.0] * 9
            seat_oh[min(pid, 8)] = 1.0
            opp_states.append(torch.tensor(
                seat_oh + [
                    p.stack / (100.0 * bb),
                    p.bet_this_street / pot if pot > 0 else 0.0,
                    p.bet_total / (100.0 * bb),
                    float(p.is_active),
                    float(p.is_all_in),
                ], dtype=torch.float32
            ))
        if not opp_states:
            return torch.zeros(1, 1, OPP_GAME_STATE_DIM, device=self.device)
        return torch.stack(opp_states).unsqueeze(0).to(self.device)

    def _get_personality_gto_fraction(self) -> float:
        """Graduated personality curriculum schedule."""
        epoch = self.current_epoch
        if epoch < 20:
            return 1.0  # 100% GTO until epoch 20
        return 0.6      # 60% GTO / 40% Personalities after epoch 20

    def _maybe_reset_histories(self, table: TableState):
        """Periodically reset opponent histories and personalities (simulate new table)."""
        table.hands_since_reset += 1
        table.hands_since_personality_reset += 1

        # Hard reset: total flush of histories and stats (e.g. 300-500 hands)
        if table.hands_since_reset >= table.next_reset_at:
            table.action_histories.clear()
            table.stat_tracker.reset()
            table.hands_since_reset = 0
            table.next_reset_at = self.rng.randint(*self.config.history_reset_interval)
            # Reshuffle personalities and per-seat weights when we "sit down at a new table"
            table.table_personalities = []
            table.seat_pool_idx = {}
            table.hands_since_personality_reset = 0

        # Soft reset: flush personalities only (keep histories for OpponentEncoder to adapt)
        elif table.hands_since_personality_reset >= table.next_personality_reset_at:
            table.table_personalities = []
            table.seat_pool_idx = {}
            table.hands_since_personality_reset = 0

    # ─────────────────────────────────────────────────────────
    # State encoding
    # ─────────────────────────────────────────────────────────

    def _encode_action_mask(self, game_state: GameState) -> torch.Tensor:
        """Encode legal actions as (1, 4) bool tensor on device."""
        legal_types = game_state.get_legal_actions()
        mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool, device=self.device)

        for at in legal_types:
            if at == ActionType.FOLD:
                mask[0, ActionIndex.FOLD] = True
            elif at == ActionType.CHECK:
                mask[0, ActionIndex.CHECK] = True
            elif at == ActionType.CALL:
                mask[0, ActionIndex.CALL] = True
            elif at == ActionType.RAISE:
                mask[0, ActionIndex.RAISE] = True
            elif at == ActionType.ALL_IN:
                mask[0, ActionIndex.RAISE] = True

        return mask

    def _encode_state(self, game_state: GameState, player_idx: int) -> dict:
        """Encode game state to device tensors."""
        p = game_state.players[player_idx]

        hole = self._to(torch.tensor([list(p.hole_cards)], dtype=torch.long))
        board = list(game_state.board)
        while len(board) < 5:
            board.append(-1)
        community = self._to(torch.tensor([board[:5]], dtype=torch.long))

        bb = game_state.big_blind
        norm = 100.0 * bb
        pot = game_state.pot / norm
        own_stack = p.stack / norm
        own_bet = p.bet_this_street / norm

        # 9-dim seat one-hot (relative position from BTN)
        rel_pos = (player_idx - game_state.dealer_button) % game_state.num_players
        seat_onehot = [0.0] * 9
        seat_onehot[rel_pos] = 1.0

        # IP flag: hero acts last postflop?
        # Approximate: highest relative position among active players = in position
        active_positions = []
        for i, pp in enumerate(game_state.players):
            if pp.is_active:
                active_positions.append((i - game_state.dealer_button) % game_state.num_players)
        ip_flag = 1.0 if (active_positions and rel_pos == max(active_positions)) else 0.0

        # 4-dim street one-hot
        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        street_idx = street_map.get(game_state.street, 0)
        street_onehot = [0.0] * 4
        street_onehot[street_idx] = 1.0

        num_active = sum(1 for pp in game_state.players if pp.is_active)
        current_bet = game_state.current_bet / norm
        min_raise = game_state.min_raise / norm
        amount_to_call = max(0.0, current_bet - own_bet)
        spr = p.stack / max(game_state.pot, 0.01)

        numeric = self._to(torch.tensor([[
            pot, own_stack, own_bet,
            *seat_onehot,        # 9 dims
            ip_flag,             # 1 dim
            *street_onehot,      # 4 dims
            game_state.num_players / 9.0, num_active / 9.0,
            current_bet, min_raise, amount_to_call,
            spr,                 # 1 dim
        ]], dtype=torch.float32))

        action_mask = self._encode_action_mask(game_state)
        
        from model.action_space import get_sizing_mask
        sizing_mask = self._to(get_sizing_mask(game_state).unsqueeze(0))

        return {
            'hole_cards': hole,
            'community_cards': community,
            'numeric_features': numeric,
            'action_mask': action_mask,
            'sizing_mask': sizing_mask,
        }

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

        # Per-seat opponent assignment
        if not table.seat_pool_idx or len(table.seat_pool_idx) != num_p - 1:
            table.seat_pool_idx = {}
            for pid in range(1, num_p):
                combined_pool = self._get_combined_pool()
                if combined_pool:
                    table.seat_pool_idx[pid] = self.rng.randint(0, len(combined_pool) - 1)
                else:
                    table.seat_pool_idx[pid] = 0

        # Table-level personality
        if len(table.table_personalities) != num_p:
            if self.current_epoch < 20 or self.rng.random() < 0.33:
                gto_frac = 1.0
            else:
                gto_frac = 0.60
            table.table_personalities = sample_table_personalities(
                num_p, gto_fraction=gto_frac, rng=self.rng
            )

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
        preflop_folders = set()  # players who folded preflop (for saw_flop)

        # Phase 5: Accumulate all raw action tokens during this hand
        hand_action_tokens = []  # list of (raw_token_13d, actor_pid)

        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            if not p.is_active:
                break

            encoded = self._encode_state(game_state, pid)
            opp_stats = self._get_opponent_stats(table, pid, num_p)
            own_stats = self._to(table.stat_tracker.get_stats(pid).unsqueeze(0))
            opp_game_state = self._get_opponent_game_state(game_state, pid, num_p)

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
                if self.search_engine is not None and self.rng.random() < self.config.search_fraction:
                    street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
                    cur_street = street_map.get(game_state.street, 0)
                    pot_bb = game_state.pot / game_state.big_blind
                    if self.search_engine.should_search(probs, pot_bb, cur_street):
                        search_stacks = [pp.stack for pp in game_state.players]
                        search_actions, search_probs = self.search_engine.search(
                            pot=game_state.pot, stacks=search_stacks, board=list(game_state.board),
                            street=cur_street, hero=pid,
                            opp_embed=opp_embed_tensor.clone().cpu() if opp_embed_tensor is not None else None,
                            opp_stats=opp_stats.clone().cpu(),
                            own_stats=own_stats.clone().cpu(),
                        )
                        refined = torch.zeros(NUM_ACTION_TYPES, device=self.device)
                        for sa, sp in zip(search_actions, search_probs):
                            if sa == 'check': refined[ActionIndex.CHECK] += sp
                            elif sa == 'fold': refined[ActionIndex.FOLD] += sp
                            elif sa == 'call': refined[ActionIndex.CALL] += sp
                            elif sa.startswith('raise_') or sa == 'allin': refined[ActionIndex.RAISE] += sp
                        if refined.sum() > 0:
                            refined = refined / refined.sum()
                            probs = 0.7 * refined + 0.3 * probs.to(self.device)

                action_mask_cpu = encoded['action_mask'].squeeze(0).cpu()
                legal_indices = action_mask_cpu.nonzero(as_tuple=True)[0].tolist()

                if self.rng.random() < epsilon and len(legal_indices) > 1:
                    action_idx = self.rng.choice(legal_indices)
                else:
                    dist = Categorical(probs)
                    action_idx = dist.sample().item()

                raw_dist = Categorical(raw_probs)
                action_lp = raw_dist.log_prob(torch.tensor(action_idx)).item()
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

                    sizing_lp = Categorical(sizing_tensor).log_prob(torch.tensor(sizing_idx)).item()

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
                })
                hero_step_idx += 1

            else:
                # ── OPPONENT ──
                personality = None
                is_facing_raise = game_state.current_bet > 0 and p.bet_this_street < game_state.current_bet
                if pid < len(table.table_personalities):
                    personality = table.table_personalities[pid]
                    if personality is not None:
                        street_map_sit = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
                        situations = detect_situations(
                            street=street_map_sit.get(game_state.street, 0),
                            board_cards=list(game_state.board) if game_state.board else None,
                            stack_bb=p.stack / max(game_state.big_blind, 1),
                            is_facing_raise=is_facing_raise,
                        )
                        c1, c2 = p.hole_cards
                        r1, r2 = c1 // 4, c2 // 4
                        if game_state.street == Street.PREFLOP:
                            if r1 == r2: hand_strength = 0.55 + r1 * 0.035
                            elif max(r1, r2) >= 10: hand_strength = 0.45 + max(r1, r2) * 0.02
                            elif (c1 % 4) == (c2 % 4): hand_strength = 0.3
                            else: hand_strength = 0.15 + max(r1, r2) * 0.01
                        else:
                            paired_board = any(r1 == (bc // 4) or r2 == (bc // 4) for bc in game_state.board if bc >= 0)
                            if paired_board: hand_strength = 0.65 + max(r1, r2) * 0.02
                            elif max(r1, r2) >= 10: hand_strength = 0.4
                            else: hand_strength = 0.2

                        probs = personality.apply(
                            probs, 
                            situations, 
                            hand_strength=hand_strength, 
                            is_facing_raise=is_facing_raise,
                            opponent_stats=table.stat_tracker.get_stats(0).to(probs.device)
                        )

                dist = Categorical(probs)
                action_idx = dist.sample().item()
                sizing_idx = 0
                if action_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
                    sizing_tensor = torch.tensor(sizing_probs, dtype=torch.float32)
                    if personality is not None:
                        sizing_tensor = personality.apply_sizing(
                            sizing_tensor, 
                            situations, 
                            hand_strength=hand_strength,
                            opponent_stats=table.stat_tracker.get_stats(0)
                        )
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
                current_street = game_state.street
                player_checked_this_street = [False] * num_p
                first_bet_this_street_by = -1

        # --- End of hand: populate remaining fields ---
        results = dealer.get_results()
        profits = results['profit']
        for pid in range(num_p):
            hand_records[pid].result = profits[pid]
            # saw_flop: player didn't fold preflop and hand went past preflop
            hand_records[pid].saw_flop = (pid not in preflop_folders) and game_state.street.value > Street.PREFLOP.value
            hand_records[pid].was_pf_aggressor = (pf_aggressor == pid)
            hand_records[pid].went_to_showdown = (game_state.street == Street.SHOWDOWN and not game_state.players[pid].is_folded)
            hand_records[pid].won_at_showdown = (hand_records[pid].went_to_showdown and pid in (results.get('winners', [])))
            table.stat_tracker.record_hand(pid, hand_records[pid])

        self._maybe_reset_histories(table)

        hero_reward = profits[0] / max(self.config.big_blind, 1.0)
        hero_exp_list = []
        for exp_dict in hero_experiences:
            # Need to get opponent embeddings back from dict, but actually, precompute calls _get_all_opponent_embeddings!
            # Wait, precompute doesn't call it, it pulls from Experience.
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
                actor_profiles_seq=torch.zeros(1, MAX_HAND_ACTIONS, PROFILE_DIM),  # placeholder — profiles built in batch
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
            ))

        return [hero_exp_list] + [[] for _ in range(num_p - 1)], hero_reward

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
                combined_pool = self._get_combined_pool()
                for pool_idx in tables[i].seat_pool_idx.values():
                    if pool_idx not in self._frozen_models:
                        if combined_pool and pool_idx < len(combined_pool):
                            fm = copy.deepcopy(self.policy).to(self.device)
                            fm.load_state_dict(combined_pool[pool_idx])
                            for p in fm.parameters(): p.requires_grad = False
                            fm.eval()
                            self._frozen_models[pool_idx] = fm

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

                    # Phase 5: Batch hand action sequences
                    batch_ha_seq = torch.cat([s.get('hand_action_seq', torch.zeros(1, MAX_HAND_ACTIONS, ACTION_FEATURE_DIM)) for s in sub_states], dim=0).to(self.device)
                    batch_ha_len = torch.cat([s.get('hand_action_len', torch.ones(1, dtype=torch.long)) for s in sub_states], dim=0)
                    batch_actor_prof = torch.zeros(len(sub_states), MAX_HAND_ACTIONS, PROFILE_DIM, device=self.device)

                    target_model = self.policy if model_id == 'hero' else self._frozen_models.get(model_id, self._frozen_template)
                    
                    with torch.no_grad():
                        output = target_model(
                            hole_cards=batch_hole, community_cards=batch_comm, numeric_features=batch_num,
                            opponent_embeddings=batch_opp_embed, opponent_stats=batch_opp_stats, own_stats=batch_own,
                            opponent_game_state=batch_opp_gs,
                            action_mask=batch_mask, sizing_mask=batch_s_mask, opponent_mask=batch_opp_mask,
                            hand_action_seq=batch_ha_seq, hand_action_len=batch_ha_len,
                            actor_profiles_seq=batch_actor_prof,
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


    def _count_action_distribution(self, experiences: List[Experience]) -> Dict[str, float]:
        """Compute action choice rates WHEN each action was legal (conditional %)."""
        from model.action_space import POT_FRACTIONS
        allin_idx = len(POT_FRACTIONS) - 1

        # chosen[action] = times chosen, legal[action] = times it was legal
        chosen = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'allin': 0}
        legal = {'fold': 0, 'check': 0, 'call': 0, 'raise': 0, 'allin': 0}
        deep_raise_count = 0   # raises when hero_stack_bb > 50 (any sizing)
        deep_allin_count = 0   # raises when hero_stack_bb > 50 AND sizing = all-in

        for e in experiences:
            mask = e.action_mask.squeeze(0)  # (4,)
            if mask[ActionIndex.FOLD]:
                legal['fold'] += 1
            if mask[ActionIndex.CHECK]:
                legal['check'] += 1
            if mask[ActionIndex.CALL]:
                legal['call'] += 1
            if mask[ActionIndex.RAISE]:
                legal['raise'] += 1
                legal['allin'] += 1  # all-in is a subset of raise legality

            if e.action_idx == ActionIndex.FOLD:
                chosen['fold'] += 1
            elif e.action_idx == ActionIndex.CHECK:
                chosen['check'] += 1
            elif e.action_idx == ActionIndex.CALL:
                chosen['call'] += 1
            elif e.action_idx == ActionIndex.RAISE:
                if e.sizing_idx == allin_idx:
                    chosen['allin'] += 1
                else:
                    chosen['raise'] += 1
                # Track deep-stack sizing discipline
                if e.hero_stack_bb > 50:
                    deep_raise_count += 1
                    if e.sizing_idx == allin_idx:
                        deep_allin_count += 1

        # Return conditional % (chosen / legal) + deep-stack all-in rate
        rates = {}
        for k in chosen:
            rates[k] = (chosen[k] / legal[k] * 100) if legal[k] > 0 else 0.0
        # "When deep (>50bb) and raising, what % is all-in?" — healthy = 10-20%
        rates['deep_ai'] = (deep_allin_count / max(deep_raise_count, 1)) * 100
        return rates

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
        
        for e in experiences:
            emb = e.opponent_embeddings
            stat = e.opponent_stats
            gs = e.opponent_game_state
            
            curr_opps = emb.shape[1]
            pad_len = max_opps - curr_opps
            
            if pad_len > 0:
                emb_pad = torch.zeros(1, pad_len, emb.shape[2])
                stat_pad = torch.zeros(1, pad_len, stat.shape[2])
                gs_pad = torch.zeros(1, pad_len, OPP_GAME_STATE_DIM)
                
                emb = torch.cat([emb, emb_pad], dim=1)
                stat = torch.cat([stat, stat_pad], dim=1)
                gs = torch.cat([gs, gs_pad], dim=1)
                
                mask = torch.tensor([[False]*curr_opps + [True]*pad_len], dtype=torch.bool)
            else:
                mask = torch.tensor([[False]*curr_opps], dtype=torch.bool)
                
            opp_emb_list.append(emb)
            opp_stat_list.append(stat)
            opp_gs_list.append(gs)
            opp_mask_list.append(mask)
            
        opp_embeds = torch.cat(opp_emb_list, dim=0)
        opp_stats = torch.cat(opp_stat_list, dim=0)
        opp_gs = torch.cat(opp_gs_list, dim=0)
        opp_masks = torch.cat(opp_mask_list, dim=0)
        
        # 3. Stack hand action history (already padded to MAX_HAND_ACTIONS)
        hand_action_seqs = torch.cat([e.hand_action_seq for e in experiences], dim=0)
        hand_action_lens = torch.cat([e.hand_action_len for e in experiences], dim=0)
        actor_profiles_seqs = torch.cat([e.actor_profiles_seq for e in experiences], dim=0)

        # 4. Stack targets (keep on CPU)
        action_t = torch.tensor([e.action_idx for e in experiences], dtype=torch.long)
        sizing_t = torch.tensor([e.sizing_idx for e in experiences], dtype=torch.long)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], dtype=torch.float32)
        old_action_log_probs = torch.tensor([e.action_log_prob for e in experiences], dtype=torch.float32)
        old_sizing_log_probs = torch.tensor([e.sizing_log_prob for e in experiences], dtype=torch.float32)

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
            advantages, returns = self._compute_gae(trajectory)
            for local_idx, global_idx in enumerate(indices):
                gae_advantages[global_idx] = advantages[local_idx]
                gae_returns[global_idx] = returns[local_idx]

        # Per-hand stack normalization: divide advantages by effective stack
        # so 20bb and 200bb hands contribute equally to policy gradient
        for hid, indices in hand_groups.items():
            eff_stack = experiences[indices[0]].effective_stack_bb
            for idx in indices:
                gae_advantages[idx] /= eff_stack

        # Soft cross-batch normalization on top (stabilizes gradient scale)
        adv_std = max(gae_advantages.std().item(), 0.01)
        gae_advantages = gae_advantages / adv_std

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
            'action_t': action_t,
            'sizing_t': sizing_t,
            'old_log_probs': old_log_probs,
            'old_action_log_probs': old_action_log_probs,
            'old_sizing_log_probs': old_sizing_log_probs,
            'gae_advantages': gae_advantages,
            'gae_returns': gae_returns,
        }

    def _compute_ppo_loss_minibatch(self, data: dict, indices: List[int]) -> float:
        """Compute PPO loss on a mini-batch specified by indices."""
        idx = torch.tensor(indices, dtype=torch.long)

        # Slice all tensors on CPU, THEN send to device exactly shaped to avoid MPS cache bloat!
        hole_cards = data['hole_cards'][idx].to(self.device)
        community = data['community'][idx].to(self.device)
        numeric = data['numeric'][idx].to(self.device)
        own_stats = data['own_stats'][idx].to(self.device)
        action_masks = data['action_masks'][idx].to(self.device)
        sizing_masks = data['sizing_masks'][idx].to(self.device)
        opp_embeds = data['opp_embeds'][idx].to(self.device)
        opp_stats = data['opp_stats'][idx].to(self.device)
        opp_gs = data['opp_gs'][idx].to(self.device)
        opp_masks = data['opp_masks'][idx].to(self.device)
        hand_action_seqs = data['hand_action_seqs'][idx].to(self.device)
        hand_action_lens = data['hand_action_lens'][idx]
        actor_profiles_seqs = data['actor_profiles_seqs'][idx].to(self.device)
        action_t = data['action_t'][idx].to(self.device)
        sizing_t = data['sizing_t'][idx].to(self.device)
        old_log_probs = data['old_log_probs'][idx].to(self.device)
        old_action_log_probs = data['old_action_log_probs'][idx].to(self.device)
        old_sizing_log_probs = data['old_sizing_log_probs'][idx].to(self.device)
        gae_advantages = data['gae_advantages'][idx].to(self.device)
        gae_returns = data['gae_returns'][idx].to(self.device)

        # Forward pass
        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embeds,
            opponent_stats=opp_stats,
            own_stats=own_stats,
            opponent_game_state=opp_gs,
            action_mask=action_masks,
            sizing_mask=sizing_masks,
            opponent_mask=opp_masks,
            hand_action_seq=hand_action_seqs,
            hand_action_len=hand_action_lens,
            actor_profiles_seq=actor_profiles_seqs,
        )
        
        # ── Decoupled PPO: action type and sizing get independent credit ──
        
        # Action type PPO (all experiences)
        dist = Categorical(output.action_type_probs)
        action_log_probs = dist.log_prob(action_t)
        action_entropy = dist.entropy().mean()

        is_raise = (action_t == ActionIndex.RAISE)

        # Action head ratio
        action_ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = action_ratio * gae_advantages
        surr2 = torch.clamp(action_ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * gae_advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Sizing head PPO — only computed on raise experiences to avoid NaN
        # from all-masked (-inf) sizing logits on non-raise actions
        if is_raise.any():
            raise_logits = output.bet_size_logits[is_raise]
            raise_sizing_t = sizing_t[is_raise]
            raise_old_slp = old_sizing_log_probs[is_raise]
            raise_adv = gae_advantages[is_raise]

            sizing_dist = Categorical(logits=raise_logits)
            sizing_log_probs = sizing_dist.log_prob(raise_sizing_t)
            sizing_entropy = sizing_dist.entropy().mean()

            sizing_ratio = torch.exp(sizing_log_probs - raise_old_slp)
            s_surr1 = sizing_ratio * raise_adv
            s_surr2 = torch.clamp(sizing_ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * raise_adv
            sizing_loss = -torch.min(s_surr1, s_surr2).mean()
        else:
            sizing_loss = torch.tensor(0.0, device=self.device)
            sizing_entropy = torch.tensor(0.0, device=self.device)
        
        entropy = action_entropy + sizing_entropy
        
        value_pred = output.value.squeeze(-1)
        value_loss = torch.nn.functional.smooth_l1_loss(value_pred, gae_returns)
        
        entropy_coef = self._get_entropy_coef()
        loss = (
            action_loss
            + 0.5 * sizing_loss  # sizing weighted slightly less
            + self.config.value_coef * value_loss
            - entropy_coef * entropy
        )
        
        loss.backward()
        
        return loss.item()

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
                    pool_total = len(self.opponent_pool_recent) + len(self.opponent_pool_archive)
                    print(f"    [Frozen sync] Pool: {len(self.opponent_pool_recent)} recent + {len(self.opponent_pool_archive)} archive = {pool_total}")

            # Batched simulation: all hands run as generators with batched GPU calls
            all_exp, total_reward = self._run_batched_epoch()

            avg_reward = (total_reward / self.config.hands_per_epoch) * 100.0

            # Action distribution (conditional: % chosen when legal)
            action_pcts = self._count_action_distribution(all_exp)

            # Pre-compute GAE and batch tensors ONCE
            self.policy.train()
            self.opponent_encoder.train()
            ppo_data = self._precompute_ppo_data(all_exp)

            # Mini-batch PPO update
            total_loss = 0.0
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
                        loss_val = self._compute_ppo_loss_minibatch(ppo_data, mb_indices)
                        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                        nn.utils.clip_grad_norm_(self.opponent_encoder.parameters(), 1.0)
                        self.optimizer.step()
                        total_loss += loss_val
                        num_updates += 1

            avg_loss = total_loss / max(num_updates, 1)

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
                    f"Loss: {avg_loss:.4f} ({num_updates} updates) | "
                    f"F/Ch/Ca/R/AI: {action_pcts['fold']:.0f}/{action_pcts['check']:.0f}/{action_pcts['call']:.0f}/{action_pcts['raise']:.0f}/{action_pcts['allin']:.0f}% "
                    f"DAI:{deep_ai:.0f}%"
                )

            if epoch_callback:
                epoch_callback(self, epoch + 1, metrics)

        return metrics

