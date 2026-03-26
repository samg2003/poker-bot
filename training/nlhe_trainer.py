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
from model.policy_network import PolicyNetwork
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



class NLHESelfPlayTrainer:
    """
    Balanced self-play trainer on full NLHE with:
    - Frozen opponent pool (hero-only training)
    - Per-hand trajectory GAE for credit assignment
    - Real opponent embeddings from action history tracking
    - Exploration floor to prevent local minima collapse
    - Auto GPU/MPS acceleration
    - Optional search-guided expert iteration
    """

    def __init__(self, config: Optional[NLHETrainingConfig] = None, seed: int = 42):
        self.config = config or NLHETrainingConfig()
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

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
        self._frozen_template = copy.deepcopy(self.policy).to('cpu')
        self.frozen_opponent_encoder = copy.deepcopy(self.opponent_encoder).to('cpu')
        for p in self._frozen_template.parameters():
            p.requires_grad = False
        for p in self.frozen_opponent_encoder.parameters():
            p.requires_grad = False

        # Two-tier opponent pool: recent (FIFO) + archive (old preserved)
        self.opponent_pool_recent: List[dict] = [
            {k: v.cpu() for k, v in self.policy.state_dict().items()}
        ]
        self.opponent_pool_archive: List[dict] = []
        # Per-table frozen models: {pool_idx: model} — built at table setup
        self._frozen_models: Dict[int, 'PolicyNetwork'] = {}

        # ── Opponent tracking ─────────────────────────────────
        # action_histories[player_id] = list of (action_type, bet_frac, pot, stack, street)
        self.action_histories: Dict[int, List[torch.Tensor]] = {}
        self.stat_tracker = StatTracker()
        self.hands_since_reset = 0
        self.next_reset_at = self.rng.randint(*self.config.history_reset_interval)
        self.hands_since_personality_reset = 0
        self.next_personality_reset_at = 50  # Soft reset every 50 hands

        # Personality curriculum — initially no personalities (pure self-play GTO)
        self.current_epoch = 0
        self.total_epochs = 1  # set by train() for annealing schedules
        self.table_personalities: List[SituationalPersonality] = []

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
        Per-seat loading happens in _play_hand_gen.
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

    def _record_action(self, player_id: int, action_type: int,
                       bet_frac: float, pot: float, stack: float, street: int):
        """Record an observed action for a player."""
        if player_id not in self.action_histories:
            self.action_histories[player_id] = []

        token = encode_action(action_type, bet_frac, pot, stack, street)
        self.action_histories[player_id].append(token)

        # Cap at 16 actions (rolling window — last ~4 hands of context)
        if len(self.action_histories[player_id]) > 16:
            self.action_histories[player_id] = self.action_histories[player_id][-16:]

    def _get_opponent_embedding(self, player_id: int) -> torch.Tensor:
        """
        Get opponent embedding from action history (cached per hand).
        Returns: (1, embed_dim) tensor.
        """
        # Cache keyed by player_id — cleared once per hand in _play_hand_gen
        if player_id in self._opp_embed_cache:
            return self._opp_embed_cache[player_id]

        history = self.action_histories.get(player_id, [])

        if not history:
            emb = self.opponent_encoder.encode_empty(1, device=str(self.device))
        else:
            seq = torch.stack(history).unsqueeze(0)
            seq = self._to(seq)
            with torch.no_grad():
                emb = self.opponent_encoder(seq)
        
        self._opp_embed_cache[player_id] = emb.detach()
        return emb

    def _get_all_opponent_embeddings(self, hero_id: int, num_players: int) -> torch.Tensor:
        """
        Get embeddings for all opponents (from hero's perspective).
        Batches all uncached opponent encoder calls into a single forward pass.
        Returns: (1, num_opp, embed_dim) tensor.
        """
        opp_ids = [pid for pid in range(num_players) if pid != hero_id]
        if not opp_ids:
            return self.opponent_encoder.encode_empty(1, device=str(self.device)).unsqueeze(1)

        # Split into cached and uncached
        cached_embs = {}
        uncached = []  # (pid, history_tensors)
        for pid in opp_ids:
            if pid in self._opp_embed_cache:
                cached_embs[pid] = self._opp_embed_cache[pid]
            else:
                history = self.action_histories.get(pid, [])
                uncached.append((pid, history))

        # Batch encode all uncached opponents in one forward pass
        if uncached:
            # Separate empty vs non-empty histories
            with_history = [(pid, h) for pid, h in uncached if h]
            without_history = [pid for pid, h in uncached if not h]

            # Encode empty opponents
            for pid in without_history:
                emb = self.opponent_encoder.encode_empty(1, device=str(self.device))
                self._opp_embed_cache[pid] = emb.detach()
                cached_embs[pid] = emb

            # Batch encode opponents with history
            if with_history:
                seqs = [torch.stack(h) for _, h in with_history]  # list of (seq_len, features)
                max_len = max(s.shape[0] for s in seqs)
                feat_dim = seqs[0].shape[1]

                # Pad to same length and build mask
                padded = torch.zeros(len(seqs), max_len, feat_dim, device=self.device)
                mask = torch.ones(len(seqs), max_len, device=self.device, dtype=torch.bool)  # True = masked
                for i, s in enumerate(seqs):
                    s = self._to(s)
                    padded[i, :s.shape[0]] = s
                    mask[i, :s.shape[0]] = False  # unmask real positions

                with torch.no_grad():
                    embs = self.opponent_encoder(padded, mask=mask)  # (N, embed_dim)

                for i, (pid, _) in enumerate(with_history):
                    emb = embs[i:i+1].detach()  # (1, embed_dim)
                    self._opp_embed_cache[pid] = emb
                    cached_embs[pid] = emb

        # Assemble in order
        ordered = [cached_embs[pid] for pid in opp_ids]
        return torch.cat(ordered, dim=0).unsqueeze(0)  # (1, num_opp, embed_dim)

    def _get_opponent_stats(self, hero_id: int, num_players: int) -> torch.Tensor:
        """Get HUD stats for all opponents. Returns (1, num_opp, stat_dim)."""
        stats = []
        for pid in range(num_players):
            if pid == hero_id:
                continue
            stats.append(self.stat_tracker.get_stats(pid))

        if not stats:
            return torch.zeros(1, 1, NUM_STAT_FEATURES, device=self.device)

        return torch.stack(stats).unsqueeze(0).to(self.device)  # (1, num_opp, stat_dim)

    def _get_personality_gto_fraction(self) -> float:
        """Graduated personality curriculum schedule."""
        epoch = self.current_epoch
        if epoch < 20:
            return 1.0  # 100% GTO until epoch 20
        return 0.6      # 60% GTO / 40% Personalities after epoch 20

    def _maybe_reset_histories(self):
        """Periodically reset opponent histories and personalities (simulate new table)."""
        self.hands_since_reset += 1
        self.hands_since_personality_reset += 1

        # Hard reset: total flush of histories and stats (e.g. 300-500 hands)
        if self.hands_since_reset >= self.next_reset_at:
            self.action_histories.clear()
            self.stat_tracker.reset()
            self.hands_since_reset = 0
            self.next_reset_at = self.rng.randint(*self.config.history_reset_interval)
            # Reshuffle personalities and per-seat weights when we "sit down at a new table"
            self.table_personalities = []
            self._seat_pool_idx = {}
            self._frozen_models.clear()
            self.hands_since_personality_reset = 0

        # Soft reset: flush personalities only (keep histories for OpponentEncoder to adapt)
        elif self.hands_since_personality_reset >= self.next_personality_reset_at:
            self.table_personalities = []
            self._seat_pool_idx = {}
            self._frozen_models.clear()
            self.hands_since_personality_reset = 0

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
        rel_pos = (player_idx - game_state.dealer_button) % game_state.num_players
        position = rel_pos / max(game_state.num_players - 1, 1)
        street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        street_val = street_map.get(game_state.street, 0) / 3.0
        num_active = sum(1 for pp in game_state.players if pp.is_active)
        current_bet = game_state.current_bet / norm
        min_raise = game_state.min_raise / norm

        numeric = self._to(torch.tensor([[
            pot, own_stack, own_bet, position, street_val,
            game_state.num_players / 9.0, num_active / 9.0,
            current_bet, min_raise,
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

    def _play_hand_gen(self) -> Generator[dict, Tuple[torch.Tensor, float, float], Tuple[List[List[Experience]], float]]:
        """
        Generator for batched GPU inference with balanced self-play.

        Hero (player 0) uses LIVE policy (yielded for batched inference).
        Opponents use FROZEN policy (run inline with no_grad).
        Only hero experiences are collected.

        Yields a dict of encoded tensors for hero decisions.
        Receives (probs, value, sizing) via .send().
        Returns (experiences, reward) via StopIteration.value.
        """
        num_p, stacks = self._sample_table()
        self._opp_embed_cache = {}  # Clear cache for new hand

        # No effective stack normalization — reward in bb/100 units

        # Unique hand ID for GAE grouping
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
        hero_experiences: List[dict] = []  # only hero's decisions
        hero_step_idx = 0

        # Per-seat opponent assignment: each seat gets an independent pool entry
        # Persistent across hands at the same table (cleared on table reset)
        if not hasattr(self, '_seat_pool_idx') or len(self._seat_pool_idx) != num_p - 1:
            self._seat_pool_idx = {}
            for pid in range(1, num_p):
                combined_pool = self._get_combined_pool()
                if combined_pool:
                    self._seat_pool_idx[pid] = self.rng.randint(0, len(combined_pool) - 1)
                else:
                    self._seat_pool_idx[pid] = 0
            # Build frozen models for each unique pool index (one-time per table)
            self._build_table_models(self._seat_pool_idx)
        seat_pool_idx = self._seat_pool_idx

        # Table-level personality decision (persistent across hands):
        # 33% of tables = fully unperturbed (all GTO)
        # 67% of tables = 60% GTO / 40% personality per seat
        if len(self.table_personalities) != num_p:
            if self.current_epoch < 20 or self.rng.random() < 0.33:
                # Fully unperturbed table
                gto_frac = 1.0
            else:
                # Mixed table: 60% GTO, 40% personality
                gto_frac = 0.60
            self.table_personalities = sample_table_personalities(
                num_p, gto_fraction=gto_frac, rng=self.rng
            )

        hand_records = [HandRecord() for _ in range(num_p)]
        epsilon = self._get_epsilon()

        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            if not p.is_active:
                break

            # Encode state
            encoded = self._encode_state(game_state, pid)
            opp_embed = self._get_all_opponent_embeddings(pid, num_p)
            opp_stats = self._get_opponent_stats(pid, num_p)
            own_stats = self._to(self.stat_tracker.get_stats(pid).unsqueeze(0))

            if pid == 0:
                # ── HERO: yield for batched live policy inference ──
                probs, value, sizing_probs = yield {
                    'hole_cards': encoded['hole_cards'],
                    'community_cards': encoded['community_cards'],
                    'numeric_features': encoded['numeric_features'],
                    'opponent_embeddings': opp_embed,
                    'opponent_stats': opp_stats,
                    'own_stats': own_stats,
                    'action_mask': encoded['action_mask'],
                    'sizing_mask': encoded['sizing_mask'],
                    # No personality for hero — plays raw policy
                    '_personality': None,
                    '_situations': [],
                    '_hand_strength': 0.5,
                    '_is_facing_raise': False,
                    '_game_id': id(dealer),
                }

                # Store RAW log_prob from model (before floor) for PPO
                raw_probs = probs.clone()

                # Search-guided action selection (expert iteration)
                if self.search_engine is not None and self.rng.random() < self.config.search_fraction:
                    street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
                    cur_street = street_map.get(game_state.street, 0)
                    pot_bb = game_state.pot / game_state.big_blind

                    if self.search_engine.should_search(probs, pot_bb, cur_street):
                        search_stacks = [pp.stack for pp in game_state.players]
                        search_actions, search_probs = self.search_engine.search(
                            pot=game_state.pot,
                            stacks=search_stacks,
                            board=list(game_state.board),
                            street=cur_street,
                            hero=pid,
                        )
                        refined = torch.zeros(NUM_ACTION_TYPES, device=self.device)
                        for sa, sp in zip(search_actions, search_probs):
                            if sa == 'check':
                                refined[ActionIndex.CHECK] += sp
                            elif sa == 'fold':
                                refined[ActionIndex.FOLD] += sp
                            elif sa == 'call':
                                refined[ActionIndex.CALL] += sp
                            elif sa.startswith('raise_') or sa == 'allin':
                                refined[ActionIndex.RAISE] += sp
                        if refined.sum() > 0:
                            refined = refined / refined.sum()
                            probs = 0.7 * refined + 0.3 * probs.to(self.device)

                # ── Epsilon-greedy exploration (training only) ──
                action_mask_cpu = encoded['action_mask'].squeeze(0).cpu()  # (4,)
                legal_indices = action_mask_cpu.nonzero(as_tuple=True)[0].tolist()

                if self.rng.random() < epsilon and len(legal_indices) > 1:
                    # Random action — explore something the model wouldn't try
                    action_idx = self.rng.choice(legal_indices)
                else:
                    # Model's policy
                    dist = Categorical(probs)
                    action_idx = dist.sample().item()

                # Always compute log_prob from model's OWN distribution for PPO
                raw_dist = Categorical(raw_probs)
                action_lp = raw_dist.log_prob(torch.tensor(action_idx)).item()
                sizing_lp = 0.0

                # Sizing with epsilon-greedy
                sizing_idx = 0
                if action_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
                    sizing_tensor = torch.tensor(sizing_probs, dtype=torch.float32)
                    sizing_mask_1d = encoded['sizing_mask'].squeeze(0).cpu()  # (10,)
                    legal_sizing = sizing_mask_1d.nonzero(as_tuple=True)[0].tolist()

                    if self.rng.random() < epsilon and len(legal_sizing) > 1:
                        # Random sizing — explore different bet amounts
                        sizing_idx = self.rng.choice(legal_sizing)
                    else:
                        s_dist = Categorical(sizing_tensor)
                        sizing_idx = s_dist.sample().item()

                    # Log prob from model's sizing distribution for PPO
                    sizing_lp = Categorical(sizing_tensor).log_prob(torch.tensor(sizing_idx)).item()

                log_prob = action_lp + sizing_lp

                # Store hero experience
                hero_experiences.append({
                    'hole_cards': encoded['hole_cards'].detach(),
                    'community_cards': encoded['community_cards'].detach(),
                    'numeric_features': encoded['numeric_features'].detach(),
                    'opponent_embeddings': opp_embed.detach(),
                    'opponent_stats': opp_stats.detach(),
                    'own_stats': own_stats.detach(),
                    'action_mask': encoded['action_mask'].detach(),
                    'sizing_mask': encoded['sizing_mask'].detach(),
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
                # ── OPPONENT: run through frozen policy (no grad, no yield) ──
                pool_idx = seat_pool_idx.get(pid, 0)
                frozen_model = self._frozen_models.get(pool_idx, self._frozen_template)

                with torch.no_grad():
                    opp_mask = torch.zeros(1, num_p - 1, dtype=torch.bool)
                    output = frozen_model(
                        hole_cards=encoded['hole_cards'].cpu(),
                        community_cards=encoded['community_cards'].cpu(),
                        numeric_features=encoded['numeric_features'].cpu(),
                        opponent_embeddings=opp_embed.cpu(),
                        opponent_stats=opp_stats.cpu(),
                        own_stats=own_stats.cpu(),
                        action_mask=encoded['action_mask'].cpu(),
                        sizing_mask=encoded['sizing_mask'].cpu(),
                        opponent_mask=opp_mask,
                    )
                    probs = output.action_type_probs[0]
                    sizing_probs = torch.softmax(output.bet_size_logits[0], dim=-1).tolist()

                # Apply personality to opponent (hero never gets personality)
                personality = None
                is_facing_raise = game_state.current_bet > 0 and p.bet_this_street < game_state.current_bet
                if pid < len(self.table_personalities):
                    personality = self.table_personalities[pid]
                    if personality is not None:
                        street_map_sit = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
                        situations = detect_situations(
                            street=street_map_sit.get(game_state.street, 0),
                            board_cards=list(game_state.board) if game_state.board else None,
                            stack_bb=p.stack / max(game_state.big_blind, 1),
                            is_facing_raise=is_facing_raise,
                        )
                        # Approximate hand strength (cheap heuristic)
                        c1, c2 = p.hole_cards
                        r1, r2 = c1 // 4, c2 // 4
                        if game_state.street == Street.PREFLOP:
                            if r1 == r2:
                                hand_strength = 0.55 + r1 * 0.035
                            elif max(r1, r2) >= 10:
                                hand_strength = 0.45 + max(r1, r2) * 0.02
                            elif (c1 % 4) == (c2 % 4):
                                hand_strength = 0.3
                            else:
                                hand_strength = 0.15 + max(r1, r2) * 0.01
                        else:
                            paired_board = any(r1 == (bc // 4) or r2 == (bc // 4)
                                               for bc in game_state.board if bc >= 0)
                            if paired_board:
                                hand_strength = 0.65 + max(r1, r2) * 0.02
                            elif max(r1, r2) >= 10:
                                hand_strength = 0.4
                            else:
                                hand_strength = 0.2

                        probs = personality.apply(
                            probs, situations,
                            hand_strength=hand_strength,
                            is_facing_raise=is_facing_raise,
                        )
                        sizing_probs = personality.apply_sizing(sizing_probs, situations)

                # Sample opponent action (no experience stored)
                dist = Categorical(probs)
                action_idx = dist.sample().item()
                sizing_idx = 0
                if action_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
                    s_dist = Categorical(torch.tensor(sizing_probs))
                    sizing_idx = s_dist.sample().item()

            # Decode and apply action
            action = self._decode_action(action_idx, sizing_idx, game_state)
            dealer.apply_action(action)

            # Record action in opponent history (for all players — hero needs to see this)
            pot_for_record = game_state.pot
            bet_frac = action.amount / max(pot_for_record, 1.0) if action.amount > 0 else 0.0
            street_map = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
            self._record_action(
                player_id=pid,
                action_type=self._action_to_type_idx(action),
                bet_frac=bet_frac,
                pot=pot_for_record,
                stack=p.stack,
                street=street_map.get(game_state.street, 0),
            )

            # Track HUD stats
            if action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].vpip = True
            if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].pfr = True

        # Record hand results
        results = dealer.get_results()
        profits = results['profit']
        for pid in range(num_p):
            hand_records[pid].result = profits[pid]
            hand_records[pid].went_to_showdown = (
                game_state.street == Street.SHOWDOWN and
                not game_state.players[pid].is_folded
            )
            self.stat_tracker.record_hand(pid, hand_records[pid])

        self._maybe_reset_histories()

        # ── Hero-only experience: reward in bb/100 ──
        hero_reward = profits[0] / 100.0

        hero_exp_list = []
        for exp_dict in hero_experiences:
            hero_exp_list.append(Experience(
                hole_cards=exp_dict['hole_cards'],
                community_cards=exp_dict['community_cards'],
                numeric_features=exp_dict['numeric_features'],
                opponent_embeddings=exp_dict['opponent_embeddings'],
                opponent_stats=exp_dict['opponent_stats'],
                own_stats=exp_dict['own_stats'],
                action_mask=exp_dict['action_mask'],
                sizing_mask=exp_dict['sizing_mask'],
                action_idx=exp_dict['action_idx'],
                sizing_idx=exp_dict['sizing_idx'],
                log_prob=exp_dict['log_prob'],
                value=exp_dict['value'],
                reward=hero_reward,  # same reward for all steps — GAE distributes credit
                hand_id=hand_id,
                step_idx=exp_dict['step_idx'],
                hero_stack_bb=stacks[0] / self.config.big_blind,
            ))

        # Return as [[hero_exps]] with empty lists for opponents (backward compat)
        all_experiences = [hero_exp_list] + [[] for _ in range(num_p - 1)]
        return all_experiences, hero_reward

    def _run_batched_epoch(self) -> Tuple[List[Experience], float]:
        """
        Run all hands for one epoch with batched GPU inference.
        
        Processes hands in sub-batches (chunks) to limit peak memory usage.
        Each chunk runs all its games to completion before starting the next.
        """
        self.policy.eval()
        self.opponent_encoder.eval()

        # Clear stale embedding cache from previous epoch
        self._opp_embed_cache = {}

        num_hands = self.config.hands_per_epoch
        all_exp: List[Experience] = []
        total_reward = 0.0
        total_finished = 0

        # Process in chunks to limit memory
        chunk_size = min(self.config.batch_chunk_size, num_hands)
        epoch_start = time.time()

        for chunk_start in range(0, num_hands, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_hands)
            chunk_n = chunk_end - chunk_start

            # Start this chunk of games
            games = [self._play_hand_gen() for _ in range(chunk_n)]

            # Prime each generator
            pending = {}
            for i, g in enumerate(games):
                try:
                    state = next(g)
                    pending[i] = (g, state)
                except StopIteration as e:
                    exps, reward = e.value
                    for pexp in exps:
                        all_exp.extend(pexp)
                    total_reward += reward
                    total_finished += 1

            # Run this chunk to completion
            while pending:
                game_indices = list(pending.keys())
                states = [pending[i][1] for i in game_indices]

                # Pad opponent tensors to same shape for batching
                max_opps = max(s['opponent_embeddings'].shape[1] for s in states)
                embed_dim = states[0]['opponent_embeddings'].shape[2]
                stat_dim = states[0]['opponent_stats'].shape[2]

                # Collate into batched tensors
                batch_hole = torch.cat([s['hole_cards'] for s in states], dim=0)
                batch_comm = torch.cat([s['community_cards'] for s in states], dim=0)
                batch_num = torch.cat([s['numeric_features'] for s in states], dim=0)
                batch_mask = torch.cat([s['action_mask'] for s in states], dim=0)
                batch_s_mask = torch.cat([s['sizing_mask'] for s in states], dim=0)
                batch_own = torch.cat([s['own_stats'] for s in states], dim=0)

                padded_opp_embeds = []
                padded_opp_stats = []
                opp_masks = []
                for s in states:
                    oe = s['opponent_embeddings']
                    os_t = s['opponent_stats']
                    n_opp = oe.shape[1]
                    if n_opp < max_opps:
                        pad_e = torch.zeros(1, max_opps - n_opp, embed_dim, device=oe.device)
                        oe = torch.cat([oe, pad_e], dim=1)
                        pad_s = torch.zeros(1, max_opps - n_opp, stat_dim, device=os_t.device)
                        os_t = torch.cat([os_t, pad_s], dim=1)
                    padded_opp_embeds.append(oe)
                    padded_opp_stats.append(os_t)
                    m = torch.zeros(1, max_opps, dtype=torch.bool, device=oe.device)
                    m[0, :n_opp] = True
                    opp_masks.append(m)

                batch_opp_embed = torch.cat(padded_opp_embeds, dim=0)
                batch_opp_stats = torch.cat(padded_opp_stats, dim=0)
                batch_opp_mask = torch.cat(opp_masks, dim=0)

                # Move entire batch to device
                batch_hole = batch_hole.to(self.device)
                batch_comm = batch_comm.to(self.device)
                batch_num = batch_num.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_s_mask = batch_s_mask.to(self.device)
                batch_own = batch_own.to(self.device)
                batch_opp_embed = batch_opp_embed.to(self.device)
                batch_opp_stats = batch_opp_stats.to(self.device)
                batch_opp_mask = batch_opp_mask.to(self.device)

                # Batched GPU forward pass
                with torch.no_grad():
                    output = self.policy(
                        hole_cards=batch_hole,
                        community_cards=batch_comm,
                        numeric_features=batch_num,
                        opponent_embeddings=batch_opp_embed,
                        opponent_stats=batch_opp_stats,
                        own_stats=batch_own,
                        action_mask=batch_mask,
                        sizing_mask=batch_s_mask,
                        opponent_mask=batch_opp_mask,
                    )

                # Distribute results back to each game
                for batch_idx, game_idx in enumerate(game_indices):
                    g, state_dict = pending[game_idx]

                    probs = output.action_type_probs[batch_idx].cpu()
                    value = output.value[batch_idx, 0].item()
                    
                    sizing_logits = output.bet_size_logits[batch_idx]
                    sizing_probs = torch.softmax(sizing_logits, dim=-1).cpu().tolist()

                    # Hero always has _personality=None — skip post-processing
                    # (opponent personalities are applied inline in _play_hand_gen)

                    try:
                        new_state = g.send((probs, value, sizing_probs))
                        pending[game_idx] = (g, new_state)
                    except StopIteration as e:
                        exps, reward = e.value
                        for pexp in exps:
                            all_exp.extend(pexp)
                        total_reward += reward
                        del pending[game_idx]
                        total_finished += 1

            # Verbose progress per chunk
            if self.config.verbose and num_hands > chunk_size:
                elapsed = time.time() - epoch_start
                hands_done = chunk_end
                hands_per_s = hands_done / elapsed if elapsed > 0 else 0
                remaining = num_hands - hands_done
                eta = remaining / hands_per_s if hands_per_s > 0 else 0
                print(f"    {hands_done}/{num_hands} hands ({elapsed:.1f}s, {hands_per_s:.1f} h/s, ETA {eta:.0f}s)")

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

        # 1. Stack scalar/fixed-size features
        hole_cards = torch.cat([e.hole_cards.to(self.device) for e in experiences], dim=0)
        community = torch.cat([e.community_cards.to(self.device) for e in experiences], dim=0)
        numeric = torch.cat([e.numeric_features.to(self.device) for e in experiences], dim=0)
        own_stats = torch.cat([e.own_stats.to(self.device) for e in experiences], dim=0)
        action_masks = torch.cat([e.action_mask.to(self.device) for e in experiences], dim=0)
        sizing_masks = torch.cat([e.sizing_mask.to(self.device) for e in experiences], dim=0)
        
        # 2. Pad opponent sequence arrays
        max_opps = max([e.opponent_embeddings.shape[1] for e in experiences])
        
        opp_emb_list = []
        opp_stat_list = []
        opp_mask_list = []
        
        for e in experiences:
            curr_opps = e.opponent_embeddings.shape[1]
            pad_len = max_opps - curr_opps
            
            emb = e.opponent_embeddings.to(self.device)
            stat = e.opponent_stats.to(self.device)
            
            if pad_len > 0:
                emb_pad = torch.zeros(1, pad_len, emb.shape[2], device=self.device)
                stat_pad = torch.zeros(1, pad_len, stat.shape[2], device=self.device)
                
                emb = torch.cat([emb, emb_pad], dim=1)
                stat = torch.cat([stat, stat_pad], dim=1)
                
                mask = torch.tensor([[False]*curr_opps + [True]*pad_len], device=self.device, dtype=torch.bool)
            else:
                mask = torch.tensor([[False]*curr_opps], device=self.device, dtype=torch.bool)
                
            opp_emb_list.append(emb)
            opp_stat_list.append(stat)
            opp_mask_list.append(mask)
            
        opp_embeds = torch.cat(opp_emb_list, dim=0)
        opp_stats = torch.cat(opp_stat_list, dim=0)
        opp_masks = torch.cat(opp_mask_list, dim=0)
        
        # 3. Stack targets
        action_t = torch.tensor([e.action_idx for e in experiences], device=self.device, dtype=torch.long)
        sizing_t = torch.tensor([e.sizing_idx for e in experiences], device=self.device, dtype=torch.long)
        old_log_probs = torch.tensor([e.log_prob for e in experiences], device=self.device, dtype=torch.float32)
        old_action_log_probs = torch.tensor([e.action_log_prob for e in experiences], device=self.device, dtype=torch.float32)
        old_sizing_log_probs = torch.tensor([e.sizing_log_prob for e in experiences], device=self.device, dtype=torch.float32)

        # 4. Compute GAE advantages and returns ONCE (using original values)
        hand_groups: Dict[int, List[int]] = {}
        for idx, e in enumerate(experiences):
            hid = e.hand_id
            if hid not in hand_groups:
                hand_groups[hid] = []
            hand_groups[hid].append(idx)

        gae_advantages = torch.zeros(len(experiences), device=self.device)
        gae_returns = torch.zeros(len(experiences), device=self.device)

        for hid, indices in hand_groups.items():
            indices.sort(key=lambda i: experiences[i].step_idx)
            trajectory = [experiences[i] for i in indices]
            advantages, returns = self._compute_gae(trajectory)
            for local_idx, global_idx in enumerate(indices):
                gae_advantages[global_idx] = advantages[local_idx]
                gae_returns[global_idx] = returns[local_idx]

        # Soft advantage normalization: scale magnitude without changing sign
        # Floor of 1.0 lets big mistakes teach louder, just not 900x louder
        adv_std = max(gae_advantages.std().item(), 1.0)
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
            'opp_masks': opp_masks,
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
        idx = torch.tensor(indices, device=self.device, dtype=torch.long)

        # Slice all tensors by indices
        hole_cards = data['hole_cards'][idx]
        community = data['community'][idx]
        numeric = data['numeric'][idx]
        own_stats = data['own_stats'][idx]
        action_masks = data['action_masks'][idx]
        sizing_masks = data['sizing_masks'][idx]
        opp_embeds = data['opp_embeds'][idx]
        opp_stats = data['opp_stats'][idx]
        opp_masks = data['opp_masks'][idx]
        action_t = data['action_t'][idx]
        sizing_t = data['sizing_t'][idx]
        old_log_probs = data['old_log_probs'][idx]
        old_action_log_probs = data['old_action_log_probs'][idx]
        old_sizing_log_probs = data['old_sizing_log_probs'][idx]
        gae_advantages = data['gae_advantages'][idx]
        gae_returns = data['gae_returns'][idx]

        # Forward pass
        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embeds,
            opponent_stats=opp_stats,
            own_stats=own_stats,
            action_mask=action_masks,
            sizing_mask=sizing_masks,
            opponent_mask=opp_masks,
        )
        
        # ── Decoupled PPO: action type and sizing get independent credit ──
        
        # Action type PPO (all experiences)
        dist = Categorical(output.action_type_probs)
        action_log_probs = dist.log_prob(action_t)
        action_entropy = dist.entropy().mean()

        is_raise = (action_t == ActionIndex.RAISE)
        
        sizing_dist = Categorical(logits=output.bet_size_logits)
        sizing_log_probs = sizing_dist.log_prob(sizing_t)
        sizing_entropy = sizing_dist.entropy()

        # Separate old log probs stored from collection — no approximation needed
        
        # Action head ratio
        action_ratio = torch.exp(action_log_probs - old_action_log_probs)
        
        surr1 = action_ratio * gae_advantages
        surr2 = torch.clamp(action_ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * gae_advantages
        action_loss = -torch.min(surr1, surr2).mean()

        # Sizing head PPO (only raise experiences, independent ratio)
        sizing_loss = torch.tensor(0.0, device=self.device)
        if is_raise.any():
            raise_sizing_log = sizing_log_probs[is_raise]
            raise_advantages = gae_advantages[is_raise]
            old_sizing_lp = old_sizing_log_probs[is_raise]
            
            sizing_ratio = torch.exp(raise_sizing_log - old_sizing_lp)
            s_surr1 = sizing_ratio * raise_advantages
            s_surr2 = torch.clamp(sizing_ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * raise_advantages
            sizing_loss = -torch.min(s_surr1, s_surr2).mean()
        
        entropy = action_entropy + (sizing_entropy * is_raise.float()).mean()
        
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

