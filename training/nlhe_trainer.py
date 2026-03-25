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
    ACTION_FEATURE_DIM, encode_action,
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
    action_idx: int                   # chosen action
    log_prob: float                   # log prob of chosen action
    value: float                      # value estimate
    reward: float                     # final reward (set after hand ends)


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
    gamma: float = 1.0
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.02
    value_coef: float = 0.5

    # Self-play
    hands_per_epoch: int = 128
    batch_chunk_size: int = 500  # Max simultaneous games per sub-batch

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
    Self-play trainer on full NLHE with:
    - Real opponent embeddings from action history tracking
    - Auto GPU/MPS acceleration
    - Optional search-guided expert iteration
    """

    def __init__(self, config: Optional[NLHETrainingConfig] = None, seed: int = 42):
        self.config = config or NLHETrainingConfig()
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # ── Device ────────────────────────────────────────────
        self.device = self._resolve_device(self.config.device)

        # ── Models ────────────────────────────────────────────
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

        # ── Opponent tracking ─────────────────────────────────
        # action_histories[player_id] = list of (action_type, bet_frac, pot, stack, street)
        self.action_histories: Dict[int, List[torch.Tensor]] = {}
        self.stat_tracker = StatTracker()
        self.hands_since_reset = 0
        self.next_reset_at = self.rng.randint(*self.config.history_reset_interval)

        # Personality curriculum — initially no personalities (pure self-play GTO)
        self.current_epoch = 0
        self.table_personalities: List[SituationalPersonality] = []

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
        Returns: (1, num_opp, embed_dim) tensor.
        """
        opp_embeds = []
        for pid in range(num_players):
            if pid == hero_id:
                continue
            opp_embeds.append(self._get_opponent_embedding(pid))

        if not opp_embeds:
            return self.opponent_encoder.encode_empty(1, device=str(self.device)).unsqueeze(1)

        return torch.cat(opp_embeds, dim=0).unsqueeze(0)  # (1, num_opp, embed_dim)

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
        if epoch < 10:
            return 1.0      # 0% personalities — pure GTO self-play foundation
        elif epoch < 20:
            return 0.875    # 12.5% (1/8) opponents get personalities
        elif epoch < 30:
            return 0.75     # 25% (1/4) opponents get personalities
        else:
            return 0.667    # 33% (1/3) opponents get personalities

    def _maybe_reset_histories(self):
        """Periodically reset opponent histories and personalities (simulate new table)."""
        self.hands_since_reset += 1
        if self.hands_since_reset >= self.next_reset_at:
            self.action_histories.clear()
            self.stat_tracker.reset()
            self.hands_since_reset = 0
            self.next_reset_at = self.rng.randint(*self.config.history_reset_interval)
            # Reshuffle personalities when we "sit down at a new table"
            self.table_personalities = []

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

        return {
            'hole_cards': hole,
            'community_cards': community,
            'numeric_features': numeric,
            'action_mask': action_mask,
        }

    def _decode_action(self, action_idx: int, bet_sizing: float,
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
            min_r = game_state.get_min_raise_to()
            max_r = game_state.get_max_raise_to()
            raise_to = min_r + bet_sizing * (max_r - min_r)
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
        Generator version of _play_hand for batched GPU inference.
        
        Yields a dict of encoded tensors when needing a policy decision.
        Receives (probs, value, sizing) via .send().
        Returns (experiences, reward) via StopIteration.value.
        """
        num_p, stacks = self._sample_table()
        self._opp_embed_cache = {}  # Clear cache for new hand

        dealer = Dealer(
            num_players=num_p,
            stacks=stacks,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            dealer_button=self.rng.randint(0, num_p - 1),
            seed=self.rng.randint(0, 2**31),
        )

        game_state = dealer.start_hand()
        experiences: List[List[dict]] = [[] for _ in range(num_p)]

        # Sample table personalities
        gto_frac = self._get_personality_gto_fraction()
        if len(self.table_personalities) != num_p:
            self.table_personalities = sample_table_personalities(
                num_p, gto_fraction=gto_frac, rng=self.rng
            )

        hand_records = [HandRecord() for _ in range(num_p)]

        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            if not p.is_active:
                break

            # Encode state (on CPU — device placement happens in batch)
            encoded = self._encode_state(game_state, pid)
            opp_embed = self._get_all_opponent_embeddings(pid, num_p)
            opp_stats = self._get_opponent_stats(pid, num_p)
            own_stats = self._to(self.stat_tracker.get_stats(pid).unsqueeze(0))

            # Build personality context for post-processing
            personality = None
            situations = []
            if pid < len(self.table_personalities):
                personality = self.table_personalities[pid]
                street_map_sit = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
                situations = detect_situations(
                    street=street_map_sit.get(game_state.street, 0),
                    board_cards=list(game_state.board) if game_state.board else None,
                    stack_bb=p.stack / max(game_state.big_blind, 1),
                )

            # YIELD — pause game, wait for batched GPU result
            probs, value, sizing = yield {
                'hole_cards': encoded['hole_cards'],
                'community_cards': encoded['community_cards'],
                'numeric_features': encoded['numeric_features'],
                'opponent_embeddings': opp_embed,
                'opponent_stats': opp_stats,
                'own_stats': own_stats,
                'action_mask': encoded['action_mask'],
                # context for post-processing (not sent to GPU)
                '_personality': personality,
                '_situations': situations,
                '_game_id': id(dealer),
            }

            # Note: personality perturbation is applied by _run_batched_epoch before .send()

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
                    # Map search result to 4-way action distribution
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

                    # Blend: 70% search, 30% policy (smooth transition)
                    if refined.sum() > 0:
                        refined = refined / refined.sum()
                        probs = 0.7 * refined + 0.3 * probs.to(self.device) # Ensure probs is on device for blending

            # Sample action (probs is on CPU from orchestrator)
            dist = Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()

            # Store experience (detach to prevent double backward in PPO)
            experiences[pid].append({
                'hole_cards': encoded['hole_cards'].detach(),
                'community_cards': encoded['community_cards'].detach(),
                'numeric_features': encoded['numeric_features'].detach(),
                'opponent_embeddings': opp_embed.detach(),
                'opponent_stats': opp_stats.detach(),
                'own_stats': own_stats.detach(),
                'action_mask': encoded['action_mask'].detach(),
                'action_idx': action_idx,
                'log_prob': log_prob,
                'value': value,
            })

            # Decode and apply action
            action = self._decode_action(action_idx, sizing, game_state)
            dealer.apply_action(action)

            # Record action in opponent history
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

        # Convert to Experience objects
        all_experiences = []
        hero_reward = 0.0
        for pid in range(num_p):
            player_exp = []
            reward = (profits[pid] / self.config.big_blind) / 100.0
            if pid == 0:
                hero_reward = reward
            for exp_dict in experiences[pid]:
                player_exp.append(Experience(
                    hole_cards=exp_dict['hole_cards'],
                    community_cards=exp_dict['community_cards'],
                    numeric_features=exp_dict['numeric_features'],
                    opponent_embeddings=exp_dict['opponent_embeddings'],
                    opponent_stats=exp_dict['opponent_stats'],
                    own_stats=exp_dict['own_stats'],
                    action_mask=exp_dict['action_mask'],
                    action_idx=exp_dict['action_idx'],
                    log_prob=exp_dict['log_prob'],
                    value=exp_dict['value'],
                    reward=reward,
                ))
            all_experiences.append(player_exp)

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
                        opponent_mask=batch_opp_mask,
                    )

                # Distribute results back to each game
                for batch_idx, game_idx in enumerate(game_indices):
                    g, state_dict = pending[game_idx]

                    probs = output.action_type_probs[batch_idx].cpu()
                    value = output.value[batch_idx, 0].item()
                    sizing = output.bet_sizing[batch_idx, 0].item()

                    personality = state_dict['_personality']
                    situations = state_dict['_situations']
                    if personality is not None:
                        probs = personality.apply(probs, situations, hand_strength=0.5)

                    try:
                        new_state = g.send((probs, value, sizing))
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

    # ─────────────────────────────────────────────────────────
    # PPO loss
    # ─────────────────────────────────────────────────────────

    def _compute_ppo_loss(self, experiences: List[Experience]) -> float:
        """Compute PPO loss from collected experience using batched operations."""
        if not experiences:
            return 0.0

        self.policy.train()
        self.opponent_encoder.train()
        
        # 1. Stack scalar/fixed-size features
        hole_cards = torch.cat([e.hole_cards.to(self.device) for e in experiences], dim=0)
        community = torch.cat([e.community_cards.to(self.device) for e in experiences], dim=0)
        numeric = torch.cat([e.numeric_features.to(self.device) for e in experiences], dim=0)
        own_stats = torch.cat([e.own_stats.to(self.device) for e in experiences], dim=0)
        action_masks = torch.cat([e.action_mask.to(self.device) for e in experiences], dim=0)
        
        # 2. Pad opponent sequence arrays (shape is [1, num_opp, dim])
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
        old_log_probs = torch.tensor([e.log_prob for e in experiences], device=self.device, dtype=torch.float32)
        rewards = torch.tensor([e.reward for e in experiences], device=self.device, dtype=torch.float32)
        old_values = torch.tensor([e.value for e in experiences], device=self.device, dtype=torch.float32)
        
        # 4. Batched forward pass
        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embeds,
            opponent_stats=opp_stats,
            own_stats=own_stats,
            action_mask=action_masks,
            opponent_mask=opp_masks,
        )
        
        # 5. Compute PPO stats and loss
        dist = Categorical(output.action_type_probs)
        new_log_probs = dist.log_prob(action_t)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        advantages = rewards - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_pred = output.value.squeeze(-1)
        value_loss = ((value_pred - rewards) ** 2).mean()
        
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )
        
        loss.backward()
        
        return loss.item()

    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────

    def train(self, num_epochs: int = 100, epoch_callback=None) -> Dict[str, List[float]]:
        """Run NLHE self-play training with batched GPU inference.
        
        Args:
            num_epochs: Number of epochs to train.
            epoch_callback: Optional function(trainer, epoch, metrics) called after each epoch.
        """
        metrics = {
            'epoch_reward': [],
            'epoch_loss': [],
        }

        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            # Batched simulation: all hands run as generators with batched GPU calls
            all_exp, total_reward = self._run_batched_epoch()

            avg_reward = (total_reward / self.config.hands_per_epoch) * 100.0

            # PPO update (fully batched on GPU)
            total_loss = 0.0
            for _ in range(self.config.ppo_epochs):
                self.optimizer.zero_grad()
                loss_val = self._compute_ppo_loss(all_exp)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.opponent_encoder.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss_val

            avg_loss = total_loss / self.config.ppo_epochs

            # Free experience memory and flush device cache
            del all_exp
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            elif self.device.type == 'cuda':
                torch.cuda.empty_cache()

            metrics['epoch_reward'].append(avg_reward)
            metrics['epoch_loss'].append(avg_loss)

            epoch_duration = time.time() - epoch_start
            hands_per_sec = self.config.hands_per_epoch / epoch_duration

            if self.config.verbose or (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch + 1:4d} ({epoch_duration:.1f}s, {hands_per_sec:.1f} hands/s) | "
                    f"Reward: {avg_reward:+.3f} bb | "
                    f"Loss: {avg_loss:.4f}"
                )

            if epoch_callback:
                epoch_callback(self, epoch + 1, metrics)

        return metrics

