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


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

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
        """Sample random player count + per-player stacks."""
        c = self.config
        num_p = c.num_players if c.num_players > 0 else self.rng.randint(c.min_players, c.max_players)

        if c.starting_bb > 0:
            stacks = [c.starting_bb * c.big_blind] * num_p
        elif c.uniform_stacks:
            bb = self.rng.randint(c.min_bb, c.max_bb)
            stacks = [bb * c.big_blind] * num_p
        else:
            stacks = [self.rng.randint(c.min_bb, c.max_bb) * c.big_blind for _ in range(num_p)]

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

        # Cap at 512 actions (rolling window)
        if len(self.action_histories[player_id]) > 512:
            self.action_histories[player_id] = self.action_histories[player_id][-512:]

    def _get_opponent_embedding(self, player_id: int) -> torch.Tensor:
        """
        Get opponent embedding from action history.
        Returns: (1, embed_dim) tensor.
        """
        history = self.action_histories.get(player_id, [])

        if not history:
            return self.opponent_encoder.encode_empty(1, device=str(self.device))

        # Stack to (1, seq_len, 7)
        seq = torch.stack(history).unsqueeze(0)
        seq = self._to(seq)
        embedding = self.opponent_encoder(seq)  # (1, embed_dim)
        return embedding

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

    def _maybe_reset_histories(self):
        """Periodically reset opponent histories (simulate new table)."""
        self.hands_since_reset += 1
        if self.hands_since_reset >= self.next_reset_at:
            self.action_histories.clear()
            self.stat_tracker.reset()
            self.hands_since_reset = 0
            self.next_reset_at = self.rng.randint(*self.config.history_reset_interval)

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

    def _play_hand(self, use_search: bool = False) -> Tuple[List[Experience], ...]:
        """
        Play one full hand of NLHE using self-play.

        Args:
            use_search: if True, use System 2 search for action selection

        Returns experience lists for each player.
        """
        self.policy.eval()
        self.opponent_encoder.eval()
        num_p, stacks = self._sample_table()

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

        # Build hand records for stat tracking
        hand_records = [HandRecord() for _ in range(num_p)]

        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            if not p.is_active:
                break

            # Encode state
            encoded = self._encode_state(game_state, pid)

            # Forward pass
            with torch.no_grad():
                # Get REAL opponent embeddings from history
                opp_embed = self._get_all_opponent_embeddings(pid, num_p)
                opp_stats = self._get_opponent_stats(pid, num_p)
                own_stats = self._to(self.stat_tracker.get_stats(pid).unsqueeze(0))

                output = self.policy(
                    hole_cards=encoded['hole_cards'],
                    community_cards=encoded['community_cards'],
                    numeric_features=encoded['numeric_features'],
                    opponent_embeddings=opp_embed,
                    opponent_stats=opp_stats,
                    own_stats=own_stats,
                    action_mask=encoded['action_mask'],
                )

            probs = output.action_type_probs[0]
            value = output.value[0, 0].item()
            sizing = output.bet_sizing[0, 0].item()

            # Search-guided action selection (expert iteration)
            if use_search and self.search_engine is not None:
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
                        probs = 0.7 * refined + 0.3 * probs

            # Sample action
            dist = Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()

            # Store experience
            experiences[pid].append({
                'hole_cards': encoded['hole_cards'],
                'community_cards': encoded['community_cards'],
                'numeric_features': encoded['numeric_features'],
                'opponent_embeddings': opp_embed,
                'opponent_stats': opp_stats,
                'own_stats': own_stats,
                'action_mask': encoded['action_mask'],
                'action_idx': action_idx,
                'log_prob': log_prob,
                'value': value,
            })

            # Decode and apply action
            action = self._decode_action(action_idx, sizing, game_state)
            dealer.apply_action(action)

            # Record action in opponent history for future hands
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

            # Track for HUD-style stats
            if action.action_type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].vpip = True
            if action.action_type in (ActionType.RAISE, ActionType.ALL_IN):
                hand_records[pid].pfr = True

        # Record hand results for stat tracker
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

        # Convert to Experience with rewards
        all_experiences = []
        for pid in range(num_p):
            player_exp = []
            reward = profits[pid] / self.config.big_blind
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

        return tuple(all_experiences)

    # ─────────────────────────────────────────────────────────
    # PPO loss
    # ─────────────────────────────────────────────────────────

    def _compute_ppo_loss(self, experiences: List[Experience]) -> float:
        """Compute PPO loss from collected experience."""
        if not experiences:
            return 0.0

        self.policy.train()
        self.opponent_encoder.train()
        total_loss = 0.0

        for exp in experiences:
            output = self.policy(
                hole_cards=exp.hole_cards,
                community_cards=exp.community_cards,
                numeric_features=exp.numeric_features,
                opponent_embeddings=exp.opponent_embeddings,
                opponent_stats=exp.opponent_stats,
                own_stats=exp.own_stats,
                action_mask=exp.action_mask,
            )

            probs = output.action_type_probs[0]
            new_dist = Categorical(probs)
            action_t = torch.tensor(exp.action_idx, device=self.device)
            new_log_prob = new_dist.log_prob(action_t)

            ratio = (new_log_prob - exp.log_prob).exp()
            advantage = exp.reward - exp.value

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantage
            policy_loss = -torch.min(surr1, surr2)

            value_pred = output.value[0, 0]
            value_loss = (value_pred - exp.reward) ** 2

            entropy = new_dist.entropy()

            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            ) / len(experiences)
            
            loss.backward()
            total_loss += loss.item()

        return total_loss

    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────

    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Run NLHE self-play training."""
        metrics = {
            'epoch_reward': [],
            'epoch_loss': [],
        }

        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            all_exp: List[Experience] = []
            epoch_reward = 0.0
            epoch_start = time.time()

            for hand_i in range(self.config.hands_per_epoch):
                # Decide whether to use search for this hand
                use_search = (
                    self.search_engine is not None and
                    self.rng.random() < self.config.search_fraction
                )

                player_experiences = self._play_hand(use_search=use_search)
                for pexp in player_experiences:
                    all_exp.extend(pexp)
                if player_experiences[0]:
                    epoch_reward += player_experiences[0][0].reward

                # Verbose progress tracking
                if self.config.verbose and (hand_i + 1) % max(1, self.config.hands_per_epoch // 5) == 0:
                    elapsed = time.time() - epoch_start
                    print(f"  [Epoch {epoch+1:4d}] Hand {hand_i+1}/{self.config.hands_per_epoch}... ({elapsed:.1f}s)")

            avg_reward = epoch_reward / self.config.hands_per_epoch

            # PPO update
            total_loss = 0.0
            for _ in range(self.config.ppo_epochs):
                self.optimizer.zero_grad()
                loss_val = self._compute_ppo_loss(all_exp)
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.opponent_encoder.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss_val

            avg_loss = total_loss / self.config.ppo_epochs

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

        return metrics
