"""
NLHE Self-Play Trainer — full No-Limit Hold'em training loop.

Uses the full game engine (engine/dealer.py) + NLHEEncoder to bridge
GameState → model tensors. Same PPO training as the Leduc trainer,
but on full 52-card NLHE with 2-9 players, side pots, etc.

Each hand randomly samples:
- Number of players (min_players to max_players)
- Per-player stack depth (min_bb to max_bb, independently)

This trains one universal model that handles any table configuration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from engine.game_state import GameState, Action, ActionType, Street
from engine.dealer import Dealer
from model.action_space import ActionIndex, ActionOutput, NUM_ACTION_TYPES, encode_action
from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork
from model.stat_tracker import StatTracker, NUM_STAT_FEATURES
from model.nlhe_encoder import NLHEEncoder


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


@dataclass
class NLHETrainingConfig:
    """Configuration for NLHE training."""
    # Architecture
    embed_dim: int = 128
    opponent_embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3

    # Game setup — ranges for per-hand randomization
    min_players: int = 2         # minimum players at the table
    max_players: int = 6         # maximum players at the table
    min_bb: int = 20             # minimum stack depth (in BB)
    max_bb: int = 200            # maximum stack depth (in BB)
    small_blind: float = 0.5
    big_blind: float = 1.0
    uniform_stacks: bool = False # if False, each player gets independent random stack

    # For backwards compatibility — if set, overrides ranges
    num_players: int = 0         # 0 = use min/max range
    starting_bb: int = 0         # 0 = use min/max range

    # Training
    lr: float = 3e-4
    gamma: float = 1.0
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.02
    value_coef: float = 0.5

    # Self-play
    hands_per_epoch: int = 256
    history_reset_interval: Tuple[int, int] = (300, 500)

    # Personality
    gto_fraction: float = 0.8

    # Logging
    log_interval: int = 10


class NLHESelfPlayTrainer:
    """
    Self-play trainer on full NLHE.

    Each hand randomly samples:
    - Number of players (min_players to max_players)
    - Per-player stack depth (min_bb to max_bb, independently)

    This trains one universal model that handles any table configuration.
    """

    def __init__(self, config: Optional[NLHETrainingConfig] = None, seed: int = 42):
        self.config = config or NLHETrainingConfig()
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Models
        self.opponent_encoder = OpponentEncoder(
            embed_dim=self.config.opponent_embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
        )
        self.policy = PolicyNetwork(
            embed_dim=self.config.embed_dim,
            opponent_embed_dim=self.config.opponent_embed_dim,
            num_cross_attn_heads=self.config.num_heads,
            num_cross_attn_layers=self.config.num_layers,
        )

        all_params = list(self.policy.parameters()) + list(self.opponent_encoder.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.lr)

        self.encoder = NLHEEncoder()

        # Metrics
        self.epoch_rewards: List[float] = []
        self.epoch_losses: List[float] = []

    def _sample_table(self) -> Tuple[int, List[float]]:
        """
        Sample a random table configuration for one hand.

        Returns:
            num_players: number of players (2-9)
            stacks: per-player chip stacks
        """
        c = self.config

        # Player count
        if c.num_players > 0:
            num_p = c.num_players
        else:
            num_p = self.rng.randint(c.min_players, c.max_players)

        # Stack depths (in chips)
        if c.starting_bb > 0:
            stacks = [c.starting_bb * c.big_blind] * num_p
        elif c.uniform_stacks:
            bb_depth = self.rng.randint(c.min_bb, c.max_bb)
            stacks = [bb_depth * c.big_blind] * num_p
        else:
            # Each player gets an independent random stack
            stacks = [
                self.rng.randint(c.min_bb, c.max_bb) * c.big_blind
                for _ in range(num_p)
            ]

        return num_p, stacks

    def _encode_action_mask(self, game_state: GameState) -> torch.Tensor:
        """Encode legal actions for current player as (1, 4) bool tensor."""
        legal_types = game_state.get_legal_actions()
        mask = torch.zeros(1, NUM_ACTION_TYPES, dtype=torch.bool)

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
        """Encode game state to model tensors for a player."""
        p = game_state.players[player_idx]

        # Card encoding
        hole = torch.tensor([list(p.hole_cards)], dtype=torch.long)
        board = list(game_state.board)
        while len(board) < 5:
            board.append(-1)
        community = torch.tensor([board[:5]], dtype=torch.long)

        # Numeric features
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

        numeric = torch.tensor([[
            pot, own_stack, own_bet, position, street_val,
            game_state.num_players / 9.0, num_active / 9.0,
            current_bet, min_raise,
        ]], dtype=torch.float32)

        action_mask = self._encode_action_mask(game_state)

        return {
            'hole_cards': hole,
            'community_cards': community,
            'numeric_features': numeric,
            'action_mask': action_mask,
        }

    def _decode_action(
        self,
        action_idx: int,
        bet_sizing: float,
        game_state: GameState,
    ) -> Action:
        """Convert model output to engine Action."""
        legal = game_state.get_legal_actions()

        if action_idx == ActionIndex.FOLD and ActionType.FOLD in legal:
            return Action(ActionType.FOLD)
        elif action_idx == ActionIndex.CHECK and ActionType.CHECK in legal:
            return Action(ActionType.CHECK)
        elif action_idx == ActionIndex.CALL and ActionType.CALL in legal:
            return Action(ActionType.CALL)
        elif action_idx == ActionIndex.RAISE and ActionType.RAISE in legal:
            min_raise = game_state.get_min_raise_to()
            max_raise = game_state.get_max_raise_to()
            raise_to = min_raise + bet_sizing * (max_raise - min_raise)
            raise_to = max(min_raise, min(raise_to, max_raise))
            return Action(ActionType.RAISE, amount=raise_to)

        # Fallback: check > call > fold
        if ActionType.CHECK in legal:
            return Action(ActionType.CHECK)
        if ActionType.CALL in legal:
            return Action(ActionType.CALL)
        return Action(ActionType.FOLD)

    def _play_hand(self) -> Tuple[List[Experience], ...]:
        """
        Play one full hand of NLHE using self-play.

        Randomly samples table config (player count, stack depths).
        Returns experience lists for each player.
        """
        self.policy.eval()
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

        # Play out the hand
        while not dealer.is_hand_over():
            pid = game_state.current_player_idx
            p = game_state.players[pid]

            if not p.is_active:
                break

            # Encode state
            encoded = self._encode_state(game_state, pid)
            opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
            opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)
            own_stats = torch.zeros(1, NUM_STAT_FEATURES)

            # Forward pass
            with torch.no_grad():
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

            # Sample action
            dist = Categorical(probs)
            action_idx = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action_idx)).item()

            # Store experience (reward filled in later)
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

        # Calculate rewards (profit in bb)
        results = dealer.get_results()
        profits = results['profit']

        # Convert to Experience tuples with rewards
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

    def _compute_ppo_loss(self, experiences: List[Experience]) -> torch.Tensor:
        """Compute PPO loss from collected experience."""
        if not experiences:
            return torch.tensor(0.0, requires_grad=True)

        self.policy.train()
        total_loss = torch.tensor(0.0)

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
            new_log_prob = new_dist.log_prob(torch.tensor(exp.action_idx))

            # PPO ratio
            ratio = (new_log_prob - exp.log_prob).exp()
            advantage = exp.reward - exp.value

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantage
            policy_loss = -torch.min(surr1, surr2)

            # Value loss
            value_pred = output.value[0, 0]
            value_loss = (value_pred - exp.reward) ** 2

            # Entropy bonus
            entropy = new_dist.entropy()

            loss = (
                policy_loss
                + self.config.value_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            total_loss = total_loss + loss

        return total_loss / len(experiences)

    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """Run NLHE self-play training."""
        metrics = {
            'epoch_reward': [],
            'epoch_loss': [],
        }

        for epoch in range(num_epochs):
            # Collect experience
            all_exp: List[Experience] = []
            epoch_reward = 0.0

            for _ in range(self.config.hands_per_epoch):
                player_experiences = self._play_hand()
                for pexp in player_experiences:
                    all_exp.extend(pexp)
                # Track P0 reward
                if player_experiences[0]:
                    epoch_reward += player_experiences[0][0].reward

            avg_reward = epoch_reward / self.config.hands_per_epoch

            # PPO update
            total_loss = 0.0
            for _ in range(self.config.ppo_epochs):
                self.optimizer.zero_grad()
                loss = self._compute_ppo_loss(all_exp)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / self.config.ppo_epochs

            metrics['epoch_reward'].append(avg_reward)
            metrics['epoch_loss'].append(avg_loss)

            if (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"Reward: {avg_reward:+.3f} bb | "
                    f"Loss: {avg_loss:.4f}"
                )

        return metrics
