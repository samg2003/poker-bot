"""
Self-Play Trainer for poker AI.

Plays the policy network against copies of itself, collects experience,
and trains via PPO (Proximal Policy Optimization).

The training loop:
1. Play N hands of self-play → collect trajectories
2. Compute advantages and returns
3. Update policy via PPO
4. Periodically snapshot the opponent pool

History resets every 300-500 hands → forces GTO emergence when no reads.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from engine.leduc_poker import LeducState, deal_leduc, CHECK, BET, FOLD, CALL, RAISE
from model.action_space import ActionIndex, NUM_ACTION_TYPES
from model.stat_tracker import StatTracker, HandRecord, NUM_STAT_FEATURES
from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork


@dataclass
class Experience:
    """A single decision point in a hand."""
    # State
    hole_card_idx: int        # card index for embedding
    board_card_idx: int       # -1 if preflop
    numeric_features: List[float]
    opponent_embedding: torch.Tensor
    opponent_stats: torch.Tensor
    own_stats: torch.Tensor
    action_mask: List[bool]

    # Action taken
    action_idx: int
    action_log_prob: float

    # Outcome (filled after hand)
    reward: float = 0.0
    value_estimate: float = 0.0


@dataclass
class TrainingConfig:
    """Hyperparameters for self-play training."""
    # Architecture
    embed_dim: int = 64
    opponent_embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2

    # Training
    lr: float = 3e-4
    gamma: float = 1.0           # no discounting (episodic)
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.02   # encourage exploration
    value_coef: float = 0.5

    # Self-play
    hands_per_epoch: int = 256
    history_reset_interval: Tuple[int, int] = (300, 500)

    # Logging
    log_interval: int = 10


class LeducSelfPlayTrainer:
    """
    Self-play trainer on Leduc Hold'em.

    Two copies of the policy play against each other.
    Both are updated from the same collected experience (symmetric game).
    """

    def __init__(self, config: Optional[TrainingConfig] = None, seed: int = 42):
        self.config = config or TrainingConfig()
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Create models
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

        # Single optimizer for all parameters
        all_params = list(self.policy.parameters()) + list(self.opponent_encoder.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.lr)

        # Stat trackers (one per player perspective)
        self.stat_trackers = [StatTracker(), StatTracker()]

        # Metrics
        self.epoch_rewards: List[float] = []
        self.epoch_losses: List[float] = []

    def _card_to_idx(self, card: Tuple[int, int]) -> int:
        """Convert Leduc card (rank, suit) to index 0-5."""
        return card[0] * 2 + card[1]

    def _get_action_mask(self, actions: List[str]) -> List[bool]:
        """Convert Leduc actions to our 4-way action mask."""
        mask = [False] * NUM_ACTION_TYPES
        for a in actions:
            if a == FOLD:
                mask[ActionIndex.FOLD] = True
            elif a == CHECK:
                mask[ActionIndex.CHECK] = True
            elif a == CALL:
                mask[ActionIndex.CALL] = True
            elif a in (BET, RAISE):
                mask[ActionIndex.RAISE] = True
        return mask

    def _action_idx_to_leduc(self, action_idx: int, legal_actions: List[str]) -> str:
        """Map our action index back to a Leduc action."""
        mapping = {
            ActionIndex.FOLD: FOLD,
            ActionIndex.CHECK: CHECK,
            ActionIndex.CALL: CALL,
            ActionIndex.RAISE: None,  # could be BET or RAISE
        }
        if action_idx == ActionIndex.RAISE:
            # BET if no prior bet, RAISE if facing a bet
            if BET in legal_actions:
                return BET
            return RAISE
        action = mapping.get(action_idx, CHECK)
        if action in legal_actions:
            return action
        # Fallback: pick first legal action
        return legal_actions[0]

    def _get_numeric_features(self, state: LeducState, player: int) -> List[float]:
        """Extract 9 numeric features from Leduc state."""
        pot = 2.0  # antes
        for rh in state.round_histories:
            bet_size = 2.0 if state.round_idx == 0 else 4.0
            for a in rh:
                if a in (BET, RAISE, CALL):
                    pot += bet_size

        return [
            pot / 10.0,                   # pot size (normalized)
            1.0,                          # stack (always enough in Leduc)
            0.0,                          # own bet this street
            float(player),                # position (0 or 1)
            float(state.round_idx) / 1.0, # street (0 or 1)
            2.0 / 9.0,                   # num_players / max
            2.0 / 9.0,                   # num_active / max
            0.0,                          # current bet (normalized)
            0.0,                          # min raise (normalized)
            0.0,                          # amount to call
        ]

    @torch.no_grad()
    def _play_hand(self) -> Tuple[List[Experience], List[Experience]]:
        """
        Play one hand of Leduc Hold'em between two policy copies.
        Returns experiences for [player_0, player_1].
        """
        self.policy.eval()

        p1_card, p2_card, board_card = deal_leduc(self.rng)
        state = LeducState(p1_card, p2_card, board_card)

        experiences = [[], []]

        while not state.is_terminal:
            actions = state.get_actions()
            if not actions:
                break

            player = state.current_player
            card = p1_card if player == 0 else p2_card

            # Build inputs
            hole_idx = self._card_to_idx(card)
            board_idx = self._card_to_idx(board_card) if state.round_idx > 0 else -1

            hole_cards = torch.tensor([[hole_idx, 0]], dtype=torch.long)
            community = torch.tensor([[board_idx, -1, -1, -1, -1]], dtype=torch.long)
            numeric = torch.tensor([self._get_numeric_features(state, player)], dtype=torch.float32)
            opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
            opp_stats = self.stat_trackers[player].get_stats(1 - player).unsqueeze(0).unsqueeze(0)
            own_stats = self.stat_trackers[player].get_stats(player).unsqueeze(0)
            action_mask_list = self._get_action_mask(actions)
            action_mask = torch.tensor([action_mask_list])

            # Forward pass
            output = self.policy(
                hole_cards=hole_cards,
                community_cards=community,
                numeric_features=numeric,
                opponent_embeddings=opp_embed,
                opponent_stats=opp_stats,
                own_stats=own_stats,
                action_mask=action_mask,
            )

            # Sample action
            probs = output.action_type_probs[0]
            dist = Categorical(probs)
            action_idx = dist.sample().item()

            # Store experience
            exp = Experience(
                hole_card_idx=hole_idx,
                board_card_idx=board_idx,
                numeric_features=self._get_numeric_features(state, player),
                opponent_embedding=opp_embed.squeeze(0).squeeze(0),
                opponent_stats=opp_stats.squeeze(0).squeeze(0),
                own_stats=own_stats.squeeze(0),
                action_mask=action_mask_list,
                action_idx=action_idx,
                action_log_prob=dist.log_prob(torch.tensor(action_idx)).item(),
                value_estimate=output.value[0, 0].item(),
            )
            experiences[player].append(exp)

            # Apply action
            leduc_action = self._action_idx_to_leduc(action_idx, actions)
            state = state.apply(leduc_action)

        # Assign rewards
        if state.is_terminal:
            for player in range(2):
                reward = state.get_payoff(player)
                for exp in experiences[player]:
                    exp.reward = reward

        return experiences[0], experiences[1]

    def _compute_ppo_loss(self, experiences: List[Experience]) -> torch.Tensor:
        """Compute PPO loss from collected experiences."""
        if not experiences:
            return torch.tensor(0.0)

        self.policy.train()

        # Batch all experiences
        hole_cards = torch.tensor([[e.hole_card_idx, 0] for e in experiences], dtype=torch.long)
        community = torch.tensor(
            [[e.board_card_idx, -1, -1, -1, -1] for e in experiences], dtype=torch.long
        )
        numeric = torch.tensor([e.numeric_features for e in experiences], dtype=torch.float32)
        opp_embeds = torch.stack([e.opponent_embedding for e in experiences]).unsqueeze(1)
        opp_stats = torch.stack([e.opponent_stats for e in experiences]).unsqueeze(1)
        own_stats = torch.stack([e.own_stats for e in experiences])
        action_masks = torch.tensor([e.action_mask for e in experiences])
        old_actions = torch.tensor([e.action_idx for e in experiences], dtype=torch.long)
        old_log_probs = torch.tensor([e.action_log_prob for e in experiences], dtype=torch.float32)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        old_values = torch.tensor([e.value_estimate for e in experiences], dtype=torch.float32)

        # Forward pass
        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embeds,
            opponent_stats=opp_stats,
            own_stats=own_stats,
            action_mask=action_masks,
        )

        # New log probs
        dist = Categorical(output.action_type_probs)
        new_log_probs = dist.log_prob(old_actions)
        entropy = dist.entropy().mean()

        # Advantages (simple: reward - value baseline)
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss
        value_loss = F.mse_loss(output.value.squeeze(-1), rewards)

        # Total loss
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        return loss

    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Run self-play training.

        Returns dict with training metrics.
        """
        metrics = {
            'epoch_reward': [],
            'epoch_loss': [],
            'p1_avg_reward': [],
        }

        for epoch in range(num_epochs):
            # Collect experience via self-play
            all_exp_p1: List[Experience] = []
            all_exp_p2: List[Experience] = []

            epoch_reward = 0.0
            for _ in range(self.config.hands_per_epoch):
                exp1, exp2 = self._play_hand()
                all_exp_p1.extend(exp1)
                all_exp_p2.extend(exp2)
                if exp1:
                    epoch_reward += exp1[0].reward

            avg_reward = epoch_reward / self.config.hands_per_epoch

            # Combine experiences (symmetric game)
            all_exp = all_exp_p1 + all_exp_p2

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

            # Record metrics
            metrics['epoch_reward'].append(avg_reward)
            metrics['epoch_loss'].append(avg_loss)
            metrics['p1_avg_reward'].append(avg_reward)

            if (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"P1 reward: {avg_reward:+.3f} | "
                    f"Loss: {avg_loss:.4f}"
                )

        return metrics


if __name__ == '__main__':
    print("Starting Leduc Hold'em self-play training...")
    trainer = LeducSelfPlayTrainer(seed=42)
    metrics = trainer.train(num_epochs=50)
    print(f"\nFinal P1 avg reward: {metrics['p1_avg_reward'][-1]:+.3f}")
