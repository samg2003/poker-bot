"""
Curriculum Training Pipeline.

Multi-stage training that gradually increases complexity:
  Stage 1: 2 players, 20bb, 100% self-play (learn basic poker)
  Stage 2: 2-4 players, 20-100bb, 100% self-play (learn multiplayer)
  Stage 3: 2-6 players, 20-200bb, 80% GTO + 20% perturbed (detect deviations)
  Stage 4: 2-9 players, 1-350bb, 60% GTO + 40% perturbed (full exploitation)

Auto-advances when win rate plateaus (no improvement over N epochs).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from model.opponent_encoder import OpponentEncoder
from model.policy_network import PolicyNetwork
from model.action_space import ActionIndex, NUM_ACTION_TYPES
from model.stat_tracker import StatTracker, HandRecord, NUM_STAT_FEATURES
from training.personality import (
    PersonalityModifier, SituationalPersonality, TiltState,
    detect_situations, sample_table_personalities,
)
from engine.leduc_poker import LeducState, deal_leduc, CHECK, BET, FOLD, CALL, RAISE


# =============================================================================
# Curriculum stages
# =============================================================================

@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    name: str
    min_players: int
    max_players: int
    min_stack_bb: int
    max_stack_bb: int
    gto_fraction: float           # fraction of seats that are GTO
    min_epochs: int = 50          # minimum epochs before advancing
    plateau_window: int = 20      # epochs to check for plateau
    plateau_threshold: float = 0.01  # improvement threshold

    def __repr__(self):
        return (f"Stage({self.name}: {self.min_players}-{self.max_players}p, "
                f"{self.min_stack_bb}-{self.max_stack_bb}bb, "
                f"GTO={self.gto_fraction:.0%})")


DEFAULT_CURRICULUM = [
    CurriculumStage("Basic Poker", 2, 2, 20, 20, 1.0, min_epochs=100),
    CurriculumStage("Multiplayer", 2, 4, 20, 100, 1.0, min_epochs=80),
    CurriculumStage("Detect Deviations", 2, 6, 20, 200, 0.8, min_epochs=60),
    CurriculumStage("Full Exploitation", 2, 9, 1, 350, 0.6, min_epochs=100),
]


# =============================================================================
# Curriculum Trainer
# =============================================================================

@dataclass
class CurriculumConfig:
    """Global curriculum training configuration."""
    # Architecture
    embed_dim: int = 128
    opponent_embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3

    # Training
    lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.02
    value_coef: float = 0.5

    # Self-play
    hands_per_epoch: int = 512
    history_reset_range: Tuple[int, int] = (300, 500)

    # Curriculum
    stages: List[CurriculumStage] = field(default_factory=lambda: list(DEFAULT_CURRICULUM))

    # Logging
    log_interval: int = 10
    checkpoint_interval: int = 50


@dataclass
class TrainingMetrics:
    """Accumulated training metrics."""
    epoch_rewards: List[float] = field(default_factory=list)
    epoch_losses: List[float] = field(default_factory=list)
    stage_transitions: List[Tuple[int, str]] = field(default_factory=list)
    current_stage: int = 0

    def should_advance(self, stage: CurriculumStage) -> bool:
        """Check if we should advance to the next curriculum stage."""
        n = len(self.epoch_rewards)
        if n < stage.min_epochs:
            return False

        # Check for plateau: compare recent window vs previous window
        window = stage.plateau_window
        if n < window * 2:
            return False

        recent = sum(self.epoch_rewards[-window:]) / window
        previous = sum(self.epoch_rewards[-window*2:-window]) / window

        improvement = abs(recent - previous)
        return improvement < stage.plateau_threshold


class CurriculumTrainer:
    """
    Multi-stage curriculum trainer using self-play + personality perturbations.

    Currently runs on Leduc Hold'em for validation.
    When ready for NLHE, swap `_play_hand` to use the full engine.
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        seed: int = 42,
    ):
        self.config = config or CurriculumConfig()
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

        # Optimizer
        all_params = list(self.policy.parameters()) + list(self.opponent_encoder.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.config.lr)

        # Metrics
        self.metrics = TrainingMetrics()
        self.hands_since_reset = 0
        self.next_reset = self.rng.randint(*self.config.history_reset_range)

        # Stat trackers
        self.stat_trackers = [StatTracker(), StatTracker()]

    def _get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        idx = min(self.metrics.current_stage, len(self.config.stages) - 1)
        return self.config.stages[idx]

    def _play_hand_with_personality(
        self,
        personalities: List[SituationalPersonality],
    ) -> Tuple[List[dict], List[dict]]:
        """
        Play one hand with personality-modified opponents on Leduc.

        Returns (experiences_p0, experiences_p1).
        """
        self.policy.eval()

        p1_card, p2_card, board_card = deal_leduc(self.rng)
        state = LeducState(p1_card, p2_card, board_card)

        experiences = [[], []]
        cards = [p1_card, p2_card]

        while not state.is_terminal:
            actions = state.get_actions()
            if not actions:
                break

            player = state.current_player
            card = cards[player]
            personality = personalities[player]

            # Build inputs (same as self_play_trainer)
            hole_idx = card[0] * 2 + card[1]
            board_idx = (board_card[0] * 2 + board_card[1]) if state.round_idx > 0 else -1

            hole_tensor = torch.tensor([[hole_idx, 0]], dtype=torch.long)
            community = torch.tensor([[board_idx, -1, -1, -1, -1]], dtype=torch.long)

            pot = 2.0
            for rh in state.round_histories:
                bet_size = 2.0 if state.round_idx == 0 else 4.0
                for a in rh:
                    if a in (BET, RAISE, CALL):
                        pot += bet_size

            # 23-dim numeric features (match NLHE layout)
            seat_onehot = [0.0] * 9
            seat_onehot[player] = 1.0
            street_onehot = [0.0] * 4
            street_onehot[min(state.round_idx, 3)] = 1.0
            ip_flag = 1.0 if player == 1 else 0.0
            spr = 1.0 / max(pot, 0.01)

            numeric = torch.tensor([[
                pot / 10.0, 1.0, 0.0,
                *seat_onehot,
                ip_flag,
                *street_onehot,
                2.0/9, 2.0/9, 0.0, 0.0, 0.0,
                spr,
            ]], dtype=torch.float32)

            opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
            opp_stats = self.stat_trackers[player].get_stats(1 - player).unsqueeze(0).unsqueeze(0)
            own_stats = self.stat_trackers[player].get_stats(player).unsqueeze(0)

            action_mask_list = [False] * NUM_ACTION_TYPES
            for a in actions:
                if a == FOLD: action_mask_list[ActionIndex.FOLD] = True
                elif a == CHECK: action_mask_list[ActionIndex.CHECK] = True
                elif a == CALL: action_mask_list[ActionIndex.CALL] = True
                elif a in (BET, RAISE): action_mask_list[ActionIndex.RAISE] = True

            action_mask = torch.tensor([action_mask_list])

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

            # Apply personality perturbation
            hand_strength = card[0] / 2.0  # crude: J=0, Q=0.5, K=1.0
            situations = detect_situations(
                street=state.round_idx,
                is_facing_bet=any(a in (BET, RAISE) for rh in state.round_histories for a in rh),
            )
            probs = personality.apply(
                probs, situations,
                hand_strength=hand_strength,
                is_facing_raise=RAISE in (state.round_histories[state.round_idx] if state.round_histories[state.round_idx] else []),
            )

            # Sample action
            dist = Categorical(probs)
            action_idx = dist.sample().item()

            exp = {
                'hole_card_idx': hole_idx,
                'board_card_idx': board_idx,
                'numeric_features': numeric[0].tolist(),
                'opponent_embedding': opp_embed.squeeze(0).squeeze(0),
                'opponent_stats': opp_stats.squeeze(0).squeeze(0),
                'own_stats': own_stats.squeeze(0),
                'action_mask': action_mask_list,
                'action_idx': action_idx,
                'action_log_prob': dist.log_prob(torch.tensor(action_idx)).item(),
                'value_estimate': output.value[0, 0].item(),
                'reward': 0.0,
            }
            experiences[player].append(exp)

            # Map back to Leduc action
            if action_idx == ActionIndex.RAISE:
                leduc_action = BET if BET in actions else RAISE
            elif action_idx == ActionIndex.FOLD:
                leduc_action = FOLD if FOLD in actions else actions[0]
            elif action_idx == ActionIndex.CALL:
                leduc_action = CALL if CALL in actions else actions[0]
            else:
                leduc_action = CHECK if CHECK in actions else actions[0]

            state = state.apply(leduc_action)

        # Assign rewards
        if state.is_terminal:
            for p in range(2):
                reward = state.get_payoff(p)
                for exp in experiences[p]:
                    exp['reward'] = float(reward)

        return experiences[0], experiences[1]

    def _compute_loss(self, all_experiences: List[dict]) -> torch.Tensor:
        """Compute PPO loss from collected experiences."""
        if not all_experiences:
            return torch.tensor(0.0)

        self.policy.train()

        hole_cards = torch.tensor([[e['hole_card_idx'], 0] for e in all_experiences], dtype=torch.long)
        community = torch.tensor([[e['board_card_idx'], -1, -1, -1, -1] for e in all_experiences], dtype=torch.long)
        numeric = torch.tensor([e['numeric_features'] for e in all_experiences], dtype=torch.float32)
        opp_embeds = torch.stack([e['opponent_embedding'] for e in all_experiences]).unsqueeze(1)
        opp_stats = torch.stack([e['opponent_stats'] for e in all_experiences]).unsqueeze(1)
        own_stats = torch.stack([e['own_stats'] for e in all_experiences])
        action_masks = torch.tensor([e['action_mask'] for e in all_experiences])
        old_actions = torch.tensor([e['action_idx'] for e in all_experiences], dtype=torch.long)
        old_log_probs = torch.tensor([e['action_log_prob'] for e in all_experiences], dtype=torch.float32)
        rewards = torch.tensor([e['reward'] for e in all_experiences], dtype=torch.float32)
        old_values = torch.tensor([e['value_estimate'] for e in all_experiences], dtype=torch.float32)

        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embeds,
            opponent_stats=opp_stats,
            own_stats=own_stats,
            action_mask=action_masks,
        )

        dist = Categorical(output.action_type_probs)
        new_log_probs = dist.log_prob(old_actions)
        entropy = dist.entropy().mean()

        advantages = rewards - old_values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = F.mse_loss(output.value.squeeze(-1), rewards)

        return policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

    def train_epoch(self) -> float:
        """Run one epoch of curriculum training. Returns avg reward."""
        stage = self._get_current_stage()

        # Sample personalities for the table
        personalities = sample_table_personalities(
            num_seats=2,  # Leduc is 2-player
            gto_fraction=stage.gto_fraction,
            rng=self.rng,
        )

        all_exp = []
        total_reward = 0.0

        for _ in range(self.config.hands_per_epoch):
            exp0, exp1 = self._play_hand_with_personality(personalities)
            all_exp.extend(exp0)
            all_exp.extend(exp1)
            if exp0:
                total_reward += exp0[0]['reward']

            # History reset
            self.hands_since_reset += 1
            if self.hands_since_reset >= self.next_reset:
                self.stat_trackers = [StatTracker(), StatTracker()]
                self.hands_since_reset = 0
                self.next_reset = self.rng.randint(*self.config.history_reset_range)
                # Re-sample personalities
                personalities = sample_table_personalities(
                    num_seats=2, gto_fraction=stage.gto_fraction, rng=self.rng,
                )

        avg_reward = total_reward / self.config.hands_per_epoch

        # PPO update
        total_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            self.optimizer.zero_grad()
            loss = self._compute_loss(all_exp)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / self.config.ppo_epochs
        self.metrics.epoch_rewards.append(avg_reward)
        self.metrics.epoch_losses.append(avg_loss)

        return avg_reward

    def train(self, max_epochs: int = 500) -> TrainingMetrics:
        """
        Run curriculum training for up to max_epochs.

        Auto-advances stages when plateau is detected.
        """
        for epoch in range(max_epochs):
            stage = self._get_current_stage()
            avg_reward = self.train_epoch()

            # Log
            if (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"Epoch {epoch+1:4d} | "
                    f"Stage: {stage.name} | "
                    f"Reward: {avg_reward:+.3f} | "
                    f"Loss: {self.metrics.epoch_losses[-1]:.4f}"
                )

            # Check for stage advancement
            if self.metrics.should_advance(stage):
                if self.metrics.current_stage < len(self.config.stages) - 1:
                    self.metrics.current_stage += 1
                    new_stage = self._get_current_stage()
                    self.metrics.stage_transitions.append((epoch, new_stage.name))
                    print(f"\n{'='*50}")
                    print(f"ADVANCING to Stage: {new_stage.name}")
                    print(f"{'='*50}\n")

        return self.metrics
