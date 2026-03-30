"""
Poker Agent — the unified inference interface.

Coordinates all components into a single `get_action()` call:
  1. Update stat tracker with latest observed actions
  2. Encode opponent histories → opponent encoder → embeddings
  3. Encode game state → card embeddings + numeric features
  4. Forward → policy network → action probs + value (System 1, ~1ms)
  5. If complex spot → run search (System 2, ~200ms)
  6. Sample action from final distribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.distributions import Categorical

from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from model.stat_tracker import StatTracker, HandRecord, NUM_STAT_FEATURES
from model.action_space import ActionIndex, NUM_ACTION_TYPES, encode_action, ACTION_FEATURE_DIM
from agent.config import AgentConfig


@dataclass
class ActionResult:
    """Result of get_action()."""
    action_type: int              # ActionIndex enum value
    bet_sizing: float             # pot fraction from POT_FRACTIONS (-1 = all-in)
    action_probs: torch.Tensor    # (4,) action type probabilities
    value_estimate: float         # expected value in bb
    used_search: bool             # whether System 2 was triggered


class PokerAgent:
    """
    Full poker agent — System 1 + System 2.

    Usage:
        agent = PokerAgent.from_config(config)
        result = agent.get_action(game_state, seat_id=0)
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        config: Optional[AgentConfig] = None,
    ):
        self.policy = policy
        self.opponent_encoder = opponent_encoder
        self.config = config or AgentConfig()

        # Per-opponent tracking
        self.stat_tracker = StatTracker()
        self.opponent_histories: Dict[int, List[torch.Tensor]] = {}

        # Action history for opponent encoder
        self._action_sequences: Dict[int, List[torch.Tensor]] = {}

    @classmethod
    def from_config(cls, config: Optional[AgentConfig] = None) -> 'PokerAgent':
        """Create agent from configuration."""
        config = config or AgentConfig()

        opponent_encoder = OpponentEncoder(
            embed_dim=config.opponent_embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )
        policy = PolicyNetwork(
            embed_dim=config.embed_dim,
            opponent_embed_dim=config.opponent_embed_dim,
            num_cross_attn_heads=config.num_heads,
            num_cross_attn_layers=config.num_layers,
        )

        return cls(policy, opponent_encoder, config)

    def observe_action(
        self,
        player_id: int,
        action_type: int,
        bet_size_frac: float = 0.0,
        pot_size: float = 0.0,
        stack_size: float = 0.0,
        street: int = 0,
    ) -> None:
        """
        Record an observed action for opponent modeling.

        Call this after every action at the table.
        """
        token = encode_action(action_type, bet_size_frac, pot_size, stack_size, street)
        if player_id not in self._action_sequences:
            self._action_sequences[player_id] = []
        self._action_sequences[player_id].append(token)

    def record_hand_result(self, player_id: int, record: HandRecord) -> None:
        """Record a completed hand for stat tracking."""
        self.stat_tracker.record_hand(player_id, record)

    def _get_opponent_embeddings(
        self,
        opponent_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get embeddings and stats for opponents.

        Returns:
            embeddings: (1, num_opp, embed_dim)
            stats: (1, num_opp, NUM_STAT_FEATURES)
        """
        embeddings = []
        stats = []

        for opp_id in opponent_ids:
            # Get action sequence for this opponent
            seq = self._action_sequences.get(opp_id, [])
            if seq:
                x = torch.stack(seq).unsqueeze(0)  # (1, seq_len, feat_dim)
                emb = self.opponent_encoder(x)      # (1, embed_dim)
                embeddings.append(emb.squeeze(0))
            else:
                emb = self.opponent_encoder.encode_empty(1)  # (1, embed_dim)
                embeddings.append(emb.squeeze(0))

            stats.append(self.stat_tracker.get_stats(opp_id))

        emb_tensor = torch.stack(embeddings).unsqueeze(0)   # (1, num_opp, embed_dim)
        stats_tensor = torch.stack(stats).unsqueeze(0)       # (1, num_opp, features)

        return emb_tensor, stats_tensor

    @torch.no_grad()
    def get_action(
        self,
        hole_cards: Tuple[int, int],
        community_cards: List[int],
        numeric_features: List[float],
        opponent_ids: List[int],
        action_mask: Optional[List[bool]] = None,
        pot_bb: float = 0.0,
        street: int = 0,
    ) -> ActionResult:
        """
        Get an action for the current game state.

        This is the main entry point — System 1 + optional System 2.
        """
        self.policy.eval()

        # Encode cards
        hole = torch.tensor([list(hole_cards)], dtype=torch.long)
        board = list(community_cards)
        while len(board) < 5:
            board.append(-1)
        community = torch.tensor([board], dtype=torch.long)

        # Numeric features
        numeric = torch.tensor([numeric_features], dtype=torch.float32)

        # Opponent info
        if opponent_ids:
            opp_embed, opp_stats = self._get_opponent_embeddings(opponent_ids)
        else:
            opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
            opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)

        own_stats = self.stat_tracker.get_stats(-1).unsqueeze(0)  # own stats

        # Action mask
        if action_mask is not None:
            mask = torch.tensor([action_mask])
        else:
            mask = None

        # System 1: policy forward pass
        output = self.policy(
            hole_cards=hole,
            community_cards=community,
            numeric_features=numeric,
            opponent_embeddings=opp_embed,
            opponent_stats=opp_stats,
            own_stats=own_stats,
            action_mask=mask,
        )

        action_probs = output.action_type_probs[0]
        value = output.value[0, 0].item()
        sizing_probs = torch.softmax(output.bet_size_logits[0], dim=-1).tolist()

        # Sample action
        dist = Categorical(action_probs)
        action_idx = dist.sample().item()

        sizing_idx = 0
        if action_idx == ActionIndex.RAISE and sum(sizing_probs) > 0:
            s_dist = Categorical(torch.tensor(sizing_probs))
            sizing_idx = s_dist.sample().item()
            
        from model.action_space import POT_FRACTIONS
        fraction = float(POT_FRACTIONS[sizing_idx])

        return ActionResult(
            action_type=action_idx,
            bet_sizing=fraction,
            action_probs=action_probs,
            value_estimate=value,
            used_search=used_search,
        )

    def reset_opponent(self, player_id: int) -> None:
        """Reset all tracking for an opponent (e.g., new session)."""
        self.stat_tracker.reset(player_id)
        self._action_sequences.pop(player_id, None)

    def reset_all(self) -> None:
        """Reset all tracking (e.g., new table)."""
        self.stat_tracker = StatTracker()
        self._action_sequences.clear()

    def get_param_count(self) -> int:
        """Total parameter count across all models."""
        total = self.policy.get_param_count()
        total += sum(p.numel() for p in self.opponent_encoder.parameters())
        return total
