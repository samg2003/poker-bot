"""
Agent configuration — all tunable parameters in one place.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the poker agent."""

    # Model architecture
    embed_dim: int = 64
    opponent_embed_dim: int = 64
    num_heads: int = 2
    num_layers: int = 2

    # Training
    lr: float = 3e-4
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    gamma: float = 1.0

    # Self-play
    hands_per_epoch: int = 512
    history_reset_min: int = 300
    history_reset_max: int = 500

    # Evaluation
    eval_num_hands: int = 10000
    eval_opponent_types: int = 5

    def __repr__(self):
        return (
            f"AgentConfig(embed={self.embed_dim}, opp_embed={self.opponent_embed_dim}, "
            f"heads={self.num_heads}, layers={self.num_layers})"
        )
