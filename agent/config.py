"""
Agent configuration — all tunable parameters in one place.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the poker agent."""

    # Model architecture
    embed_dim: int = 128
    opponent_embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3

    # Search (System 2)
    enable_search: bool = True
    search_min_pot_bb: float = 20.0
    search_entropy_threshold: float = 1.0
    search_iterations: int = 100

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
            f"heads={self.num_heads}, layers={self.num_layers}, "
            f"search={self.enable_search})"
        )
