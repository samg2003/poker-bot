"""
Inference Engine — optimized inference for production use.

Features:
- Half-precision (fp16) support for faster inference
- Batched inference for multi-table play
- TorchScript export for deployment
- Latency benchmarking
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model.policy_network import PolicyNetwork
from model.action_space import ActionOutput
from model.opponent_encoder import OpponentEncoder
from model.stat_tracker import NUM_STAT_FEATURES
from agent.config import AgentConfig


@dataclass
class InferenceStats:
    """Stats from inference runs."""
    num_calls: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.num_calls, 1)

    def record(self, latency_ms: float) -> None:
        self.num_calls += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

    def summary(self) -> str:
        if self.num_calls == 0:
            return "No inference calls recorded."
        return (
            f"Inference Stats: {self.num_calls} calls | "
            f"avg={self.avg_latency_ms:.2f}ms | "
            f"min={self.min_latency_ms:.2f}ms | "
            f"max={self.max_latency_ms:.2f}ms"
        )


class InferenceEngine:
    """
    Optimized inference engine for production deployment.

    Usage:
        engine = InferenceEngine(policy, encoder)
        engine.optimize(half_precision=True)  # optional
        output = engine.infer(hole_cards, community, numeric, ...)
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        device: str = 'cpu',
    ):
        self.policy = policy
        self.opponent_encoder = opponent_encoder
        self.device = torch.device(device)
        self.stats = InferenceStats()
        self._is_optimized = False

        # Move to device
        self.policy.to(self.device)
        self.opponent_encoder.to(self.device)

    def optimize(self, half_precision: bool = False) -> None:
        """
        Apply inference optimizations.

        Args:
            half_precision: use fp16 (requires GPU)
        """
        self.policy.eval()
        self.opponent_encoder.eval()

        if half_precision and self.device.type == 'cuda':
            self.policy.half()
            self.opponent_encoder.half()

        # Disable gradient computation globally for inference
        for param in self.policy.parameters():
            param.requires_grad_(False)
        for param in self.opponent_encoder.parameters():
            param.requires_grad_(False)

        self._is_optimized = True

    @torch.no_grad()
    def infer(
        self,
        hole_cards: torch.Tensor,
        community_cards: torch.Tensor,
        numeric_features: torch.Tensor,
        opponent_embeddings: Optional[torch.Tensor] = None,
        opponent_stats: Optional[torch.Tensor] = None,
        own_stats: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> ActionOutput:
        """
        Run inference with latency tracking.

        All inputs should have batch dimension.
        """
        start = time.perf_counter()

        # Move to device
        hole_cards = hole_cards.to(self.device)
        community_cards = community_cards.to(self.device)
        numeric_features = numeric_features.to(self.device)

        batch_size = hole_cards.shape[0]

        if opponent_embeddings is None:
            opponent_embeddings = self.opponent_encoder.encode_empty(batch_size).unsqueeze(1).to(self.device)
        else:
            opponent_embeddings = opponent_embeddings.to(self.device)

        if opponent_stats is None:
            opponent_stats = torch.zeros(batch_size, 1, NUM_STAT_FEATURES, device=self.device)
        else:
            opponent_stats = opponent_stats.to(self.device)

        if own_stats is None:
            own_stats = torch.zeros(batch_size, NUM_STAT_FEATURES, device=self.device)
        else:
            own_stats = own_stats.to(self.device)

        if action_mask is not None:
            action_mask = action_mask.to(self.device)

        output = self.policy(
            hole_cards=hole_cards,
            community_cards=community_cards,
            numeric_features=numeric_features,
            opponent_embeddings=opponent_embeddings,
            opponent_stats=opponent_stats,
            own_stats=own_stats,
            action_mask=action_mask,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.record(elapsed_ms)

        return output

    @torch.no_grad()
    def infer_batch(
        self,
        inputs: List[Dict[str, torch.Tensor]],
    ) -> List[ActionOutput]:
        """
        Batched inference for multi-table play.

        Each input dict should have: hole_cards, community_cards, numeric_features
        Optional: opponent_embeddings, opponent_stats, own_stats, action_mask
        """
        if not inputs:
            return []

        # Stack into batched tensors
        batch = {
            'hole_cards': torch.stack([inp['hole_cards'] for inp in inputs]),
            'community_cards': torch.stack([inp['community_cards'] for inp in inputs]),
            'numeric_features': torch.stack([inp['numeric_features'] for inp in inputs]),
        }

        # Optional fields
        if 'action_mask' in inputs[0]:
            batch['action_mask'] = torch.stack([inp['action_mask'] for inp in inputs])

        output = self.infer(**batch)

        # Split back into individual outputs
        results = []
        for i in range(len(inputs)):
            results.append(ActionOutput(
                action_type_logits=output.action_type_logits[i:i+1],
                action_type_probs=output.action_type_probs[i:i+1],
                bet_size_logits=output.bet_size_logits[i:i+1],
                value=output.value[i:i+1],
            ))

        return results

    def benchmark(self, num_iterations: int = 100, batch_size: int = 1) -> InferenceStats:
        """
        Run a latency benchmark.

        Returns inference stats after num_iterations forward passes.
        """
        self.stats = InferenceStats()  # reset
        self.policy.eval()

        for _ in range(num_iterations):
            hole = torch.randint(0, 52, (batch_size, 2))
            community = torch.full((batch_size, 5), -1, dtype=torch.long)
            numeric = torch.randn(batch_size, 10).to(self.device)

            self.infer(hole, community, numeric)

        return self.stats

    def export_torchscript(self, path: str) -> None:
        """
        Export the policy network as TorchScript for deployment.

        Note: This uses tracing, so the model must behave consistently.
        """
        self.policy.eval()

        # Create example inputs for tracing
        hole = torch.randint(0, 52, (1, 2))
        community = torch.full((1, 5), -1, dtype=torch.long)
        numeric = torch.randn(1, 10).to(self.device)
        opp_embed = self.opponent_encoder.encode_empty(1).unsqueeze(1)
        opp_stats = torch.zeros(1, 1, NUM_STAT_FEATURES)
        own_stats = torch.zeros(1, NUM_STAT_FEATURES)

        # TorchScript trace
        try:
            traced = torch.jit.trace(
                self.policy,
                (hole, community, numeric, opp_embed, opp_stats, own_stats),
            )
            torch.jit.save(traced, path)
        except Exception as e:
            # Fallback: save as regular state_dict
            torch.save(self.policy.state_dict(), path)
            raise RuntimeError(
                f"TorchScript export failed (saved state_dict instead): {e}"
            )

    def get_model_size_mb(self) -> float:
        """Get total model size in MB."""
        total_bytes = 0
        for p in self.policy.parameters():
            total_bytes += p.numel() * p.element_size()
        for p in self.opponent_encoder.parameters():
            total_bytes += p.numel() * p.element_size()
        return total_bytes / (1024 * 1024)
