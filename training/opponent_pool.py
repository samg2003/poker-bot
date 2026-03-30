"""
Opponent Pool — manages frozen opponent snapshots for self-play training.

Two-tier pool:
- Recent: FIFO queue of latest checkpoints (default 10)
- Archive: Random replacement of older checkpoints (default 5)
"""

from __future__ import annotations

import os
import glob
import random
from typing import Dict, List, Optional

import torch

from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder


class OpponentPool:
    """Manages frozen opponent models for diverse self-play training."""

    def __init__(
        self,
        embed_dim: int,
        opponent_embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_recent: int = 10,
        max_archive: int = 5,
        rng: Optional[random.Random] = None,
    ):
        self.embed_dim = embed_dim
        self.opponent_embed_dim = opponent_embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_recent = max_recent
        self.max_archive = max_archive
        self.rng = rng or random.Random()

        self.recent: List[dict] = []
        self.archive: List[dict] = []
        self._frozen_models: Dict[int, PolicyNetwork] = {}

        # Frozen opponent encoder (separate from training encoder)
        self.frozen_encoder = OpponentEncoder(
            embed_dim=opponent_embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.frozen_encoder.eval()
        for p in self.frozen_encoder.parameters():
            p.requires_grad = False

    def sync(self, policy_state: dict, encoder_state: dict) -> None:
        """Add current live weights to pool and update frozen encoder."""
        cpu_state = {k: v.cpu() for k, v in policy_state.items()}

        # Evict oldest recent entry → 30% chance it goes to archive
        if len(self.recent) >= self.max_recent:
            evicted = self.recent.pop(0)
            if len(self.archive) < self.max_archive:
                self.archive.append(evicted)
            elif self.rng.random() < 0.30:
                idx = self.rng.randint(0, len(self.archive) - 1)
                self.archive[idx] = evicted

        self.recent.append(cpu_state)

        # Sync frozen encoder
        cpu_enc = {k: v.cpu() for k, v in encoder_state.items()}
        self.frozen_encoder.load_state_dict(cpu_enc)

        # Force rebuild of frozen models on next table setup
        self._frozen_models = {}

    def get_combined(self) -> List[dict]:
        """Get combined pool (recent + archive) as a single list."""
        return self.recent + self.archive

    def build_table_models(self, seat_pool_idx: Dict[int, int]) -> None:
        """Build frozen models for each unique pool index at the table."""
        combined = self.get_combined()
        for idx in set(seat_pool_idx.values()):
            if idx not in self._frozen_models and idx < len(combined):
                self._frozen_models[idx] = self._make_model(combined[idx])

    def get_model(self, pool_idx: int) -> Optional[PolicyNetwork]:
        """Get a frozen model by pool index."""
        return self._frozen_models.get(pool_idx)

    def _make_model(self, state_dict: dict) -> PolicyNetwork:
        """Create a fresh frozen model from state dict."""
        model = PolicyNetwork(
            embed_dim=self.embed_dim,
            num_cross_attn_heads=self.num_heads,
            num_cross_attn_layers=self.num_layers,
            opponent_embed_dim=self.opponent_embed_dim,
        ).to('cpu')
        model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def save(self, save_dir: str) -> None:
        """Save opponent pool to disk for checkpoint persistence."""
        pool_dir = os.path.join(save_dir, 'pool')
        os.makedirs(pool_dir, exist_ok=True)
        for i, sd in enumerate(self.recent):
            torch.save(sd, os.path.join(pool_dir, f'recent_{i:03d}.pt'))
        for i, sd in enumerate(self.archive):
            torch.save(sd, os.path.join(pool_dir, f'archive_{i:03d}.pt'))

    def load(self, load_dir: str) -> None:
        """Load opponent pool from disk."""
        pool_dir = os.path.join(load_dir, 'pool')
        if not os.path.exists(pool_dir):
            return
        recent_files = sorted(glob.glob(os.path.join(pool_dir, 'recent_*.pt')))
        if recent_files:
            self.recent = [
                torch.load(f, map_location='cpu', weights_only=True) for f in recent_files
            ]
        archive_files = sorted(glob.glob(os.path.join(pool_dir, 'archive_*.pt')))
        if archive_files:
            self.archive = [
                torch.load(f, map_location='cpu', weights_only=True) for f in archive_files
            ]
        self._frozen_models = {}
