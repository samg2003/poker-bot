"""
Model Checkpoint System — save, load, and version trained models.

Handles:
- Saving full training state (model weights + optimizer + epoch + metrics)
- Loading for resumed training or inference
- Version tracking with metadata (config, training stage, test results)
- Best model tracking (auto-saves when new best is achieved)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from agent.config import AgentConfig


@dataclass
class CheckpointMetadata:
    """Metadata stored alongside each checkpoint."""
    version: str
    created_at: str
    epoch: int
    stage: str                     # curriculum stage name
    total_hands: int
    avg_reward: float
    loss: float
    test_count: int                # number of tests passing
    config: Dict[str, Any] = None
    notes: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'CheckpointMetadata':
        return cls(**d)


class CheckpointManager:
    """
    Manages model checkpoints on disk.

    Directory structure:
        checkpoints/
        ├── latest/
        │   ├── policy.pt
        │   ├── opponent_encoder.pt
        │   ├── optimizer.pt
        │   └── metadata.json
        ├── best/
        │   └── ...
        ├── v001/
        │   └── ...
        └── versions.json
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.versions_file = self.checkpoint_dir / 'versions.json'
        self._versions: List[Dict] = []

        if self.versions_file.exists():
            with open(self.versions_file) as f:
                self._versions = json.load(f)

    def save(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        optimizer: torch.optim.Optimizer,
        metadata: CheckpointMetadata,
        tag: str = 'latest',
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            policy: policy network
            opponent_encoder: opponent encoder
            optimizer: optimizer state
            metadata: training metadata
            tag: 'latest', 'best', or version string (e.g., 'v001')

        Returns:
            Path to the saved checkpoint directory.
        """
        save_dir = self.checkpoint_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(policy.state_dict(), save_dir / 'policy.pt')
        torch.save(opponent_encoder.state_dict(), save_dir / 'opponent_encoder.pt')
        torch.save(optimizer.state_dict(), save_dir / 'optimizer.pt')



        # Save metadata
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Track version
        version_entry = {
            'tag': tag,
            'version': metadata.version,
            'epoch': metadata.epoch,
            'stage': metadata.stage,
            'avg_reward': metadata.avg_reward,
            'created_at': metadata.created_at,
        }
        self._versions.append(version_entry)
        self._save_versions()

        return save_dir

    def load(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tag: str = 'latest',
        device: str = 'cpu',
        strict: bool = True,
    ) -> CheckpointMetadata:
        """
        Load a checkpoint.

        Args:
            policy: policy network (weights loaded in-place)
            opponent_encoder: opponent encoder (weights loaded in-place)
            optimizer: optional optimizer (for resumed training)
            tag: which checkpoint to load
            device: target device

        Returns:
            Loaded metadata.
        """
        load_dir = self.checkpoint_dir / tag
        if not load_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_dir}")

        missing, unexpected = policy.load_state_dict(
            torch.load(load_dir / 'policy.pt', map_location=device, weights_only=True),
            strict=strict,
        )
        if not strict and missing:
            print(f"[Checkpoint] Fresh-init keys (expected for migrations): {len(missing)} params")
        opponent_encoder.load_state_dict(
            torch.load(load_dir / 'opponent_encoder.pt', map_location=device, weights_only=True)
        )

        if optimizer and (load_dir / 'optimizer.pt').exists():
            optimizer.load_state_dict(
                torch.load(load_dir / 'optimizer.pt', map_location=device, weights_only=True)
            )



        with open(load_dir / 'metadata.json') as f:
            metadata = CheckpointMetadata.from_dict(json.load(f))

        return metadata

    def save_best(
        self,
        policy: PolicyNetwork,
        opponent_encoder: OpponentEncoder,
        optimizer: torch.optim.Optimizer,
        metadata: CheckpointMetadata,
        **kwargs,
    ) -> Optional[Path]:
        """
        Save as best if avg_reward is better than current best.

        Returns save path if saved, None if not a new best.
        """
        best_dir = self.checkpoint_dir / 'best'
        if best_dir.exists() and (best_dir / 'metadata.json').exists():
            with open(best_dir / 'metadata.json') as f:
                old = json.load(f)
            if metadata.avg_reward <= old.get('avg_reward', float('-inf')):
                return None

        return self.save(policy, opponent_encoder, optimizer, metadata, tag='best', **kwargs)

    def list_versions(self) -> List[Dict]:
        """List all saved versions."""
        return list(self._versions)

    def _save_versions(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(self._versions, f, indent=2)
