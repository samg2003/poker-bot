#!/usr/bin/env python3
"""
Training CLI — run curriculum training from the command line.

Usage:
    # Quick Leduc training (local, no GPU needed)
    python scripts/train.py --game leduc --epochs 100

    # Full curriculum training
    python scripts/train.py --game nlhe --curriculum --epochs 500

    # Resume from checkpoint
    python scripts/train.py --game leduc --resume checkpoints/latest
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datetime import datetime

from training.curriculum import CurriculumTrainer, CurriculumConfig, CurriculumStage
from training.self_play_trainer import LeducSelfPlayTrainer
from deployment.checkpoint import CheckpointManager, CheckpointMetadata
from agent.config import AgentConfig


def train_leduc(args):
    """Train on Leduc Hold'em for fast validation."""
    print(f"\n{'='*50}")
    print(f"Training on Leduc Hold'em")
    print(f"Epochs: {args.epochs} | Hands/epoch: {args.hands}")
    print(f"{'='*50}\n")

    trainer = LeducSelfPlayTrainer(
        embed_dim=args.embed_dim,
        opponent_embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        hands_per_epoch=args.hands,
    )

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    best_reward = float('-inf')
    for epoch in range(args.epochs):
        avg_reward, avg_loss = trainer.train_epoch()

        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch {epoch+1:4d} | Reward: {avg_reward:+.4f} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            metadata = CheckpointMetadata(
                version=f"leduc_v{epoch+1:04d}",
                created_at=datetime.now().isoformat(),
                epoch=epoch + 1,
                stage="Leduc Self-Play",
                total_hands=(epoch + 1) * args.hands,
                avg_reward=avg_reward,
                loss=avg_loss,
                test_count=154,
            )
            checkpoint_mgr.save(
                trainer.policy, trainer.opponent_encoder,
                trainer.optimizer, metadata,
            )

            if avg_reward > best_reward:
                best_reward = avg_reward
                checkpoint_mgr.save_best(
                    trainer.policy, trainer.opponent_encoder,
                    trainer.optimizer, metadata,
                )

    print(f"\nTraining complete. Best reward: {best_reward:+.4f}")


def train_curriculum(args):
    """Train with full curriculum."""
    print(f"\n{'='*50}")
    print(f"Curriculum Training")
    print(f"Max epochs: {args.epochs} | Hands/epoch: {args.hands}")
    print(f"{'='*50}\n")

    config = CurriculumConfig(
        embed_dim=args.embed_dim,
        opponent_embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        hands_per_epoch=args.hands,
        log_interval=args.log_interval,
        checkpoint_interval=args.save_interval,
    )

    trainer = CurriculumTrainer(config=config, seed=args.seed)
    metrics = trainer.train(max_epochs=args.epochs)

    print(f"\nTraining complete.")
    print(f"Final stage: {metrics.current_stage}")
    print(f"Stage transitions: {metrics.stage_transitions}")
    if metrics.epoch_rewards:
        print(f"Final reward: {metrics.epoch_rewards[-1]:+.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train poker AI')

    # Game selection
    parser.add_argument('--game', choices=['leduc', 'nlhe'], default='leduc',
                        help='Game to train on (default: leduc)')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum training')

    # Training params
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--hands', type=int, default=512,
                        help='Hands per epoch')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Architecture
    parser.add_argument('--embed-dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of Transformer layers')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (path to tag)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N epochs')

    args = parser.parse_args()

    if args.curriculum:
        train_curriculum(args)
    else:
        train_leduc(args)


if __name__ == '__main__':
    main()
