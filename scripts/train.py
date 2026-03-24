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
from training.self_play_trainer import LeducSelfPlayTrainer, TrainingConfig
from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig
from deployment.checkpoint import CheckpointManager, CheckpointMetadata
from agent.config import AgentConfig


def train_leduc(args):
    """Train on Leduc Hold'em for fast validation."""
    print(f"\n{'='*50}")
    print(f"Training on Leduc Hold'em")
    print(f"Epochs: {args.epochs} | Hands/epoch: {args.hands}")
    print(f"{'='*50}\n")

    config = TrainingConfig(
        embed_dim=args.embed_dim,
        opponent_embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        hands_per_epoch=args.hands,
        log_interval=args.log_interval,
    )
    trainer = LeducSelfPlayTrainer(config=config, seed=args.seed)

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    # Train
    metrics = trainer.train(num_epochs=args.epochs)

    # Save final checkpoint
    avg_reward = metrics['epoch_reward'][-1] if metrics['epoch_reward'] else 0.0
    avg_loss = metrics['epoch_loss'][-1] if metrics['epoch_loss'] else 0.0

    metadata = CheckpointMetadata(
        version=f"leduc_v{args.epochs:04d}",
        created_at=datetime.now().isoformat(),
        epoch=args.epochs,
        stage="Leduc Self-Play",
        total_hands=args.epochs * args.hands,
        avg_reward=avg_reward,
        loss=avg_loss,
        test_count=168,
    )
    checkpoint_mgr.save(
        trainer.policy, trainer.opponent_encoder,
        trainer.optimizer, metadata,
    )
    checkpoint_mgr.save_best(
        trainer.policy, trainer.opponent_encoder,
        trainer.optimizer, metadata,
    )

    best_reward = max(metrics['epoch_reward']) if metrics['epoch_reward'] else 0.0
    print(f"\nTraining complete. Best reward: {best_reward:+.4f}")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/latest")


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


def train_nlhe(args):
    """Train on full No-Limit Hold'em."""
    print(f"\n{'='*50}")
    print(f"Training on NLHE ({args.num_players} players, {args.starting_bb}bb)")
    print(f"Epochs: {args.epochs} | Hands/epoch: {args.hands}")
    print(f"{'='*50}\n")

    config = NLHETrainingConfig(
        embed_dim=args.embed_dim,
        opponent_embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        hands_per_epoch=args.hands,
        log_interval=args.log_interval,
        num_players=args.num_players,
        starting_bb=args.starting_bb,
    )
    trainer = NLHESelfPlayTrainer(config=config, seed=args.seed)

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)
    metrics = trainer.train(num_epochs=args.epochs)

    # Save checkpoint
    avg_reward = metrics['epoch_reward'][-1] if metrics['epoch_reward'] else 0.0
    avg_loss = metrics['epoch_loss'][-1] if metrics['epoch_loss'] else 0.0

    metadata = CheckpointMetadata(
        version=f"nlhe_v{args.epochs:04d}",
        created_at=datetime.now().isoformat(),
        epoch=args.epochs,
        stage=f"NLHE {args.num_players}p {args.starting_bb}bb",
        total_hands=args.epochs * args.hands,
        avg_reward=avg_reward,
        loss=avg_loss,
        test_count=168,
    )
    checkpoint_mgr.save(trainer.policy, trainer.opponent_encoder, trainer.optimizer, metadata)
    checkpoint_mgr.save_best(trainer.policy, trainer.opponent_encoder, trainer.optimizer, metadata)

    best_reward = max(metrics['epoch_reward']) if metrics['epoch_reward'] else 0.0
    print(f"\nTraining complete. Best reward: {best_reward:+.4f} bb")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/latest")


def main():
    parser = argparse.ArgumentParser(description='Train poker AI')

    # Game selection
    parser.add_argument('--game', choices=['leduc', 'nlhe'], default='leduc',
                        help='Game to train on (default: leduc)')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum training (Leduc only)')

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

    # NLHE-specific
    parser.add_argument('--num-players', type=int, default=2,
                        help='Number of players (NLHE, 2-9)')
    parser.add_argument('--starting-bb', type=int, default=100,
                        help='Starting stack in big blinds')

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
    elif args.game == 'nlhe':
        train_nlhe(args)
    else:
        train_leduc(args)


if __name__ == '__main__':
    main()
