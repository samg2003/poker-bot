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
    # Determine display string
    if args.num_players == 0:
        players_str = f"{args.min_players}-{args.max_players} players (random)"
    else:
        players_str = f"{args.num_players} players (fixed)"

    if args.starting_bb == 0:
        stacks_str = f"{args.min_bb}-{args.max_bb}bb (random per-player)"
    else:
        stacks_str = f"{args.starting_bb}bb (fixed)"

    search_str = f"{args.search_fraction*100:.0f}% hands" if args.search_fraction > 0 else "off"

    print(f"\n{'='*50}")
    print(f"Training on NLHE")
    print(f"  Players: {players_str}")
    print(f"  Stacks:  {stacks_str}")
    print(f"  Device:  {args.device}")
    print(f"  Search:  {search_str}")
    print(f"  Epochs:  {args.epochs} | Hands/epoch: {args.hands}")
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
        min_players=args.min_players,
        max_players=args.max_players,
        min_bb=args.min_bb,
        max_bb=args.max_bb,
        device=args.device,
        search_fraction=args.search_fraction,
        verbose=args.verbose,
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
        stage=f"NLHE {players_str} {stacks_str}",
        total_hands=args.epochs * args.hands,
        avg_reward=avg_reward,
        loss=avg_loss,
        test_count=175,
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
    parser.add_argument('--hands', type=int, default=128,
                        help='Hands per epoch')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Architecture
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=2,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of Transformer layers')

    # NLHE-specific
    parser.add_argument('--num-players', type=int, default=0,
                        help='Fixed player count (0 = random 2-6)')
    parser.add_argument('--starting-bb', type=int, default=0,
                        help='Fixed stack depth in BB (0 = random 20-200)')
    parser.add_argument('--min-players', type=int, default=2,
                        help='Min players when randomizing (default: 2)')
    parser.add_argument('--max-players', type=int, default=6,
                        help='Max players when randomizing (default: 6)')
    parser.add_argument('--min-bb', type=int, default=20,
                        help='Min stack depth in BB when randomizing (default: 20)')
    parser.add_argument('--max-bb', type=int, default=200,
                        help='Max stack depth in BB when randomizing (default: 200)')

    # Hardware & features
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of CPU threads to use (0 = unconstrained, default: 0)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, mps, cpu (default: auto)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with timing and progress updates')
    parser.add_argument('--search-fraction', type=float, default=0.0,
                        help='Fraction of hands using search (0-1, default: 0 = off)')

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

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.curriculum:
        train_curriculum(args)
    elif args.game == 'nlhe':
        train_nlhe(args)
    else:
        train_leduc(args)


if __name__ == '__main__':
    main()
