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

# Allow MPS to use full available memory (prevents buffer allocation failures)
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datetime import datetime


from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig
from deployment.checkpoint import CheckpointManager, CheckpointMetadata
from agent.config import AgentConfig


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


    print(f"\n{'='*50}")
    print(f"Training on NLHE")
    print(f"  Players: {players_str}")
    print(f"  Stacks:  {stacks_str}")
    print(f"  Device:  {args.device}")
    print(f"  Epochs:  {args.epochs} | Hands/epoch: {args.hands}")
    print(f"{'='*50}\n")

    config = NLHETrainingConfig(
        embed_dim=args.embed_dim,
        opponent_embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        epsilon=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        hands_per_epoch=args.hands,
        log_interval=args.log_interval,
        num_players=args.num_players,
        starting_bb=args.starting_bb,
        min_players=args.min_players,
        max_players=args.max_players,
        min_bb=args.min_bb,
        max_bb=args.max_bb,
        device=args.device,
        mc_equity_sims=args.mc_equity_sims,
        verbose=args.verbose,
        batch_chunk_size=args.batch_chunk_size,
        num_workers=args.num_workers,
        seed=args.seed,
        frozen_update_interval=args.save_interval,
        remove_clip=args.remove_clip,
        kl_beta=args.kl_beta,
        v_res_alpha=args.v_res_alpha,

    )

    
    if args.entropy is not None:
        config.entropy_coef = args.entropy
        config.entropy_coef_end = args.entropy

    trainer = NLHESelfPlayTrainer(config=config)

    checkpoint_mgr = CheckpointManager(args.checkpoint_dir)

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        tag = args.resume
        reset_encoder = getattr(args, 'reset_hand_encoder', False)
        try:
            meta = checkpoint_mgr.load(
                trainer.policy, trainer.opponent_encoder,
                trainer.optimizer if not reset_encoder else None,  # skip stale optimizer on migration
                tag=tag,
                device=str(trainer.device),
                strict=not reset_encoder,
            )
            start_epoch = meta.epoch
            if reset_encoder:
                print(f"\n↺ --reset-hand-encoder: HandHistoryEncoder fresh-init, skipping pool load")
            else:
                # Load opponent pool from checkpoint
                load_dir = os.path.join(args.checkpoint_dir, tag)
                trainer.load_pool(load_dir)
                pool_total = len(trainer.pool.recent) + len(trainer.pool.archive)
            print(f"\n✓ Resumed from checkpoint '{tag}' (epoch {start_epoch}, reward={meta.avg_reward:+.3f})")
        except FileNotFoundError:
            print(f"\n✗ Checkpoint '{tag}' not found, training from scratch")

    def _make_metadata(epoch, metrics):
        avg_reward = metrics['epoch_reward'][-1] if metrics['epoch_reward'] else 0.0
        avg_loss = metrics['epoch_loss'][-1] if metrics['epoch_loss'] else 0.0
        return CheckpointMetadata(
            version=f"nlhe_v{epoch:04d}",
            created_at=datetime.now().isoformat(),
            epoch=epoch,
            stage=f"NLHE {players_str} {stacks_str}",
            total_hands=epoch * args.hands,
            avg_reward=avg_reward,
            loss=avg_loss,
            test_count=175,
            config={
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'game': 'nlhe',
            },
        )

    def _on_epoch(trainer, epoch, metrics):
        if args.save_interval > 0 and epoch % args.save_interval == 0:
            meta = _make_metadata(epoch, metrics)
            save_path = checkpoint_mgr.save(trainer.policy, trainer.opponent_encoder, trainer.optimizer, meta)
            trainer.save_pool(str(save_path))
            print(f"    ✓ Checkpoint saved at epoch {epoch}")

    metrics = trainer.train(num_epochs=args.epochs, epoch_callback=_on_epoch, start_epoch=start_epoch)

    # Final save
    meta = _make_metadata(args.epochs, metrics)
    save_path = checkpoint_mgr.save(trainer.policy, trainer.opponent_encoder, trainer.optimizer, meta)
    trainer.save_pool(str(save_path))
    checkpoint_mgr.save_best(trainer.policy, trainer.opponent_encoder, trainer.optimizer, meta)

    best_reward = max(metrics['epoch_reward']) if metrics['epoch_reward'] else 0.0
    print(f"\nTraining complete. Best reward: {best_reward:+.4f} bb")
    print(f"Checkpoint saved to: {args.checkpoint_dir}/latest")


def main():
    parser = argparse.ArgumentParser(description='Train poker AI')


    # Training params
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--hands', type=int, default=128,
                        help='Hands per epoch')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                        help='Number of PPO mini-batch passes per epoch')
    parser.add_argument('--epsilon-start', type=float, default=0.15,
                        help='Starting exploration rate')
    parser.add_argument('--epsilon-end', type=float, default=0.08,
                        help='Ending exploration rate')
    parser.add_argument('--entropy', type=float, default=None,
                        help='Constant entropy coefficient (overrides default decay)')
    parser.add_argument('--remove-clip', action='store_true',
                        help='Removes PPO hard clipping and uses KL divergence penalty instead')
    parser.add_argument('--kl-beta', type=float, default=1.0,
                        help='KL penalty coefficient if --remove-clip is used')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--v-res-alpha', type=float, default=1.0,
                        help='V_res influence on advantages (0=pure equity, 1=full V_res)')


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
    parser.add_argument('--mc-equity-sims', type=int, default=500,
                        help='MC equity simulations per decision (default: 500, lower=faster)')
    parser.add_argument('--batch-chunk-size', type=int, default=500,
                        help='Max simultaneous games per sub-batch (default: 500, lower to save memory)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of parallel Gym environments (default: 0 = sequential engine)')

    parser.add_argument('--reset-hand-encoder', action='store_true',
                        help='One-time migration: load checkpoint with strict=False to fresh-init '
                             'HandHistoryEncoder (GRU→Transformer). Skip pool load. Use once, then resume normally.')

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

    train_nlhe(args)


if __name__ == '__main__':
    main()
