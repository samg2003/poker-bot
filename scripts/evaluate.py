#!/usr/bin/env python3
"""
Evaluation CLI — run benchmarks and generate reports.

Usage:
    # Run all benchmarks
    python scripts/evaluate.py

    # Run with more hands for accuracy
    python scripts/evaluate.py --num-hands 1000

    # Run with latency benchmark
    python scripts/evaluate.py --benchmark-latency
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.policy_network import PolicyNetwork
from model.opponent_encoder import OpponentEncoder
from evaluation.evaluator import Evaluator
from deployment.inference import InferenceEngine
from deployment.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description='Evaluate poker AI')
    parser.add_argument('--num-hands', type=int, default=500,
                        help='Number of hands per benchmark')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load from checkpoint (tag)')
    parser.add_argument('--benchmark-latency', action='store_true',
                        help='Run latency benchmark')
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory where checkpoints are stored')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--game', type=str, choices=['leduc', 'nlhe'], default='leduc',
                        help='Game to evaluate (default: leduc)')

    args = parser.parse_args()

    # Auto-load architecture config from checkpoint if available
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_layers = args.num_layers

    if args.checkpoint:
        mgr = CheckpointManager(args.checkpoint_dir)
        meta_path = mgr.checkpoint_dir / args.checkpoint / 'metadata.json'
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            cfg = meta.get('config') or {}
            if cfg:
                embed_dim = cfg.get('embed_dim', embed_dim)
                num_heads = cfg.get('num_heads', num_heads)
                num_layers = cfg.get('num_layers', num_layers)
                args.game = cfg.get('game', args.game)
                print(f"Auto-loaded config: game={args.game}, embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}")

    # Create models
    policy = PolicyNetwork(
        embed_dim=embed_dim,
        opponent_embed_dim=embed_dim,
        num_cross_attn_heads=num_heads,
        num_cross_attn_layers=num_layers,
    )
    encoder = OpponentEncoder(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        mgr = CheckpointManager(args.checkpoint_dir)
        metadata = mgr.load(policy, encoder, tag=args.checkpoint)
        print(f"Loaded checkpoint: {metadata.version} (epoch {metadata.epoch})")
        print()

    # Run evaluation benchmarks
    print(f"Running evaluation benchmarks for {args.game.upper()}...")
    print()

    evaluator = Evaluator(policy, encoder, seed=args.seed, num_hands=args.num_hands, game=args.game)
    results = evaluator.run_all_benchmarks()
    print(results.summary())

    # Optional latency benchmark
    if args.benchmark_latency:
        print()
        print("Running latency benchmark...")
        engine = InferenceEngine(policy, encoder)
        engine.optimize()
        stats = engine.benchmark(num_iterations=100)
        print(stats.summary())
        print(f"Model size: {engine.get_model_size_mb():.2f} MB")


if __name__ == '__main__':
    main()
