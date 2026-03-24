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

    args = parser.parse_args()

    # Create models
    policy = PolicyNetwork(
        embed_dim=args.embed_dim,
        opponent_embed_dim=args.embed_dim,
        num_cross_attn_heads=args.num_heads,
        num_cross_attn_layers=args.num_layers,
    )
    encoder = OpponentEncoder(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        mgr = CheckpointManager(args.checkpoint_dir)
        metadata = mgr.load(policy, encoder, tag=args.checkpoint)
        print(f"Loaded checkpoint: {metadata.version} (epoch {metadata.epoch})")
        print()

    # Run evaluation benchmarks
    print("Running evaluation benchmarks...")
    print()

    evaluator = Evaluator(policy, encoder, seed=args.seed, num_hands=args.num_hands)
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
