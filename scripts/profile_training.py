"""Profile training bottlenecks — run from project root."""
import sys, os, time, functools, subprocess
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
import torch
import gc

def get_mem_mb():
    try:
        kb = int(subprocess.check_output(['ps', '-p', str(os.getpid()), '-o', 'rss=']).strip())
        return kb / 1024.0
    except:
        return 0.0

def get_gpu_mem_mb():
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024 / 1024
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

# Add project root to path
sys.path.insert(0, os.getcwd())

from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num-workers', type=int, default=0, help='Number of vector workers')
parser.add_argument('--device', type=str, default='auto', help='Device to run on (e.g. cpu, cuda, mps)')
args, _ = parser.parse_known_args()

config = NLHETrainingConfig(
    embed_dim=256, opponent_embed_dim=256, num_layers=4, num_heads=4,
    hands_per_epoch=100,
    min_players=2, max_players=6,
    min_bb=20, max_bb=200,
    num_workers=args.num_workers,
    device=args.device,
)
trainer = NLHESelfPlayTrainer(config)

# Monkey-patch key methods to time them
timers = {}
def make_timer(name, orig_fn):
    timers[name] = [0.0, 0]
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = orig_fn(*args, **kwargs)
        timers[name][0] += time.time() - t0
        timers[name][1] += 1
        return result
    return wrapper

trainer._get_opponent_stats = make_timer('get_opp_stats', trainer._get_opponent_stats)
# trainer._encode_game_state doesn't exist — encoding is inline in _play_hand_gen

if __name__ == "__main__":
    import random
    import torch.nn as nn
    print(f"Device: {trainer.device}")
    print(f"Running 10 epochs (100 hands each)...")

    for epoch in range(10):
        print(f"\n--- EPOCH {epoch+1} ---")
        print(f"Initial Mem: CPU {get_mem_mb():.1f}MB | GPU {get_gpu_mem_mb():.1f}MB")
        t0 = time.time()
        
        if config.num_workers > 0:
            all_exp, reward = trainer._run_vectorized_epoch()
        else:
            all_exp, reward = trainer._run_batched_epoch()
        sim_time = time.time() - t0
        print(f"Post-Sim Mem: CPU {get_mem_mb():.1f}MB | GPU {get_gpu_mem_mb():.1f}MB")

        trainer.policy.train()
        trainer.opponent_encoder.train()
        t1 = time.time()
        ppo_data = trainer._precompute_ppo_data(all_exp)
        precompute_time = time.time() - t1

        t2 = time.time()
        if ppo_data:
            n = len(all_exp)
            indices = list(range(n))
            for _ in range(4):
                random.shuffle(indices)
                for start in range(0, n, 64):
                    mb = indices[start:start+64]
                    if len(mb) < 64:
                        continue  # Drop last incomplete batch to avoid MPS graph leak!
                    trainer.optimizer.zero_grad()
                    trainer._compute_ppo_loss_minibatch(ppo_data, mb)
                    nn.utils.clip_grad_norm_(trainer.policy.parameters(), 1.0)
                    trainer.optimizer.step()
        ppo_time = time.time() - t2

        num_exps = len(all_exp)
        # Trigger a collection to see retained memory
        del all_exp
        del ppo_data
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Post-PPO/GC Mem: CPU {get_mem_mb():.1f}MB | GPU {get_gpu_mem_mb():.1f}MB")
        print(f"Collected {num_exps} experiences in {sim_time:.2f}s")
