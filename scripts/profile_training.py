"""Profile training bottlenecks — run from project root."""
import sys, os, time, functools

# Add project root to path
sys.path.insert(0, os.getcwd())

from training.nlhe_trainer import NLHESelfPlayTrainer, NLHETrainingConfig

config = NLHETrainingConfig(
    embed_dim=256, num_layers=4, num_heads=4,
    hands_per_epoch=100,
    min_players=2, max_players=6,
    min_bb=20, max_bb=200,
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

trainer._build_table_models = make_timer('build_table_models', trainer._build_table_models)
trainer._get_opponent_embedding = make_timer('get_opp_embedding', trainer._get_opponent_embedding)
trainer._get_all_opponent_embeddings = make_timer('get_all_opp_embeddings', trainer._get_all_opponent_embeddings)
trainer._get_opponent_stats = make_timer('get_opp_stats', trainer._get_opponent_stats)
# trainer._encode_game_state doesn't exist — encoding is inline in _play_hand_gen

# Run 1 epoch
print(f"Device: {trainer.device}")
print(f"Running 100 hands...")
t0 = time.time()
all_exp, reward = trainer._run_batched_epoch()
sim_time = time.time() - t0

# PPO
import torch.nn as nn
trainer.policy.train()
trainer.opponent_encoder.train()
t1 = time.time()
ppo_data = trainer._precompute_ppo_data(all_exp)
precompute_time = time.time() - t1

t2 = time.time()
if ppo_data:
    n = len(all_exp)
    indices = list(range(n))
    import random
    for _ in range(4):
        random.shuffle(indices)
        for start in range(0, n, 64):
            mb = indices[start:start+64]
            trainer.optimizer.zero_grad()
            trainer._compute_ppo_loss_minibatch(ppo_data, mb)
            nn.utils.clip_grad_norm_(trainer.policy.parameters(), 1.0)
            trainer.optimizer.step()
ppo_time = time.time() - t2

print(f"\n{'='*60}")
print(f"PROFILING RESULTS (100 hands)")
print(f"{'='*60}")
print(f"Total simulation:      {sim_time:.2f}s ({100/sim_time:.1f} hands/s)")
print(f"Total PPO update:      {ppo_time:.2f}s")
print(f"  Precompute (GAE):    {precompute_time:.2f}s")
print(f"  Mini-batch updates:  {ppo_time:.2f}s")
print(f"\nSimulation breakdown:")
for name, (total, count) in sorted(timers.items(), key=lambda x: -x[1][0]):
    avg = total/count if count else 0
    pct = total/sim_time*100
    print(f"  {name:30s}: {total:.3f}s ({count:5d} calls, {avg*1000:.1f}ms avg) [{pct:.0f}%]")
other = sim_time - sum(v[0] for k,v in timers.items())
print(f"  {'other (engine, batching)':30s}: {other:.3f}s [{other/sim_time*100:.0f}%]")
print(f"\n{len(all_exp)} experiences collected")
