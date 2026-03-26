# Changelog

All notable changes to the poker AI training pipeline are documented here.

---

## [2026-03-25] Training Pipeline Overhaul

### 🔴 Critical Fixes

#### Entropy Coefficient Reduction
- **Before:** `entropy_coef=0.05`, `entropy_coef_end=0.01`
- **After:** `entropy_coef=0.005`, `entropy_coef_end=0.001`
- **Why:** Entropy bonus was 6x larger than policy gradient signal, causing negative loss and preventing learning. Confirmed by loss values going consistently negative.

#### Mini-Batch PPO
- **Before:** Single forward pass on all ~1500 experiences, 4 PPO epochs = 4 gradient updates
- **After:** Shuffled mini-batches of 64 samples, 4 PPO epochs ≈ 92 gradient updates
- **Why:** Dramatically improves sample efficiency — each mini-batch gives a unique gradient signal from a different subset of experiences.

#### Epsilon-Greedy Exploration
- **Before:** Exploration floor (min 5% probability per action type, off-policy PPO bug)
- **After:** Epsilon-greedy (`ε=0.15→0.08`), separate epsilon for both action type AND sizing
- **Why:** Floor was too weak for genuine exploration and caused an off-policy error (log_prob from raw model, action from floored distribution). Epsilon-greedy forces complete random actions, with log_prob always from model's own distribution.

### 🟡 Moderate Improvements

#### Decoupled Action/Sizing PPO Loss
- **Before:** Combined `log_prob = action_log_prob + sizing_log_prob`, single ratio/advantage
- **After:** Independent PPO losses for action type and sizing head
- **Why:** Bad all-in sizing was teaching the model "don't raise" instead of "raise with smaller sizing." Decoupled credit assignment lets the action head learn WHEN to raise independently from the sizing head learning HOW MUCH.

#### Soft Advantage Normalization
- **Before:** Raw GAE advantages (range: -200 to +200), causing loss spikes up to 930
- **After:** `advantages / max(std, 1.0)` — scales to ~unit variance, preserves sign
- **Why:** Mini-batch PPO clusters big-pot experiences randomly, creating extreme per-batch gradients. Normalization bounds magnitude while still letting big mistakes teach louder than small ones. Floor of 1.0 (not 0.1) ensures big losses remain impactful.

#### DAI Metric Redesign
- **Before:** `DAI = deep_all_ins / total_all_ins` (misleading — correlated with stack distribution)
- **After:** `DAI = deep_all_ins / total_deep_raises` ("when deep and raising, what % is all-in?")
- **Why:** Old metric was ~85% simply because 70% of stacks are >50bb. New metric directly measures sizing discipline at deep stacks. Healthy target: 10-20%.

#### Separate Action/Sizing Log Probs for Decoupled PPO
- **Before:** Only stored combined `log_prob = action_lp + sizing_lp` — decoupled sizing ratio used current action log prob to approximate old sizing log prob, which drifted after 64 mini-batch updates causing ratio explosion (loss = 9000+)
- **After:** Store `action_log_prob` and `sizing_log_prob` separately in Experience — decoupled loss uses exact old values, no approximation
- **Why:** Root cause of loss spikes was the sizing ratio using stale approximations, not the advantage magnitude.

### ⚡ Performance

#### Frozen Model Caching
- **Before:** `_build_table_models` rebuilt fresh PolicyNetwork instances every hand when player count changed (68% of simulation time)
- **After:** Cache models by pool index, only build for new indices
- **Result:** 0.53s → 0.01s per 20 hands (53x faster)

#### Batched Opponent Encoder
- **Before:** Individual forward pass per opponent per decision (3381 calls, 44% of simulation time)
- **After:** Single batched forward pass for all opponents with padded sequences
- Patched `TransformerEncoder` inside `OpponentEncoder` to set `enable_nested_tensor=False`. This completely bypasses the MPS `aten::_nested_tensor_from_mask_left_aligned` bug and allows padded sequence batching on Apple Silicon.
- **Expected:** 12 hands/s → 25-30 hands/s on MPS

### Massive Batched Inference & TableState Isolation (March 25, 2026)
- **State Leak Fixed**: Identified a critical bug where 100 parallel games running in a chunk were all mutating the *same* global `self.action_histories` and `self.stat_tracker`. This meant independent opponents from completely different hands were clobbering each other's historical tracks. Fixed by isolating game memory into persistent `TableState` environments.
- **Reward Scaling Bug Fixed**: Discovered a hardcoded `/ 100.0` reward scalar inside `_play_hand_gen` that was causing training advantages to shrink by 100x when `config.big_blind` was 1.0. This paralyzed the RL gradients, freezing the model structure wherever it was loaded from. Changed to `/ max(config.big_blind, 1)`.
- **Global Batching Rewrite**: Completely rewrote the core `_play_hand_gen` and `_run_batched_epoch` architecture.
  - Sub-generators now pause and yield state queries for BOTH the Hero (live policy) and the Opponents (frozen pool models).
  - Neural network inference is no longer processed game-by-game. Instead, `_run_batched_epoch` collects queries from *all* active games, groups them by target model identity, and evaluates them batched on the GPU simultaneously.
  - The `OpponentEncoder` is also batched across all parallel game environments at once.
- **Result**: Completely removed the O(num_opponents) sequential CPU bottleneck on frozen models. Training throughput surged from ~17 hands/sec to roughly **30 hands/sec** on MPS.

### 📊 Results (First 47 Epochs with Fixes)

| Metric | Before (epoch 200+) | After (epoch 47) |
|--------|---------------------|------------------|
| Loss | Near-zero, negative | 0.02 - 0.20 (positive) |
| Raise% | 5-6% (stuck) | 27% (climbing) |
| Check% | 72-77% (stuck) | 43-54% (exploring) |
| DAI | N/A (old metric) | 28% (improving) |
| Reward | Oscillating ±3bb | Trending positive +2-10bb |
