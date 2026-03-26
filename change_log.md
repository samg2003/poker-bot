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

### 📊 Results (First 47 Epochs with Fixes)

| Metric | Before (epoch 200+) | After (epoch 47) |
|--------|---------------------|------------------|
| Loss | Near-zero, negative | 0.02 - 0.20 (positive) |
| Raise% | 5-6% (stuck) | 27% (climbing) |
| Check% | 72-77% (stuck) | 43-54% (exploring) |
| DAI | N/A (old metric) | 28% (improving) |
| Reward | Oscillating ±3bb | Trending positive +2-10bb |
