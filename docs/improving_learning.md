# Improving Learning: Architecture Proposals

## 1. Hierarchical Q-Ensemble (Per-Action Value Heads)

**Problem solved:** Raise-or-fold polarization (call rate collapsed to 8-11%)

**Design:** Replace the single V(s) value head with per-action Q-value heads using an ensemble for uncertainty-driven exploration.

### Level 1: Action Type Q-Values (4 heads)
- `Q(s, fold)`, `Q(s, check)`, `Q(s, call)`, `Q(s, raise)`
- Equal representation — no bias toward any action
- Compete on equal footing for action selection

### Level 2: Sizing Q-Values (10 heads, conditional on raise)
- `Q(s, raise | size_k)` for each of the 10 sizing buckets
- Only used AFTER raise is selected at Level 1
- `Q(s, raise)` = weighted average of sizing Q-values

### Ensemble Uncertainty (3 copies per Q-head)
- Each Q-head is an ensemble of 3 networks
- `q_std = std across ensemble` = confidence signal
- **Exploration bonus**: `adjusted_logits = policy_logits + β × q_std`
- Under-sampled actions (call) have high ensemble disagreement → get sampled more → self-correcting

### PPO Integration
```python
# Value loss: only update Q-head for action taken
q_pred = Q_ensemble_mean(s, action_taken)
value_loss = smooth_l1_loss(q_pred, returns)

# Advantage: per-action
V_s = sum(π(a|s) * Q(s,a) for all a)
advantage = Q(s, action_taken) - V_s
```

### Pros
- Direct EV comparison between fold/call/raise
- Self-correcting exploration for undersampled actions
- Fine-grained sizing evaluation

### Cons
- Sparse supervision (each Q-head only learns from its own action)
- Breaks standard PPO theory (designed for V(s), not Q(s,a))
- 3× value head parameters (ensemble)
- Counterfactual problem: Q(s, call) when you actually raised is unobserved

---

## 2. Auxiliary Hand Strength Prediction Head

**Problem solved:** Card transformer doesn't learn hand evaluation → model can't distinguish call spots from raise spots

**Design:** Add a secondary training objective that predicts hand equity against a random range.

### Architecture
```python
self.equity_head = nn.Sequential(
    nn.Linear(embed_dim, embed_dim // 2),
    nn.GELU(),
    nn.Linear(embed_dim // 2, 1),
    nn.Sigmoid(),  # output: 0-1 equity
)
```

### Training Signal
- At showdown (or after all-in), compute actual equity from card matchup
- `aux_loss = MSE(predicted_equity, actual_equity)`
- Total loss: `policy_loss + value_coef * value_loss + aux_coef * aux_loss`

### Why It Helps
- Forces card transformer to learn actual hand evaluation
- Medium-strength hands get labeled as ~50% equity → model learns "this is a call spot"
- Gradient flows back through shared card encoder → improves policy decisions
- Does NOT change PPO math — purely additive auxiliary signal

### Pros
- Clean, doesn't touch PPO core
- Improves card representations for ALL downstream tasks
- Cheap to compute (hand equity available at showdown)

### Cons
- Only provides signal at showdown (not all hands reach showdown)
- Hand strength alone doesn't determine best action (position, stack depth, opponent tendencies matter too)
- Less direct impact on action selection than Q-heads

---

## Recommendation

Both are complementary:
- **Q-Ensemble** fixes the *learning loop* (explore undersampled actions)
- **Aux Hand Strength** fixes the *representation* (model understands hand quality)

Implement **Aux Hand Strength first** (simpler, no PPO changes) → then **Q-Ensemble** if polarization persists.
