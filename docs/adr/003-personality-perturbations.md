# ADR-003: Self-Play + Personality Perturbations (No Scripted Bots)

**Status**: Accepted  
**Date**: 2024-03-24  
**Decision Makers**: Project team

## Context

The agent needs diverse training opponents to learn exploitation. The initial design used hard-coded scripted bots (TAG, LAG, Nit, etc.) with rule-based decision trees.

## Decision

**No scripted bots.** Instead:

1. **Phase 1**: Pure self-play — model copies play each other, converge toward GTO
2. **Phase 2**: Apply **personality perturbations** to the learned policy — continuous modifiers that reshape the action distribution

```python
class PersonalityModifier:
    range_mult: float      # <1 = tighter, >1 = looser
    aggression_mult: float # <1 = passive, >1 = aggressive
    fold_pressure: float   # >1 = overfolds, <1 = calls down
    bluff_mult: float      # >1 = bluffs more, <1 = bluffs less
```

A "Nit" is the GTO model with `range_mult=0.5` — it's still a competent player that understands pot odds and position, it just plays too few hands.

### Situational Perturbations

Modifiers are **context-dependent** with per-situation overrides:
- Per-street (preflop vs. river behavior)
- Board texture (wet vs. dry)
- Position-dependent (IP vs. OOP)
- Stack-depth-dependent
- Tilt triggers (after losing big pots)

This creates realistic opponents like "tight preflop but chases draws" or "overbets with nuts, min-bets as bluff."

## Consequences

**Positive:**
- Perturbed opponents are still fundamentally competent (not cartoon bots)
- Agent learns to detect subtle deviations from GTO (the real skill)
- Continuous space of personalities, not discrete archetypes
- No hand-crafted decision logic to maintain

**Negative:**
- Perturbation design affects what exploits the agent learns
- May miss some human tendencies not captured by the modifier axes

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| Hard-coded scripted bots | Artificial behavior, agent learns to beat decision trees not real players |
| Human hand history pre-training | Difficult to integrate with RL pipeline, teaches human mistakes |
