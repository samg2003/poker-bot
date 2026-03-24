# ADR-002: Emergent GTO via Self-Play + History Resets

**Status**: Accepted  
**Date**: 2024-03-24  
**Decision Makers**: Project team

## Context

The agent needs to play near-GTO when it has no reads on opponents, and shift to exploitative play as it gathers information. The original design had a separate GTO safety-net module.

## Decision

**No separate GTO module.** GTO behavior emerges naturally through two training mechanisms:

1. **Self-play**: Model copies play against each other → arms race → convergence toward equilibrium
2. **History resets every 300-500 hands**: After a reset, the model has zero opponent info. The optimal play with zero info *is* GTO. The network learns this automatically.

This creates a continuous spectrum:
- 0 hands after reset → near-GTO
- 50 hands → light exploitation
- 500+ hands → full exploitation

## Consequences

**Positive:**
- Simpler architecture (one network, no blending parameter α)
- No explicit equilibrium computation needed
- Natural degradation when reads are thin

**Negative:**
- No formal proof of convergence to Nash in multiplayer (mitigated by Kuhn/Leduc validation)

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| Separate GTO module (CFR-based) | Redundant — one network can handle both modes |
| 30% embedding dropout | Unnecessary — history resets already create zero-info scenarios organically |
