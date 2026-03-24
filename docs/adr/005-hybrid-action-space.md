# ADR-005: Hybrid Action Space (Discrete Type + Continuous Sizing)

**Status**: Accepted  
**Date**: 2024-03-24  
**Decision Makers**: Project team

## Context

NLHE has a continuous action space (any bet size from min-raise to all-in). The model needs to both choose action types and select precise bet sizes.

## Decision

**Hybrid two-headed output:**

```
Head 1 (action type):  [fold, check, call, raise]  ← 4-way softmax
Head 2 (bet sizing):   continuous value in [min_raise, all_in] as pot fraction
                       (only used when action_type = raise)
```

### Input Handling
Opponent bet sizes are represented as continuous pot-fraction values, not bucketed. The model can respond to any bet size (e.g., 0.37× pot, 2.7× pot).

## Consequences

**Positive:**
- Can express any bet size precisely (not limited to preset buckets)
- Can learn sizing tells (villain's bet size patterns)
- Handles arbitrary opponent bet sizes as input
- Enables polarized vs. merged range sizing strategies

**Negative:**
- Two-headed training is slightly more complex
- Continuous output needs careful loss function design (e.g., beta distribution)

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| Pure discrete bins (9 actions) | Can't express precise sizing, misses sizing tells |
| Fine-grained bins (14+ bins) | Still can't do arbitrary sizing, wasteful |
