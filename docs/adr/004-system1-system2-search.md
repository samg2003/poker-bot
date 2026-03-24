# ADR-004: System 1 + System 2 (Policy + Lightweight Search)

**Status**: Accepted  
**Date**: 2024-03-24  
**Decision Makers**: Project team

## Context

A pure policy network gives fast answers (~1ms) but may miss complex multi-street reasoning in critical spots. Pluribus-style full search is too expensive for real-time play.

## Decision

**Dual-system approach:**

- **System 1 (Policy Network)**: Fast forward pass, handles ~95% of decisions
- **System 2 (Lightweight Search)**: Opponent-aware CFR on small subtrees for hard spots

### Search Trigger
```python
should_search = (
    pot_size > 20 * big_blind       # significant pot
    and policy_entropy > threshold   # model is uncertain
    and remaining_streets <= 2       # manageable tree size
)
```

### Search Mechanics
- Build subtree for 1-2 streets ahead
- Use opponent model for range estimation
- 50-100 CFR iterations with policy as leaf evaluator
- ~200ms latency (only on complex spots)

## Consequences

**Positive:**
- Best of both worlds: fast routine play + deep thinking when it matters
- Search uses opponent model (something Pluribus doesn't have)
- Policy warm-starts the search (much fewer iterations needed)

**Negative:**
- Added complexity in the inference pipeline
- Search quality depends on range estimation accuracy
- Latency spike on complex hands (~200ms vs ~1ms)

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| Policy-only (no search) | May miss complex multi-street reasoning in large pots |
| Full Pluribus-style search | Too slow for real-time play, doesn't use opponent reads |
