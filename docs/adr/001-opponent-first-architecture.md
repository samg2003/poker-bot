# ADR-001: Opponent-First Architecture

**Status**: Accepted  
**Date**: 2024-03-24  
**Decision Makers**: Project team

## Context

Traditional poker AI pipelines (Pluribus/Libratus) compute a static GTO strategy via CFR, then bolt on exploitation as a post-hoc adjustment. This has three problems:

1. CFR loses Nash convergence guarantees beyond 2 players
2. Scaling search from 2→9 players is computationally intractable
3. Opponent modeling is an afterthought — the model never *learns to read opponents*

## Decision

Use a **unified Transformer architecture** where opponent embeddings are first-class inputs to the policy network via cross-attention. The network cannot make a decision without considering who it's playing against.

```
Opponent Encoder → embeddings → cross-attention → Policy Network → action + value
```

## Consequences

**Positive:**
- Opponent understanding feeds every decision from the start
- In-context learning: reads improve as hands accumulate, without weight updates
- Natural multiplayer scaling via attention masks (2-9 players)
- One model, continuous spectrum from GTO to full exploitation

**Negative:**
- No formal equilibrium guarantees (mitigated by self-play → GTO emergence)
- Quality depends on opponent encoder capacity and training data diversity

## Alternatives Considered

| Approach | Why rejected |
|---|---|
| CFR + exploitation bolt-on (Pluribus) | Opponent modeling is 3rd class, doesn't scale to 9p |
| Pure end-to-end RL (AlphaHoldem) | Implicit-only opponent awareness, weak exploitation |
