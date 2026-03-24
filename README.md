# 🃏 Opponent-Aware Poker AI

A world-class No-Limit Texas Hold'em AI that prioritizes **exploitative play** through advanced opponent modeling.

## Architecture Overview

```
Opponent Encoder (Transformer) ──→ opponent embeddings
                                         │
Game State ──→ Policy Network (cross-attention) ──→ System 1 (fast, ~1ms)
                                                         │
                                                    Complex spot?
                                                    ╱          ╲
                                                  NO           YES
                                                   │    Lightweight Search
                                                   │     (System 2, ~200ms)
                                                   └──→ Final Action
```

**Key innovations:**
- Opponent embeddings are **first-class inputs** to the policy (not bolt-on)
- **GTO emerges naturally** — after history resets, model has no reads → plays equilibrium
- **Self-play → personality perturbations** — no scripted bots, realistic opponents
- **System 1 + System 2** — fast policy for routine spots, search for hard spots
- **Hybrid action space** — discrete action type + continuous bet sizing

## Project Structure

```
code-poker-bot/
├── engine/                 # Core poker engine
│   ├── game_state.py       # NLHE game state (2-9 players, 1-350bb, side pots)
│   ├── hand_evaluator.py   # 5-7 card hand ranking
│   ├── dealer.py           # Game loop (shuffle, deal, streets, showdown)
│   └── kuhn_poker.py       # Kuhn Poker (3-card game for validation)
├── model/                  # Neural network components (Phase 2)
│   ├── opponent_encoder.py
│   ├── stat_tracker.py
│   ├── policy_network.py
│   └── action_space.py
├── search/                 # Lightweight search (Phase 4)
│   ├── search.py
│   └── range_estimator.py
├── training/               # Training system
│   ├── cfr.py              # ✅ CFR solver (validated on Kuhn Poker)
│   ├── self_play_trainer.py
│   ├── personality.py
│   ├── trainer.py
│   ├── curriculum.py
│   └── rewards.py
├── agent/                  # Agent interface (Phase 2)
│   ├── poker_agent.py
│   └── config.py
├── tests/                  # Test suite (53 tests)
│   ├── test_engine.py
│   └── test_kuhn.py
├── docs/                   # Architecture Decision Records
│   └── adr/
└── scripts/                # Training & evaluation scripts
```

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# (Phase 2+) Train on Kuhn Poker
python scripts/train.py --game kuhn --episodes 100000

# (Phase 3+) Train on NLHE
python scripts/train.py --game nlhe --curriculum
```

## Development

### Running Tests
```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### Architecture Decisions
See [docs/adr/](docs/adr/) for recorded architecture decisions and their rationale.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **1. Engine** | ✅ Complete | NLHE rules, hand evaluator, dealer, Kuhn Poker + CFR validation |
| **2. Architecture** | 🔲 Next | Opponent encoder, policy network, self-play |
| **3. Perturbations** | 🔲 | Situational personality modifiers, NLHE training |
| **4. Search** | 🔲 | Lightweight real-time search (System 2) |
| **5. Evaluation** | 🔲 | Benchmarks, GTO verification |
| **6. Deployment** | 🔲 | Inference optimization |

## Key Concepts

### Opponent Modeling
The agent uses a **dual-signal** system:
- **Learned embeddings** — Transformer encodes raw action history into a latent vector
- **Explicit HUD stats** — ~30 interpretable features (VPIP, PFR, fold-to-cbet, etc.)

### Training Pipeline
1. **Self-play** — model copies play each other → converge toward GTO
2. **Perturbation** — apply personality modifiers (range_mult, aggression_mult, etc.)
3. **Situational** — per-context overrides (e.g., "tight preflop, loose on wet boards")

### Action Space
- **Action type**: `[fold, check, call, raise]` — 4-way classification
- **Bet sizing**: continuous value `[min_raise, all_in]` as pot fraction (when raising)
