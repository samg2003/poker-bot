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
│   ├── game_state.py       # ✅ NLHE game state (2-9 players, 1-350bb, side pots)
│   ├── hand_evaluator.py   # ✅ 5-7 card hand ranking
│   ├── dealer.py           # ✅ Game loop (shuffle, deal, streets, showdown)
│   ├── kuhn_poker.py       # ✅ Kuhn Poker (3-card, Nash validated)
│   └── leduc_poker.py      # ✅ Leduc Hold'em (6-card, ~1000 info sets)
├── model/                  # Neural network components
│   ├── action_space.py     # ✅ Hybrid action type + continuous sizing
│   ├── stat_tracker.py     # ✅ ~30 HUD features per opponent
│   ├── opponent_encoder.py # ✅ Causal Transformer (history → embedding)
│   ├── policy_network.py   # ✅ Cross-attention policy + value + sizing heads
│   └── nlhe_encoder.py     # ✅ GameState → model tensor bridge
├── search/                 # Lightweight search (System 2)
│   ├── search.py           # ✅ Subtree CFR with policy leaf evaluation
│   └── range_estimator.py  # ✅ Neural range estimation (1326 combos)
├── training/               # Training system
│   ├── cfr.py              # ✅ CFR solver (validated on Kuhn Poker)
│   ├── self_play_trainer.py # ✅ PPO self-play on Leduc Hold'em
│   ├── personality.py      # ✅ Continuous personality perturbations + tilt
│   └── curriculum.py       # ✅ Multi-stage curriculum trainer
├── agent/                  # Agent interface
│   ├── poker_agent.py      # ✅ Unified System 1+2 inference
│   └── config.py           # ✅ Central configuration
├── evaluation/             # Benchmarks
│   └── evaluator.py        # ✅ 6 automated benchmarks
├── deployment/             # Production deployment
│   ├── checkpoint.py       # ✅ Model save/load/versioning
│   └── inference.py        # ✅ Optimized inference + benchmarking
├── tests/                  # Test suite (168 tests)
│   ├── test_engine.py
│   ├── test_kuhn.py
│   ├── test_model.py
│   ├── test_self_play.py
│   ├── test_personality.py
│   ├── test_search.py
│   ├── test_evaluation.py
│   ├── test_deployment.py
│   └── test_nlhe_trainer.py
├── scripts/                # CLI scripts
│   ├── train.py            # ✅ Training CLI (Leduc + NLHE)
│   ├── evaluate.py         # ✅ Evaluation CLI + latency benchmark
│   └── aws_setup.sh        # ✅ AWS GPU instance setup
├── docs/                   # Architecture Decision Records
│   └── adr/
└── requirements.txt
```

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests (175 tests)
python3 -m pytest tests/ -v
```

## Training

### Leduc Hold'em (local, no GPU needed)
```bash
# Quick validation (~5 min on CPU)
python3 scripts/train.py --game leduc --epochs 200 --embed-dim 64 --num-heads 2 --num-layers 2

# With curriculum (personality perturbations)
python3 scripts/train.py --curriculum --epochs 500
```

### No-Limit Hold'em (GPU recommended)
```bash
# Heads-up, 100bb deep
python3 scripts/train.py --game nlhe --epochs 500 --num-players 2 --starting-bb 100

# 6-max, 100bb deep
python3 scripts/train.py --game nlhe --epochs 500 --num-players 6 --starting-bb 100

# Short-stacked
python3 scripts/train.py --game nlhe --epochs 500 --num-players 2 --starting-bb 20
```

### Full CLI Options
```
--game {leduc,nlhe}     Game to train on (default: leduc)
--curriculum            Use curriculum training (Leduc only)
--epochs N              Number of epochs (default: 100)
--hands N               Hands per epoch (default: 512)
--embed-dim N           Model embedding dimension (default: 128)
--num-heads N           Attention heads (default: 4)
--num-layers N          Transformer layers (default: 3)
--num-players N         Players at table, NLHE only (default: 2)
--starting-bb N         Starting stack in BB, NLHE only (default: 100)
--checkpoint-dir DIR    Where to save checkpoints (default: checkpoints/)
--lr FLOAT              Learning rate (default: 3e-4)
--seed N                Random seed (default: 42)
```

## Evaluation

```bash
# Run benchmarks (untrained model)
python3 scripts/evaluate.py

# With trained checkpoint + latency benchmark
python3 scripts/evaluate.py --checkpoint best --benchmark-latency
```

## Development

### Running Tests
```bash
python3 -m pytest tests/ -v
```

### Architecture Decisions
See [docs/adr/](docs/adr/) for recorded architecture decisions and their rationale.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **1. Engine** | ✅ Complete | NLHE rules, hand evaluator, dealer, Kuhn + Leduc games |
| **2. Architecture** | ✅ Complete | Opponent encoder, policy network, PPO self-play on Leduc |
| **3. Perturbations** | ✅ Complete | Personality system, NLHE encoder, curriculum training |
| **4. Search** | ✅ Complete | Subtree CFR, range estimation, search triggering |
| **5. Evaluation** | ✅ Complete | Agent interface, 6 automated benchmarks |
| **6. Deployment** | ✅ Complete | Checkpointing, optimized inference, CLI scripts |

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
