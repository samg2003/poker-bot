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
│   ├── nlhe_trainer.py     # ✅ Full NLHE self-play (batched GPU inference, opponent tracking)
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
├── tests/                  # Test suite (185 tests)
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
│   ├── connect-ec2.sh      # ✅ Interactive SSH connection to EC2
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

# Run tests (185 tests)
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
# Universal training — randomizes players (2-9) & stacks (10-300bb) per hand
# Auto-detects GPU (CUDA/MPS), personality curriculum kicks in at epoch 10
python3 scripts/train.py --game nlhe --epochs 500

# With search-guided expert iteration (10% of hands use System 2 CFR)
python3 scripts/train.py --game nlhe --epochs 500 --search-fraction 0.1

# Fixed heads-up, 100bb deep
python3 scripts/train.py --game nlhe --epochs 500 --num-players 2 --starting-bb 100

# Full config for GPU instance (e.g., g5.xlarge with CUDA)
python3 scripts/train.py --game nlhe --embed-dim 256 --num-layers 4 --num-heads 4 --hands 2048 --epochs 500 --min-players 2 --max-players 9 --min-bb 10 --max-bb 300 --device cuda --verbose --save-interval 50

python3 scripts/train.py --game nlhe --embed-dim 256 --num-layers 4 --num-heads 4 --hands 2048 --epochs 500 --min-players 2 --max-players 9 --min-bb 10 --max-bb 300 --verbose --save-interval 50

python3 scripts/train.py --game nlhe --embed-dim 512 --num-layers 8 --num-heads 8 --hands 3000 --epochs 500 --min-players 2 --max-players 9 --min-bb 10 --max-bb 300 --verbose --save-interval 50 --batch-chunk-size 2000
```

### Full CLI Options
```
--game {leduc,nlhe}     Game to train on (default: leduc)
--curriculum            Use curriculum training (Leduc only)
--epochs N              Number of epochs (default: 100)
--hands N               Hands per epoch (default: 128)
--embed-dim N           Model embedding dimension (default: 64)
--num-heads N           Attention heads (default: 2)
--num-layers N          Transformer layers (default: 2)
--num-players N         Fixed player count, 0=random (default: 0)
--starting-bb N         Fixed stack in BB, 0=random (default: 0)
--min-players N         Min players when random (default: 2)
--max-players N         Max players when random (default: 6)
--min-bb N              Min stack in BB when random (default: 20)
--max-bb N              Max stack in BB when random (default: 200)
--device STR            Device: auto, cuda, mps, cpu (default: auto)
--threads N             Number of CPU threads to use; 0=all (default: 0)
--verbose               Enable verbose output with timing and progress updates
--search-fraction F     Fraction of hands using search (default: 0)
--batch-chunk-size N    Max simultaneous games per sub-batch (default: 500)
--save-interval N       Save checkpoint every N epochs (default: 0 = end only)
--resume TAG            Resume training from checkpoint tag
--checkpoint-dir DIR    Where to save checkpoints (default: checkpoints/)
--lr FLOAT              Learning rate (default: 3e-4)
--seed N                Random seed (default: 42)
```

## Evaluation

```bash
# Run all benchmarks on the default Leduc game
python3 scripts/evaluate.py

# Run all benchmarks on full No-Limit Texas Hold'em (NLHE)
python3 scripts/evaluate.py --game nlhe

# Run evaluation with more hands per benchmark (slower but more accurate)
python3 scripts/evaluate.py --game nlhe --num-hands 2000

# Evaluate a specific checkpoint (architecture and game type auto-loaded)
python3 scripts/evaluate.py --checkpoint latest
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
2. **Opponent tracking** — action history + HUD stats feed real embeddings during training
3. **Perturbation** — personality modifiers create diverse opponents
4. **Expert iteration** — optional search-guided training refines the policy
5. **History reset** — periodic reset (300-500 hands) ensures model handles unknown opponents

### Action Space
- **Action type**: `[fold, check, call, raise]` — 4-way classification
- **Bet sizing**: continuous value `[min_raise, all_in]` as pot fraction (when raising)
