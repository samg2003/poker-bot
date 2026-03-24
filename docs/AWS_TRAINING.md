# AWS Training & Deployment Guide

## Overview

Train locally on Leduc for quick validation, then scale to full NLHE on AWS GPU.

## Step 1: Quick Local Test (No GPU Needed)

```bash
cd code-poker-bot
source venv/bin/activate  # or python3 -m venv venv && source venv/bin/activate
python3 -m pytest tests/ -v  # make sure 168 tests pass

# Quick Leduc training (~5 min, CPU)
python3 scripts/train.py --game leduc --epochs 100 --embed-dim 64
```

## Step 2: Launch AWS GPU Instance

### Recommended Instances

| Training Phase | Instance | GPU | Cost/hr |
|---|---|---|---|
| Validation (Leduc) | `g5.xlarge` | 1× A10G (24GB) | ~$1.00 |
| Curriculum Stage 1-2 | `g5.2xlarge` | 1× A10G (24GB) | ~$1.21 |
| Full Training | `g5.12xlarge` | 4× A10G (96GB) | ~$5.67 |

### Launch Steps

```bash
# 1. Launch instance via AWS Console:
#    - AMI: "Deep Learning AMI (Ubuntu 22.04)" — comes with CUDA pre-installed
#    - Instance type: g5.2xlarge (start small)
#    - Storage: 100 GB
#    - Security group: allow SSH (port 22)
#    - Key pair: your .pem file

# 2. SSH in
ssh -i your-key.pem ubuntu@<instance-ip>

# 3. Upload your code
# From your LOCAL machine:
scp -i your-key.pem -r code-poker-bot ubuntu@<instance-ip>:/home/ubuntu/
```

## Step 3: Setup & Train on AWS

```bash
# On the EC2 instance:
cd /home/ubuntu/code-poker-bot
bash scripts/aws_setup.sh

# Start training in tmux (so it survives SSH disconnect)
tmux new -s train
source venv/bin/activate

# Phase 1: Quick validation (~10 min)
python3 scripts/train.py --game leduc --epochs 200 --embed-dim 128

# Phase 2: Full curriculum training (~2-8 hours depending on instance)
python3 scripts/train.py --curriculum --epochs 500

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t train
```

## Step 4: Download Trained Model

```bash
# From your LOCAL machine:
scp -i your-key.pem -r ubuntu@<instance-ip>:/home/ubuntu/code-poker-bot/checkpoints ./checkpoints
```

## Step 5: Use Trained Model Locally

```bash
# Evaluate the trained model
python3 scripts/evaluate.py --checkpoint best --benchmark-latency

# Use in code:
```

```python
from agent.poker_agent import PokerAgent
from agent.config import AgentConfig
from deployment.checkpoint import CheckpointManager

# Create agent
agent = PokerAgent.from_config(AgentConfig(embed_dim=128))

# Load trained weights
mgr = CheckpointManager('checkpoints')
mgr.load(agent.policy, agent.opponent_encoder, tag='best')

# Play!
result = agent.get_action(
    hole_cards=(48, 49),         # Ah, As (pocket aces)
    community_cards=[10, 20, 30],
    numeric_features=[0.5, 1.0, 0.0, 0.0, 0.33, 0.22, 0.22, 0.1, 0.02],
    opponent_ids=[1, 2],
)
print(f"Action: {result.action_type}, Sizing: {result.bet_sizing:.2f}")
print(f"Probs: fold={result.action_probs[0]:.2f} check={result.action_probs[1]:.2f} "
      f"call={result.action_probs[2]:.2f} raise={result.action_probs[3]:.2f}")
```

## Cost Estimate

| Phase | Instance | Duration | Cost |
|---|---|---|---|
| Validation (Leduc) | g5.xlarge | ~30 min | ~$0.50 |
| Curriculum Stage 1-2 | g5.2xlarge | ~3-5 hrs | ~$5-6 |
| Full Curriculum | g5.12xlarge | ~5-8 hrs | ~$30-45 |
| **Total** | | | **~$35-50** |

> **Tip**: Start with `g5.xlarge` to validate everything works, then scale up. Don't forget to **stop the instance** when done!

## Important: Don't Forget

- [ ] **Stop instance** after training: `aws ec2 stop-instances --instance-ids i-xxx`
- [ ] Download checkpoints before stopping
- [ ] Verify loaded model locally with `scripts/evaluate.py`
