#!/bin/bash
# ============================================================
# AWS GPU Training Setup Script
#
# Run this ON the EC2 instance after SSHing in.
# It installs everything and starts training.
# ============================================================

set -e

echo "=== Poker AI — AWS GPU Setup ==="

# 1. System packages
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git tmux htop

# 2. Clone your repo (or upload via scp)
# Option A: From GitHub
# git clone https://github.com/YOUR_USER/code-poker-bot.git
# cd code-poker-bot

# Option B: Already uploaded via scp (see README)
cd /home/ubuntu/code-poker-bot

# 3. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies (GPU-accelerated PyTorch)
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pytest

# 5. Verify GPU
python3 -c "import torch; print(f'GPU available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 6. Run tests to make sure everything works
python3 -m pytest tests/ -v --tb=short

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start training:"
echo "  tmux new -s train"
echo "  source venv/bin/activate"
echo "  python3 scripts/train.py --game leduc --epochs 200    # quick test"
echo "  python3 scripts/train.py --curriculum --epochs 500     # full training"
echo ""
echo "To detach from tmux: Ctrl+B then D"
echo "To reattach: tmux attach -t train"
