#!/bin/bash
# Sync latest checkpoint from EC2 to local Mac.
# Run from project root: ./scripts/sync-checkpoint.sh
#
# Downloads: checkpoints/latest/ (policy.pt, opponent_encoder.pt, metadata.json, versions.json, pool/)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Defaults ---
DEFAULT_PEM="$PROJECT_DIR/carte6.pem"
DEFAULT_IP="54.219.184.157"
DEFAULT_USER="ubuntu"

# --- Config (press Enter for defaults) ---
read -p "EC2 IP address [$DEFAULT_IP]: " EC2_IP
EC2_IP=${EC2_IP:-$DEFAULT_IP}

read -p "PEM file [$DEFAULT_PEM]: " PEM_PATH
PEM_PATH=${PEM_PATH:-$DEFAULT_PEM}

EC2_USER="$DEFAULT_USER"
REMOTE_DIR="~/poker-bot/checkpoints/latest/"
LOCAL_DIR="$PROJECT_DIR/checkpoints/latest/"

# Strip quotes
EC2_IP=$(echo "$EC2_IP" | tr -d "'" | tr -d '"')
PEM_PATH=$(echo "$PEM_PATH" | tr -d "'" | tr -d '"')

# Fix permissions
chmod 400 "$PEM_PATH"

echo "EC2 IP address: $EC2_IP"
echo "============================================"
echo "Syncing checkpoint from EC2..."
echo "  From: $EC2_USER@$EC2_IP:$REMOTE_DIR"
echo "  To:   $LOCAL_DIR"
echo "============================================"

# Create local dir
mkdir -p "$LOCAL_DIR"

# Sync with rsync (only downloads changed files)
rsync -avz --progress \
    -e "ssh -i $PEM_PATH -o StrictHostKeyChecking=no" \
    "$EC2_USER@$EC2_IP:$REMOTE_DIR" \
    "$LOCAL_DIR"

echo ""
echo "✓ Checkpoint synced!"

# Sync logs
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
echo ""
echo "Syncing logs..."
rsync -avz --progress \
    -e "ssh -i $PEM_PATH -o StrictHostKeyChecking=no" \
    "$EC2_USER@$EC2_IP:~/poker-bot/eval_history.log" \
    "$EC2_USER@$EC2_IP:~/poker-bot/train.log" \
    "$LOG_DIR/" 2>/dev/null || echo "  (some logs not found yet)"
echo "✓ Logs synced to $LOG_DIR/"

# Show what we got
if [ -f "$LOCAL_DIR/metadata.json" ]; then
    echo ""
    echo "Checkpoint metadata:"
    cat "$LOCAL_DIR/metadata.json" | python3 -m json.tool 2>/dev/null || cat "$LOCAL_DIR/metadata.json"
fi

echo ""
echo "You can now run:"
echo "  python3 scripts/train.py --game nlhe --resume latest  (to view)"
echo "  cd poker_ui/backend && python3 app.py                 (to play)"
