#!/bin/bash
# Sync latest checkpoint from EC2 to local Mac.
# Run from project root: ./scripts/sync-checkpoint.sh
#
# Downloads: checkpoints/latest/ (policy.pt, opponent_encoder.pt, metadata.json, versions.json, pool/)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Config ---
PEM_PATH="${PEM_PATH:-$PROJECT_DIR/carte6.pem}"
EC2_USER="${EC2_USER:-ubuntu}"
REMOTE_DIR="~/poker-bot/checkpoints/latest/"
LOCAL_DIR="$PROJECT_DIR/checkpoints/latest/"

# Get EC2 IP
if [ -z "$EC2_IP" ]; then
    read -p "EC2 IP address: " EC2_IP
fi

# Strip quotes
EC2_IP=$(echo "$EC2_IP" | tr -d "'" | tr -d '"')
PEM_PATH=$(echo "$PEM_PATH" | tr -d "'" | tr -d '"')

# Fix permissions
chmod 400 "$PEM_PATH"

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
