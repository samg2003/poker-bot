#!/bin/bash
# Auto-evaluate new checkpoints as they appear.
# Run in a separate tmux pane on EC2:
#   tmux new -s eval './scripts/auto-eval.sh'

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/latest"
METADATA="$CHECKPOINT_DIR/metadata.json"
EVAL_LOG="$PROJECT_DIR/eval_history.log"
LAST_EPOCH=""

echo "🔍 Watching for new checkpoints..."
echo "   Log: $EVAL_LOG"
echo ""

while true; do
    if [ -f "$METADATA" ]; then
        CURRENT_EPOCH=$(python3 -c "import json; print(json.load(open('$METADATA'))['epoch'])" 2>/dev/null || echo "")

        if [ -n "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" != "$LAST_EPOCH" ]; then
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📦 New checkpoint detected: epoch $CURRENT_EPOCH"
            echo "   Running evaluation (2000 hands)..."
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            # Run eval on CPU to avoid competing with training for GPU
            # Use tee to show output live AND append to log
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "" >> "$EVAL_LOG"
            echo "=== Epoch $CURRENT_EPOCH | $TIMESTAMP ===" >> "$EVAL_LOG"
            (cd "$PROJECT_DIR" && CUDA_VISIBLE_DEVICES= python3 scripts/evaluate.py --checkpoint latest --verbose --num-hands 1000 2>&1) | tee -a "$EVAL_LOG" || true

            echo ""
            echo "✅ Logged to $EVAL_LOG"
            echo ""

            LAST_EPOCH="$CURRENT_EPOCH"
        fi
    fi

    sleep 30
done
