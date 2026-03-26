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

            # Run eval and capture output
            EVAL_OUTPUT=$(cd "$PROJECT_DIR" && python3 scripts/evaluate.py --checkpoint latest --num-hands 2000 2>&1) || true

            # Log with timestamp
            TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
            echo "" >> "$EVAL_LOG"
            echo "=== Epoch $CURRENT_EPOCH | $TIMESTAMP ===" >> "$EVAL_LOG"
            echo "$EVAL_OUTPUT" >> "$EVAL_LOG"

            # Print summary
            echo "$EVAL_OUTPUT"
            echo ""
            echo "✅ Logged to $EVAL_LOG"
            echo ""

            LAST_EPOCH="$CURRENT_EPOCH"
        fi
    fi

    sleep 30
done
