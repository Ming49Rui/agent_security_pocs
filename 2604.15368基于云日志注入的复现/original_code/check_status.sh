#!/bin/bash
# Check status of parallel runs.
# Usage: bash benchmark/check_status.sh [mode]

MODE=${1:-active}

echo "=== $MODE ==="
for f in benchmark/logs/${MODE}_*.log; do
    [ ! -f "$f" ] && continue
    model=$(basename "$f" .log | sed "s/${MODE}_//")
    saved=$(grep "Results saved" "$f" 2>/dev/null)
    fatal=$(grep "FATAL" "$f" 2>/dev/null)
    done=$(grep -c "\.\.\." "$f" 2>/dev/null || echo 0)
    if [ -n "$fatal" ]; then
        echo "  ❌ $model: $done FATAL — $(grep FATAL "$f" | head -c 60)"
    elif [ -n "$saved" ]; then
        echo "  ✅ $model: Done"
    else
        echo "  🔄 $model: $done payloads"
    fi
done

echo ""
echo "Processes: $(ps aux | grep run_unified | grep -v grep | wc -l | tr -d ' ') running"
echo "Errors: $(grep -l 'FATAL\|ThrottlingException\|429.*Resource exhausted' benchmark/logs/${MODE}_*.log 2>/dev/null | wc -l | tr -d ' ') files with errors"
