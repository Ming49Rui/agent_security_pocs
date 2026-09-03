#!/bin/bash
# Run all models in parallel for a given mode and trial count.
# Usage: bash benchmark/run_all_parallel.sh <mode> <trials>
# Example: bash benchmark/run_all_parallel.sh active 5
#          bash benchmark/run_all_parallel.sh active 5 --resume

MODE=${1:-active}
TRIALS=${2:-5}
RESUME=${3:-}

MODELS=("Llama" "Kimi" "Qwen3" "Claude Sonnet" "Claude Opus" "Devstral" "GPT-4o" "Gemini")

echo "========================================"
echo "LogJack Parallel Runner"
echo "Mode: $MODE | Trials: $TRIALS | Resume: ${RESUME:-no}"
echo "Models: ${#MODELS[@]}"
echo "========================================"

PIDS=()
for model in "${MODELS[@]}"; do
    tag=$(echo "$model" | tr ' ' '_')
    log="benchmark/logs/${MODE}_${tag}.log"
    mkdir -p benchmark/logs
    nohup python3 benchmark/run_unified.py "$MODE" "$model" --trials "$TRIALS" $RESUME > "$log" 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "Started $model — PID $pid — log: $log"
done

echo ""
echo "Monitor: tail -f benchmark/logs/${MODE}_*.log"
echo "Status:  bash benchmark/check_status.sh $MODE"
echo ""

# Save PIDs for status checking
printf "%s\n" "${PIDS[@]}" > "benchmark/logs/${MODE}_pids.txt"
