#!/usr/bin/env bash
# Train AttentionProbe for mmlu_pro_10 (10-option).
# Usage: bash scripts/run_probe_mmlu_pro_10.sh [config_path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CFG_PATH="${1:-$REPO_ROOT/experiments/deepseek_r1_qwen_32b/probe_mmlu_pro_10.yaml}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

LOG_DIR="$REPO_ROOT/results/probe_mmlu_pro_10_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/probe.log"

echo "Running probe training with config: $CFG_PATH"
echo "Logging to: $LOG_FILE"

uv run python -m src.analysis.run_probe_mmlu_pro_10 "$CFG_PATH" 2>&1 | tee "$LOG_FILE"
