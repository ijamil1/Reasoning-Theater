#!/usr/bin/env bash
# Run HP grid search for the mmlu_pro_10 (10-option) probe.
# Usage: bash scripts/run_hpsearch_mmlu_pro_10.sh [config_path]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CFG_PATH="${1:-$REPO_ROOT/experiments/deepseek_r1_qwen_32b/hpsearch_mmlu_pro_10.yaml}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

LOG_DIR="$REPO_ROOT/results/hpsearch_mmlu_pro_10_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/hpsearch.log"

echo "Running HP search with config: $CFG_PATH"
echo "Logging to: $LOG_FILE"

uv run python -m src.analysis.run_hpsearch_mmlu_pro_10 "$CFG_PATH" 2>&1 | tee "$LOG_FILE"
