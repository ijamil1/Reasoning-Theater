#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_phase4.sh experiments/deepseek_r1_qwen_32b/phase4.yaml

CFG_PATH=$(realpath "${1:?Provide path to phase4.yaml}")

ROOT_DIR="$(cd "$(dirname "${CFG_PATH}")" && git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

LOG_DIR="${ROOT_DIR}/results/phase4_logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/phase4.log"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "===== PHASE 4 START ====="
echo "Host:   $(hostname)"
echo "Config: ${CFG_PATH}"
echo "Log:    ${LOG_PATH}"

uv run python -m src.analysis.run_phase4 "${CFG_PATH}"

echo "===== PHASE 4 DONE ====="
