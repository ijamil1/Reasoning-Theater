#!/usr/bin/env bash
# Analysis pipeline for DeepSeek-R1-Distill-Qwen-32B across all 4 datasets.
# Runs setup, probing, forced answering, CoT monitor, and plots for each dataset.
# Usage: bash scripts/run_analysis_deepseek_r1_qwen_32b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

DATASETS=(
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/mmlu_analysis.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/arc_challenge_analysis.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/medqa_analysis.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/gpqa_analysis.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/mmlu_pro_analysis.yaml"
)

for yaml in "${DATASETS[@]}"; do
    echo "===== Running analysis for ${yaml} ====="
    bash "${ROOT_DIR}/scripts/run_pipeline.sh" "${yaml}"
done
