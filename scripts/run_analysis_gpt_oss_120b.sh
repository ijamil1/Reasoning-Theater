#!/usr/bin/env bash
# Analysis pipeline for GPT-OSS 120B across all 4 datasets.
# Runs setup, probing, forced answering, CoT monitor, and plots for each dataset.
# Usage: bash scripts/run_analysis_gpt_oss_120b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

DATASETS=(
    "${ROOT_DIR}/experiments/gpt_oss_120b/mmlu_analysis.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/arc_challenge_analysis.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/medqa_analysis.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/gpqa_analysis.yaml"
)

for yaml in "${DATASETS[@]}"; do
    echo "===== Running analysis for ${yaml} ====="
    bash "${ROOT_DIR}/scripts/run_pipeline.sh" "${yaml}"
done
