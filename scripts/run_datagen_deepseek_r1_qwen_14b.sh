#!/usr/bin/env bash
# Data generation for DeepSeek-R1-Distill-Qwen-14B across all 5 datasets.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).
# Usage: bash scripts/run_datagen_deepseek_r1_qwen_14b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=47  # num_layers=48, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_14b/arc_easy_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_14b/mmlu_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_14b/arc_challenge_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_14b/medqa_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_14b/gpqa_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done
