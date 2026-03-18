#!/usr/bin/env bash
# Data generation for DeepSeek-R1-Distill-Qwen-32B across all 4 datasets.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).
# Usage: bash scripts/run_datagen_deepseek_r1_qwen_32b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=63  # num_layers=64, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/mmlu_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/arc_challenge_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/medqa_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/gpqa_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done
