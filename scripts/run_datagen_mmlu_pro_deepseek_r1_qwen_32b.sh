#!/usr/bin/env bash
# Data generation for DeepSeek-R1-Distill-Qwen-32B across only MMLU Pro.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=63  # num_layers=64, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/mmlu_pro_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done

    
