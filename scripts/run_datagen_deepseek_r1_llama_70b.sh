#!/usr/bin/env bash
# Data generation for DeepSeek-R1-Distill-Llama-70B across all 5 datasets.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).
# Usage: bash scripts/run_datagen_deepseek_r1_llama_70b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=79  # num_layers=80, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/deepseek_r1_llama_70b/arc_easy_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_70b/mmlu_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_70b/arc_challenge_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_70b/medqa_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_70b/gpqa_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done
