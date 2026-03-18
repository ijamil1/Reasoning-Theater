#!/usr/bin/env bash
# Data generation for DeepSeek-R1-Distill-Llama-8B across all 5 datasets.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).
# Usage: bash scripts/run_datagen_deepseek_r1_llama_8b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=31  # num_layers=32, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/deepseek_r1_llama_8b/arc_easy_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_8b/mmlu_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_8b/arc_challenge_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_8b/medqa_datagen.yaml"
    "${ROOT_DIR}/experiments/deepseek_r1_llama_8b/gpqa_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done
