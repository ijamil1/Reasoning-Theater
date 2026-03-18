#!/usr/bin/env bash
# Data generation for GPT-OSS 120B across all 5 datasets.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).
# Usage: bash scripts/run_datagen_gpt_oss_120b.sh
set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=35  # num_layers=36, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/gpt_oss_120b/arc_easy_datagen.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/mmlu_datagen.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/arc_challenge_datagen.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/medqa_datagen.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/gpqa_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done
