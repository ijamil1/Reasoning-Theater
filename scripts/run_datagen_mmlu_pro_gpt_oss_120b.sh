#!/usr/bin/env bash
# Data generation for GPT-OSS 120B across ONLY the MMLU Pro dataset.
# Runs Stage 1 (rollout collection) + Stage 2 (activation harvesting, final layer only).

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
FINAL_LAYER=35  # num_layers=36, 0-indexed final layer

DATASETS=(
    "${ROOT_DIR}/experiments/gpt_oss_120b/mmlu_pro_datagen.yaml"
)

for yaml in "${DATASETS[@]}"; do
    bash "${ROOT_DIR}/scripts/run_datagen.sh" "${yaml}" both --layer "${FINAL_LAYER}"
done
