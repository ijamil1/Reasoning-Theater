#!/usr/bin/env bash
# Runs test_single_dataset_loading.sh once per model, each with a distinct dataset.
# Usage: bash dataset_prep/test_dataset_loading.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"

# One distinct dataset per model
YAMLS=(
    "${ROOT_DIR}/experiments/deepseek_r1_qwen_32b/mmlu_datagen.yaml"
    "${ROOT_DIR}/experiments/gpt_oss_120b/gpqa_datagen.yaml"
)

for yaml in "${YAMLS[@]}"; do
    bash "${SCRIPT_DIR}/test_single_dataset_loading.sh" "${yaml}"
done