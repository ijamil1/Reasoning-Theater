#!/usr/bin/env bash
# Usage: bash dataset_prep/test_single_dataset_loading.sh <path-to-datagen.yaml>
set -euo pipefail

CFG_PATH=$(realpath "${1:?Provide path to datagen YAML}")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT_DIR="$(cd "$(dirname "${CFG_PATH}")" && git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo ""
echo "===== Testing dataset loading ====="
echo "Config: ${CFG_PATH}"

uv run --project "${SCRIPT_DIR}" python -c "
import yaml, sys, textwrap
from pathlib import Path
from src.data_generation.data_gen_config import DataGenerationConfig
from src.data_generation.datasets import load_dataset_questions

cfg = yaml.safe_load(open(sys.argv[1]))
dg = cfg['data_generation']

config = DataGenerationConfig(
    model_id=dg.get('model_id') or '',
    dataset_name=dg['dataset_name'],
    output_dir=Path(dg['output_dir']),
    limit=int(dg['limit']) if dg.get('limit') else None,
    existing_responses_dir=Path(dg['existing_responses_dir']) if dg.get('existing_responses_dir') else None,
)

questions = load_dataset_questions(config)
print(f'Total questions loaded: {len(questions)}')
print()

for i, q in enumerate(questions[:10]):
    print(f'--- Question {i + 1} ---')
    print(f'Hash    : {q.question_hash}')
    print(f'Category: {q.category}')
    print(f'Answer  : {q.correct_answer}')
    print(f'Question: {textwrap.shorten(q.question, width=120)}')
    for label, choice in zip([\"A\",\"B\",\"C\",\"D\"], q.choices):
        print(f'  ({label}) {textwrap.shorten(choice, width=100)}')
    print()
" "${CFG_PATH}"

echo "===== Done ====="