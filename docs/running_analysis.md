# Running Analysis

## Quick start

Run the full analysis pipeline for a model across all 4 datasets:

```bash
bash scripts/run_analysis_deepseek_r1_qwen_32b.sh
bash scripts/run_analysis_gpt_oss_120b.sh
```

Or run a single dataset:

```bash
bash scripts/run_pipeline.sh experiments/deepseek_r1_qwen_32b/mmlu_analysis.yaml
```

---

## Probe modes

Each analysis YAML controls probe behaviour via two independent flags:

| `probe.train` | `probe.eval` | Effect |
|---|---|---|
| `true` | `true` | Train a new probe on this dataset's activations, then evaluate on the test set |
| `false` | `true` | Load an existing probe checkpoint, evaluate on the test set (no training) |
| `true` | `false` | Train and save checkpoint only — no step-level CSV output |
| `false` | `false` | Pointless — skip by setting `probe.enabled: false` instead |

---

## Eval-only mode (load a pre-trained probe)

Use this when you do not want to train a probe on a dataset's activations — for example, GPQA (too few questions) or when running the generalizability transfer matrix.

### What to change in the YAML

```yaml
probe:
  enabled: true
  train: false      # do not train
  eval: true        # run inference and write step-level CSVs
  num_layers: 64    # must match the model (unchanged)
  selected_layer: 63

  # Option A — point at the run whose probe you want to reuse.
  # The checkpoint is resolved as:
  #   results/models/<reuse_run_name>/probe_layer<N>.pth
  reuse_run_root: results/mmlu_deepseek_r1_qwen_32b

  # Option B — provide an explicit path to the .pth file (overrides reuse_run_root).
  # checkpoint: results/models/mmlu_deepseek_r1_qwen_32b/probe_layer63.pth
```

Use **Option A** (`reuse_run_root`) when reusing a probe from another dataset run of the
same model — the run name is the directory name under `results/`.

Use **Option B** (`checkpoint`) when you have a specific checkpoint file at an arbitrary
path (e.g. downloaded from cloud storage).

### Also set the split ratio (GPQA case)

For datasets where you want all questions evaluated (none held out for training):

```yaml
setup:
  train_val_test_split_ratio: [0.0, 0.0, 1.0]
```

---

## Probe checkpoint locations

Probes are saved to and loaded from:

```
results/models/<run_name>/probe_layer<N>.pth
```

For example, the layer-63 probe trained on MMLU for the 32B model is at:

```
results/models/mmlu_deepseek_r1_qwen_32b/probe_layer63.pth
```

This directory is shared across runs so probes trained on one dataset can be loaded
by another dataset's analysis run without copying files.

---

## CoT monitor API key

The CoT monitor calls an external LLM via OpenRouter. Before running the pipeline,
ensure your API key is set in `.env` at the repo root:

```bash
OPENROUTER_API_KEY=sk-or-...
```

The key name is controlled by `cot_monitor.api_key_env` in the YAML (defaults to
`OPENROUTER_API_KEY`). The pipeline will raise at runtime if the variable is missing
or empty.

---

## Per-dataset notes

| Dataset | Train probe? | Reason |
|---|---|---|
| MMLU | Yes | Primary training dataset (~5330 questions) |
| ARC-Challenge | Yes | Sufficient volume |
| MedQA | Yes | Sufficient volume (2000 sampled) |
| GPQA-Diamond | **No** | Only 198 questions — all go to test set; load probe from another run |
