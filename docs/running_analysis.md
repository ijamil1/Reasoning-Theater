# Running Analysis

## Recommended workflow before running the pipeline

The full analysis pipeline (`run_pipeline.sh`) requires trained probe checkpoints.
Before running it, complete **Phase 4** first — it informs which probe and training
dataset to use everywhere downstream.

### Step 1 — Phase 4: decide the probe strategy per dataset

Phase 4 trains one probe per dataset (using the best architecture from Phase 3) and
evaluates every probe on every dataset's test split, producing:

- **Raw accuracy matrix** — `results/phase4_summary.csv`
- **Degradation matrix** — `results/phase4_summary_degradation.csv`
  (`degradation[A][B] = accuracy[A][B] − accuracy[A][A]`, diagonal = 0,
   off-diagonals typically negative)
- Heatmap plots in `results/phase4_plots/`

```bash
bash scripts/run_phase4.sh experiments/deepseek_r1_qwen_32b/phase4.yaml
bash scripts/run_phase4.sh experiments/gpt_oss_120b/phase4.yaml
```

**Reading the degradation matrix:** Each row corresponds to a probe trained on
dataset A. The off-diagonal cells show how much accuracy that probe loses when
applied to dataset B. A row with small off-diagonal values means the probe trained
on A generalises well — it has learned something about how the model encodes answers
in general, not just on A's distribution.

**Decision rule for the pipeline:**

1. Find the row in the degradation matrix with the smallest mean off-diagonal
   degradation. Call this the *most generalizable training dataset* (G).
2. For each dataset D in the pipeline:
   - Always include a run with `probe.train: true` using D's own training data
     (the *in-domain probe*). Exception: GPQA-Diamond (only 198 questions —
     use `[0.0, 0.0, 1.0]` split ratio and skip training).
   - Also include a second run with `probe.train: false` and
     `reuse_run_root` pointing to G's Phase 4 training run
     (the *generalizable probe*).
   - Running both lets you compare performativity gap and temporal belief
     tracking under each probe in Phase 6/7, controlling for whether the
     probe's training distribution matches the evaluation distribution.

### Step 2 — Run the analysis pipeline

Once probe checkpoints exist (from Phase 4's training runs), run the pipeline for
each dataset. Each dataset gets at minimum two analysis runs: one with the in-domain
probe and one with the generalizable probe.

```bash
# Full model sweep (all datasets, both probe variants)
bash scripts/run_analysis_deepseek_r1_qwen_32b.sh
bash scripts/run_analysis_gpt_oss_120b.sh

# Or a single dataset/probe combination
bash scripts/run_pipeline.sh experiments/deepseek_r1_qwen_32b/mmlu_analysis.yaml
```

---

## Quick start (if phase 4 is already done)

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

Use this when you do not want to train a probe on a dataset's activations — for
example, GPQA (too few questions) or when applying the generalizable probe from
Phase 4 to a different dataset.

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
  # For the generalizable probe, use the Phase 4 training run name, e.g.:
  reuse_run_root: results/phase4_qwen_32b_train_mmlu

  # Option B — provide an explicit path to the .pth file (overrides reuse_run_root).
  # checkpoint: results/models/phase4_qwen_32b_train_mmlu/probe_layer63.pth
```

Use **Option A** (`reuse_run_root`) when reusing a probe from another dataset run of
the same model — the run name is the directory name under `results/`.

Use **Option B** (`checkpoint`) when you have a specific checkpoint file at an
arbitrary path (e.g. downloaded from cloud storage).

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

Phase 4 training runs save their checkpoints under run names of the form:

```
results/models/phase4_<model_prefix>_train_<dataset>/probe_layer<N>.pth
```

For example, the layer-63 probe trained on MMLU for the 32B model during Phase 4:

```
results/models/phase4_qwen_32b_train_mmlu/probe_layer63.pth
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

| Dataset | Train in-domain probe? | Generalizable probe source | Reason |
|---|---|---|---|
| MMLU | Yes | From Phase 4 best-generalizing row | Primary training dataset (~5330 questions) |
| ARC-Challenge | Yes | From Phase 4 best-generalizing row | Sufficient volume |
| MedQA | Yes | From Phase 4 best-generalizing row | Sufficient volume (2000 sampled) |
| GPQA-Diamond | **No** | From Phase 4 best-generalizing row | Only 198 questions — all go to test set |

For GPQA-Diamond, set `train_val_test_split_ratio: [0.0, 0.0, 1.0]` and point
`reuse_run_root` at the Phase 4 run for whichever dataset generalises best.
