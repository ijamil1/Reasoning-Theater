# Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought

This is a fork of the codebase for the paper ["Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"](https://arxiv.org/abs/2603.05488). We are replicating and verifying the original results, and extending the work along three axes: a five-dataset difficulty ladder, a probe generalizability analysis across datasets, and a probe architecture comparison.

**Original paper:** https://arxiv.org/abs/2603.05488
**Original interactive app:** https://reasoning-theater.streamlit.app/

The original experiments were run on DeepSeek-R1-0528-671B and GPT-OSS-120B across MMLU-Redux-2.0 and GPQA-Diamond.

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env  # add your OPENROUTER_API_KEY if running the CoT monitor / inflection stages
```

The project has two dependency surfaces:
- `pyproject.toml` — full project (training, inference, analysis). Managed by `uv`.
- `streamlit_app/requirements.txt` — lightweight dependencies for the Streamlit app.

## Usage

Everything is configured via YAML files in `experiments/`. See `experiments/example_datagen.yaml` and `experiments/example_analysis.yaml` for the full schema.

The pipeline has two phases:
- **Data generation**: 1) collect model responses via vLLM and 2) harvest hidden-state activations via nnsight.
- **Analysis**: 1) train probes on activations, 2) collect forced-answering predictions, 3) collect CoT monitor predictions, and 4) identify and analyze inflection points.

### Example 1: Our experimental setup (per-model runner scripts)

Each model runs on its own machine. Per-model runner scripts in `scripts/` loop over all five dataset configs for that model and call `run_datagen.sh both --layer <final_layer>`, harvesting only the final layer's activations.

```
experiments/
  deepseek_r1_llama_8b/      # one yaml per dataset
    arc_easy_datagen.yaml
    mmlu_datagen.yaml
    arc_challenge_datagen.yaml
    medqa_datagen.yaml
    gpqa_datagen.yaml
  deepseek_r1_qwen_14b/
  deepseek_r1_qwen_32b/
  deepseek_r1_llama_70b/
  gpt_oss_120b/
```

#### 1. Install dependencies

```bash
uv venv && uv sync
```

#### 2. Login to HuggingFace

```bash
# Login with your token
hf auth login

# Paste your token when prompted
# Token will be saved to ~/.cache/huggingface/token
```

#### 3. Create directories for model weight storage

Check available disk space, then create the cache directories vLLM will use when downloading model weights:

```bash
df -h

mkdir -p /workspace/hf_cache
mkdir -p /workspace/hf_cache/hub
mkdir -p /workspace/hf_cache/transformers
mkdir -p /workspace/tmp

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers
export TMPDIR=/workspace/tmp
```

#### 4. Run data generation

To run data generation for a given model, execute its runner script on the appropriate machine:

```bash
# DeepSeek-R1-Distill-Llama-8B (1x A100 80GB)
bash scripts/run_datagen_deepseek_r1_llama_8b.sh

# DeepSeek-R1-Distill-Qwen-14B (1x A100 80GB)
bash scripts/run_datagen_deepseek_r1_qwen_14b.sh

# DeepSeek-R1-Distill-Qwen-32B (2x A100 80GB)
bash scripts/run_datagen_deepseek_r1_qwen_32b.sh

# DeepSeek-R1-Distill-Llama-70B (4x A100 80GB or 2x B200)
bash scripts/run_datagen_deepseek_r1_llama_70b.sh

# GPT-OSS 120B (4x A100 80GB or equivalent)
bash scripts/run_datagen_gpt_oss_120b.sh
```

Each script runs both stages sequentially for each dataset — Stage 1 generates rollouts via vLLM, Stage 2 harvests final-layer hidden states via nnsight. Outputs land in `data/<model>/<dataset>/` relative to the repo root.

> **IMPORTANT — Clone the repo to volume disk.** Stage 2 writes large `.pt` activation tensors to `data/` relative to wherever the repo is cloned. On RunPod (and similar platforms), container disk is small and ephemeral; volume disk is large and persistent. Clone the repo onto a **Network Volume** so that `data/` lands on persistent storage that survives pod termination.
>
> See [docs/runpod_network_volume_workflow.md](docs/runpod_network_volume_workflow.md) for the full setup guide, including how to create a volume, point HF caches to it, and choose between sequential and parallel runs.
>
> ```bash
> cd /runpod-volume
> git clone <repo-url>
> cd Reasoning-Theater
> ```

#### Estimated Stage 2 `.pt` activation file sizes

Sizes are for bfloat16 tensors over reasoning-trace tokens only (`hidden_dim × seq_len × 2 bytes`). Sequence length estimated at 8192 tokens average; MedQA capped at 2000 randomly sampled questions.

| Model | ARC-Easy (2369q) | MMLU (5330q) | ARC-Challenge (1168q) | MedQA (2000q) | GPQA (198q) | **Total** |
|---|---|---|---|---|---|---|
| Llama-8B (d=4096) | ~30 GB | ~67 GB | ~15 GB | ~25 GB | ~2.5 GB | **~140 GB** |
| Qwen-14B (d=5120) | ~37 GB | ~83 GB | ~18 GB | ~31 GB | ~3 GB | **~172 GB** |
| Qwen-32B (d=5120) | ~37 GB | ~83 GB | ~18 GB | ~31 GB | ~3 GB | **~172 GB** |
| Llama-70B (d=8192) | ~60 GB | ~133 GB | ~29 GB | ~50 GB | ~5 GB | **~277 GB** |
| GPT-OSS-120B (d=4096) | ~30 GB | ~67 GB | ~15 GB | ~25 GB | ~2.5 GB | **~140 GB** |

Ensure each machine's volume disk has at least **300 GB free** before starting data generation. Run `df -h /workspace` to check.

**After data generation:** Each machine will have a `data/` folder containing one subdirectory per dataset for that model. Copy the entire `data/` directory from each remote machine to your local machine before running analysis:

```bash
rsync -avz --progress user@remote-host:/path/to/repo/data/ /local/path/to/repo/data/
```

Run this for each of the five machines and merge into a single local `data/` directory.

Then run the analysis pipeline:

```bash
bash scripts/run_pipeline.sh experiments/example_analysis.yaml
```

### Example 2: Full pipeline with all layer activations

For multi-GPU setups, use `tensor_parallel_size` for vLLM inference and sharding for parallel activation harvesting. See `experiments/example_full_datagen.yaml` and `experiments/example_full_analysis.yaml` for the full configs.

```bash
# Stage 1: generate responses (uses tensor parallelism across 4 GPUs)
bash scripts/run_datagen.sh experiments/example_full_datagen.yaml stage1

# Stage 2: harvest activations in parallel across 4 jobs
for i in 0 1 2 3; do
  bash scripts/run_datagen.sh experiments/example_full_datagen.yaml stage2 --shard $i --total-shards 4 &
done
wait

# Analysis: train probes for all layers, run all stages
bash scripts/run_pipeline.sh experiments/example_full_analysis.yaml
```

### Reference

Individual stages can be run separately:

```bash
# Data generation
bash scripts/run_datagen.sh experiments/example_datagen.yaml stage1              # responses only
bash scripts/run_datagen.sh experiments/example_datagen.yaml stage2 --layer 17   # single layer activations
bash scripts/run_datagen.sh experiments/example_datagen.yaml stage2              # all layer activations

# Single probe layer (useful for debugging)
bash scripts/run_probe.sh experiments/example_analysis.yaml 17
```

Toggle individual analysis stages in the config YAML:

```yaml
setup:
  enabled: true           # prepare metadata and train/val/test split
probe:
  enabled: true           # train attention probes on hidden-state activations
forced_answer:
  enabled: false          # force the model to answer at each reasoning step
cot_monitor:
  enabled: false          # have an external LLM predict the answer from partial reasoning text
plots:
  enabled: true           # generate comparison plots (probe vs forced answer vs CoT monitor)
inflections:
  enabled: false          # detect backtracking / realization moments and correlate with probe shifts
```

Results are written to `results/<run_name>/`, with logs in `results/<run_name>/logs/`.
