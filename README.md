# Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought

This is a fork of the codebase for the paper ["Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"](https://arxiv.org/abs/2603.05488). We are replicating and verifying the original results, and extending the work along four axes: a four-dataset difficulty ladder, probe architecture / training process variations, probe generalizability across datasets, and CoT monitor LLM variation.

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

Each model runs on its own machine. Per-model runner scripts in `scripts/` loop over all four dataset configs for that model and call `run_datagen.sh both --layer <final_layer>`, harvesting only the final layer's activations.

```
experiments/
  deepseek_r1_qwen_32b/      # one yaml per dataset
    mmlu_datagen.yaml
    arc_challenge_datagen.yaml
    medqa_datagen.yaml
    gpqa_datagen.yaml
  gpt_oss_120b/
    mmlu_datagen.yaml
    arc_challenge_datagen.yaml
    medqa_datagen.yaml
    gpqa_datagen.yaml
```


> **IMPORTANT — Clone the repo to volume disk.** Stage 2 of the data generation flow writes large `.pt` activation tensors to `data/` relative to wherever the repo is cloned. On RunPod (and similar platforms), container disk is small and ephemeral; volume disk is large and persistent. Clone the repo onto a **Network Volume** so that `data/` lands on persistent storage that survives pod termination.
>
> See [docs/runpod_network_volume_workflow.md](docs/runpod_network_volume_workflow.md) for the full setup guide. 


#### 0. Install uv and clone repo into network volume

cd /workspace

curl -LsSf https://astral.sh/uv/install.sh | sh OR wget -qO- https://astral.sh/uv/install.sh | sh

git clone https://github.com/ijamil1/Reasoning-Theater.git
cd Reasoning-Theater


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

Check available volume disk space, then create the cache directories vLLM will use when downloading model weights:

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

NOTE: the above setup will download model wts to the network volume (persistent) as the network volume replace the default volume disk.

#### 4. Run data generation

To run data generation for a given model, execute its runner script on the appropriate machine:

```bash
# DeepSeek-R1-Distill-Qwen-32B (2x A100 80GB with >= 70 GB ephemeral volume disk)
bash scripts/run_datagen_deepseek_r1_qwen_32b.sh

# GPT-OSS 120B (4x A100 80GB with >= 250 GB ephemeral volume disk)
bash scripts/run_datagen_gpt_oss_120b.sh
```

Each script runs both stages sequentially for each dataset — Stage 1 generates rollouts via vLLM, Stage 2 harvests final-layer hidden states via nnsight. Outputs land in `data/<model>/<dataset>/` relative to the repo root.

#### Estimated Stage 2 `.pt` activation file sizes

Sizes are for bfloat16 tensors over reasoning-trace tokens only (`hidden_dim × seq_len × 2 bytes`). Sequence length estimated at 8192 tokens average; MedQA capped at 2000 randomly sampled questions.

| Model | MMLU (5330q) | ARC-Challenge (1168q) | MedQA (2000q) | GPQA (198q) | **Total** |
|---|---|---|---|---|---|
| Qwen-32B (d=5120) | ~83 GB | ~18 GB | ~31 GB | ~3 GB | **~135 GB** |
| GPT-OSS-120B (d=4096) | ~67 GB | ~15 GB | ~25 GB | ~2.5 GB | **~110 GB** |

Ensure the  machine's network volume (persistent) disk has ~ **700 GB free** before starting data generation (needs to account for repo size + stage 1 outputs + stage 2 ouputs + 64 GB + 240 GB of model wts). 

**After data generation:** 
Can run phase 4 (probe generalizability):
  
  BEFORE DOING SO, WE NEED TO MOVE MMLU NORMALIZATION STATS INTO directory: {run_name_prefix}_train_{dataset['name']} which resolves to 
  “results/phase4_qwen_32b_train_mmlu/”

```bash
bash scripts/run_phase4.sh experiments/deepseek_r1_qwen_32b/phase4.yaml
```

Can run the analysis pipeline:
  NOTE 1: export OpenRouter API key in shell session so CoT monitor can work

  NOTE 2: tweak analysis yamls depending on if you want plotting, inflection, forced answering (ensure running on 2 GPUs if this is enabled). Running probe eval + forced answering sequentially MAY fail due to competing GPU memory usage

  NOTE 3: Copy normalization_stats.json from the phase 4 mmlu-trained probe into the analysis root for each dataset’s analysis

```bash
bash scripts/run_pipeline.sh experiments/example_analysis.yaml #runs analysis for a single dataset
```

```bash
bash scripts/run_analysis_deepseek_r1_qwen_32b.sh #runs analysis for 4 datasets at a time
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

### SCP from pod to local machine (General Form)
```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/token_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/token_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/step_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/token_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/token_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/step_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/token_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/token_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/step_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/token_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/token_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/step_level/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/predictions_metadata.csv /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/predictions_metadata.csv /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/predictions_metadata.csv /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/
```

```bash
scp -P 11027 -i /Users/irfanjamil/.ssh/id_ed25519 root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/predictions_metadata.csv /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/
```

```bash
scp -P 23617 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.233.200:/workspace/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/*" /Users/irfanjamil/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/
```

```bash
scp -P 23617 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.233.200:/workspace/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/*" /Users/irfanjamil/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/
```

```bash
scp -P 23617 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.233.200:/workspace/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/*" /Users/irfanjamil/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/
```

```bash
scp -P 23617 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.233.200:/workspace/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/*" /Users/irfanjamil/Reasoning-Theater/plots/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/
```

```bash
scp -P 23617 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.233.200:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmlu_pro_Eval/cot_monitor_completions.json" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmlu_pro_Eval
```

```bash
scp -P 42904 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/step_level/
```

```bash
scp -P 42904 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/step_level/
```

```bash
scp -P 42904 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/step_level/
```

```bash
scp -P 42904 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.83:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/step_level/*" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/step_level/
```


```bash
scp -P 15169 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.232.245:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/predictions_metadata.csv" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/
```

```bash
scp -P 15169 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.232.245:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/predictions_metadata.csv" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/
```

```bash
scp -P 15169 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.232.245:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/predictions_metadata.csv" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/
```

```bash
scp -P 15169 -i /Users/irfanjamil/.ssh/id_ed25519 "root@38.128.232.245:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/predictions_metadata.csv" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/
```

--

```bash
scp -P 45334 -i /Users/irfanjamil/.ssh/id_ed25519 "root@69.19.136.173:/workspace/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/predictions_metadata.csv" /Users/irfanjamil/Reasoning-Theater/results/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/
```
