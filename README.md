# Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought

This is a fork of the codebase for the paper ["Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"](https://arxiv.org/abs/2603.05488). We are empirically investigating the paper's observation that performative CoT is difficulty-dependent by testing whether this finding generalizes when evaluated across a broader set of datasets of varying difficulty.

The original experiments were run on DeepSeek-R1-0528-671B and GPT-OSS-120B across MMLU-Redux-2.0 and GPQA-Diamond. Our experiments use DeepSeek-R1-Distill-Qwen-32B across five datasets spanning a difficulty ladder: MMLU-Redux-2.0, ARC-Challenge, MMLU-Pro, MedQA, and GPQA-Diamond.

## Results

**[Full write-up on Medium](https://medium.com/@irfanjamil_72723/replicating-and-stress-testing-observations-from-reasoning-theater-disentangling-model-beliefs-04f9dde6637a?postPublishedType=repub)**

**TLDR:** I find evidence that CoT performativity in LLMs does seem to be a generalizable phenomenon occurring across models and datasets. Activation probing across multiple different datasets mostly supports the authors' observations that CoT performativity is difficulty-dependent (i.e.: as tasks get harder, performativity decreases). However, when including forced answering in our analysis, the empirical evidence does not fully support this. Putting this all together, CoT performativity seems to be a general phenomenon though the difficulty-dependent finding in the paper may not be fully generalizable/robust across models and datasets. This has implications for understanding LLM behavior and reasoning faithfulness but less so for the early-exit strategy used to reduce inference costs.

**Plots — max(Probe, Forced Answer) vs CoT Monitor accuracy by relative reasoning position:**

- [MMLU-Redux-2.0](plots/analysis_deepseek_r1_qwen_32b_mmluTrain_mmluEval/cot_vs_best_probe_forced.pdf)
- [ARC-Challenge](plots/analysis_deepseek_r1_qwen_32b_mmluTrain_arcEval/cot_vs_best_probe_forced.pdf)
- [MMLU-Pro](plots/analysis_deepseek_r1_qwen_32b_mmluTrain_mmlu_pro_Eval/cot_vs_best_probe_forced_mmluPro.jpg)
- [MedQA](plots/analysis_deepseek_r1_qwen_32b_mmluTrain_medqaEval/cot_vs_best_probe_forced.pdf)
- [GPQA-Diamond](plots/analysis_deepseek_r1_qwen_32b_mmluTrain_gpqaEval/cot_vs_best_probe_forced.pdf)

## Setup and Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh OR wget -qO- https://astral.sh/uv/install.sh | sh #install uv

git clone https://github.com/ijamil1/Reasoning-Theater.git
cd Reasoning-Theater

uv venv
uv sync
cp .env.example .env  # add your OPENROUTER_API_KEY if running the CoT monitor / inflection stages
```

The project has the following dependency surfaces
- `pyproject.toml` — full project (training, inference, analysis). Managed by `uv`.


## Usage

### Our experimental setup

All YAML configurations for our experiments live in `experiments/deepseek_r1_qwen_32b/`. These files collectively define the settings used across data generation, probe training, and analysis for all five datasets.

#### Research questions

Our experiments were motivated by two questions, both aimed at determining how robust and generalizable the paper’s difficulty-dependent performativity finding is:

**1. Probe generalizability: does the MMLU-Redux-trained probe transfer to other datasets?**

The paper’s activation probes were trained on a subset of MMLU-Redux-2.0. The authors found that fine-tuning on GPQA-Diamond yielded negligible improvement. Does this hold when including datasets of meaningfully different difficulty and domain? We train a probe on each dataset’s train split and evaluate it on every other dataset’s test split, building a full cross-dataset transfer matrix. This lets us determine whether cross-domain transfer is strong enough to justify a single fixed probe, or whether each dataset requires its own.

This question acts as a methodological guardrail. If, for some dataset D, a probe trained on D’s train split is substantially more predictive of the model’s final answer than any cross-domain probe, then using a fixed probe biases results toward *underestimating* CoT performativity on D (since the internal-representation signal would be understated). That would be a confounder when comparing performativity across datasets of varying difficulty.

**2. Difficulty-dependent performativity: does the finding generalize across more datasets?**

How robust is the observation that harder tasks elicit more genuine reasoning than easier tasks? Does the gap between internal representations and CoT text scale consistently with task difficulty across a 5-dataset ladder?

#### Step 1: Probe generalizability (Phase 4)

We first ran Phase 4 to train a probe on each dataset’s train split and evaluate it on every other dataset’s test split, producing a full cross-dataset transfer matrix. This confirmed that cross-domain transfer was strong enough to justify using a single fixed probe across all datasets, with the MMLU-Redux-trained probe selected as that probe as it was the best performing.

```bash
bash scripts/run_phase4.sh experiments/deepseek_r1_qwen_32b/phase4.yaml
```

This trains a single probe on each dataset’s train split and evaluates cross-dataset transfer, producing a summary CSV at `results/phase4_summary.csv`.

#### Step 2: Data generation

Once probe generalizability was confirmed, we ran data generation across all five datasets. This is compute- and storage-intensive: Stage 1 collects full reasoning traces via vLLM, Stage 2 re-runs them as prefill-only forward passes to extract hidden states for a yaml-specified layer. Outputs land in `data/` and can consume many GBs of disk.

```bash
bash scripts/run_datagen_deepseek_r1_qwen_32b.sh
```

Each script runs both stages sequentially for each dataset. Outputs land in `data/<model>/<dataset>/` relative to the repo root.

#### Step 3: Analysis

With data in place, run the analysis pipeline across all datasets:

```bash
bash scripts/run_analysis_deepseek_r1_qwen_32b.sh
```

This sequentially runs probe evaluation (using the MMLU-Redux-trained probe), forced answering, CoT monitoring, and plotting for each dataset. Results are written to `results/<run_name>/`, with logs in `results/<run_name>/logs/`.

  > **NOTE 1:** Export your OpenRouter API key in the shell session before running so the CoT monitor stage can make API calls.

  > **NOTE 2:** Tweak the per-dataset analysis YAMLs to enable or disable individual stages (probe eval, forced answering, CoT monitoring, plotting). Running probe eval and forced answering sequentially may fail due to competing GPU memory — forced answering requires tensor parallelism across 2 GPUs.

  > **NOTE 3:** Copy `normalization_stats.json` from the Phase 4 MMLU-trained probe results into each dataset’s analysis results root before running probe eval.

### Reference

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

