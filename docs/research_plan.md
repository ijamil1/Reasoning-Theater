# Research Plan: Extending "Reasoning Theater"

## Overview

This plan extends the findings of *Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought* (arxiv: 2603.05488). The paper introduces attention probes trained on transformer hidden states to predict a model's final answer from its reasoning trace, and defines **performativity** as the gap between what the model's internal representations encode and what its chain-of-thought expresses. We extend the paper along four primary axes: a four-dataset difficulty ladder, probe architecture / training process variations, probe generalizability across datasets, and CoT monitor LLM variation. Several secondary analyses follow from the data collected for these primary ideas.

The existing codebase implements the paper's pipeline and is the foundation for all work here. Most phases require targeted modifications rather than new infrastructure.

---

## Dataset Ladder

Four datasets ordered easiest to hardest for frontier-scale models:

| Dataset | HuggingFace Path | Notes |
|---|---|---|
| MMLU | existing pipeline | test split, ~5330 questions; used as full pool for our own train/val/test splits |
| ARC-Challenge | `allenai/ai2_arc` | config `ARC-Challenge` |
| MedQA | `openlifescienceai/MedQA-USMLE-4-options-hf` | 2000 randomly sampled from train split |
| GPQA-Diamond | existing pipeline | 198 rows total, no canonical test split; full pool used for our own train/val/test splits |

**Notes on the ladder:**
- GPQA-Diamond's 198 rows is a hard ceiling that constrains split sizes and may affect statistical power — if comparable split sizes across datasets are desired, GPQA-Diamond sets the floor.
- Dataset-specific formatting is handled separately and is not part of this plan.

---

## Phase 0: Dataset Construction

Load all five datasets from HuggingFace using the paths and configs above. Produce a unified question bank with consistent fields: `dataset`, `question_id`, `question`, `choices` (A/B/C/D), `correct_answer` (letter). Run on two models: DeepSeek-R1-Distill-Qwen-32B and GPT-OSS-120B.

Each dataset gets its own folder for all downstream outputs — Stage 1 JSONs, Stage 2 `.pt` files, and step-level CSVs — mirroring the structure of the existing pipeline but namespaced by dataset.

Sample sizes are finalized during this phase after inspecting each dataset's available rows.

---

## Phase 1: Data Generation

> **Prerequisite:** The infrastructure decision (how models are served and whether direct forward pass access is available) must be resolved before this phase begins. Whatever setup is used must support direct access to the forward pass for hidden state extraction. vLLM is already in use in the existing pipeline and is the natural choice for large models.

### Stage 1 — Rollout Collection (vLLM subprocess)

Load model weights, run inference on all questions across all datasets (train + eval combined, no split yet). Extend the existing JSON schema with a `dataset` field. Save each question's full reasoning trace and final answer as a JSON file into the dataset's own folder. Process exits, VRAM freed.

### Stage 2 — Activation Harvesting (nnsight subprocess)

Reload model weights, re-run each rollout as a prefill-only forward pass. Hook into the **final layer's post-layernorm residual stream** — `hidden_states[-1]` in HuggingFace terminology. This is the residual stream value after both residual additions within the final decoder block and after the final standalone LayerNorm. This is a one-line change from the existing implementation which hooks layer 17 or all layers.

Specifically:
- The **residual stream** is the main running accumulation `x` after both the attention and FFN residual additions within each block. It is not the attention output alone or the FFN output alone.
- `hidden_states[-1]` from a HuggingFace forward pass with `output_hidden_states=True` returns exactly this — the post-layernorm residual stream at the output of the final block.
- The hook should fire after both residual additions within the final block and after the final standalone LayerNorm.

Capture hidden states `[seq_len, hidden_dim]` at every token position of the **reasoning trace only** — not the prompt tokens, not the final answer tokens after the closing think tag. Identify the reasoning trace token range by locating the start and end think tag token IDs. Save to disk as `.pt` files into the dataset's own folder, one file per question. Process exits, VRAM freed.

**Verification step before running the full job:**
- Run on a single small model and small dataset subset first.
- Confirm that `hidden_states[-1]` corresponds to the post-layernorm residual stream by checking against the model's `modeling_*.py` file.
- Explicitly verify the hook fires after both residual additions within the final block and after the final standalone LayerNorm.
- Run with `torch.no_grad()` and call `.detach().cpu()` on hidden states before storing to avoid retaining the computation graph.

### Setup

Parse Stage 1 JSONs into `predictions_metadata.csv` with a `dataset` column. Randomly assign question hashes into train/val/test splits, **stratified by dataset** so each split has representation from all five datasets. Pre-create empty step-level CSVs for test questions only, one per dataset folder. Step-level CSVs include columns for `dataset`, `probe_architecture`, and `monitor_variant` from the start to avoid schema changes later.

---

## Phase 2: Replication Baseline

Train the attention probe on the MMLU train split for DeepSeek-R1 only. Evaluate on the MMLU test split. Verify that probe accuracy and performativity gap numbers are in the same ballpark as the paper's results.

> **Quality gate:** If this fails, something is wrong in Stage 2 extraction or the probe implementation. Do not proceed to Phase 3 until resolved. This is the only quality gate before large-scale experiments begin.

---

## Phase 3: Probe Architecture Comparison (Idea 3)

Still on DeepSeek-R1 + MMLU only. Extend the probe training stage to support all five architectures, using the same training loop, cross-entropy loss against the model's final answer (not ground truth), random prefix truncation, and best-checkpoint-by-val-loss pattern throughout.

The four architectures:

- **Attention probe** (paper's method, baseline): learned attention-weighted pooling of hidden states. Expected to perform best.
- **Recency-weighted linear probe**: weighted average of `h[0:T, :]` where weights decay exponentially with distance from the current token (i.e. more recent tokens get higher weight), followed by a linear layer + softmax. A structured alternative to uniform mean pooling that encodes the intuition that recent reasoning steps are more predictive of the final answer.
- **Recency-weighted MLP probe**: same exponential recency weighting as above, but the resulting pooled vector is passed into a two-layer MLP with ReLU non-linearity rather than a linear layer. Tests whether a non-linear read-out on top of recency-weighted pooling recovers additional signal.
- **Attention + MLP probe**: attention-weighted pooling (as in the attention probe) used as the aggregation mechanism, but the pooled vector is then passed through a two-layer MLP with ReLU non-linearity before the final classification layer. Tests whether the bottleneck in the attention probe is the linear read-out.

Evaluate all variants using the cumsum trick for simultaneous predictions at every prefix length. Plot accuracy vs. relative reasoning position for each architecture.

**Selection criteria for Phase 4+:** Carry forward the attention probe as the primary architecture. The recency-weighted linear probe serves as a structured baseline against the attention probe's learned weighting.

---

## Phase 4: Generalizability Matrix (Idea 2)

Scale probe training to all (model, dataset) training pairs using the architecture(s) selected in Phase 3. For each trained probe, evaluate on all four datasets' test splits.

**Degradation metric:**

> Δ(i→j) = Accuracy(train=j, eval=j) − Accuracy(train=i, eval=j)

This produces a **4×4 transfer matrix per model** where the diagonal is always 0 by definition and off-diagonal entries represent accuracy loss from training on a different dataset distribution. Large off-diagonal values indicate poor transfer; small values indicate the probe has learned something general about how the model encodes its answer rather than something dataset-specific.

**Additional analysis:** Test whether Δ(i→j) correlates with the difficulty distance between datasets i and j, measured as the difference in model accuracy between dataset i and dataset j. If training on an easy dataset transfers poorly to a hard one but not vice versa, this suggests the probe is learning the model's confident-answer regime specifically — which connects directly to Idea 1.

**Impact on Phase 6:** 
- If off-diagonal degradation is consistently small (under 3–5 percentage points): use a universally-trained probe for the difficulty analysis.
- If degradation is large: use per-dataset probes and note the interpretive implication — you are comparing probes trained on different distributions, not the same probe applied to different difficulty levels.

---

## Phase 5: CoT Monitor and Forced Answer Evaluation

This phase has no dependency on Phase 4 and can run in parallel with it.

### CoT Monitor (API calls, no local compute)

For each question, build cumulative prompts at each step boundary and send to Gemini Flash via OpenRouter concurrently, matching the existing pipeline. The monitor predicts which answer the generating model will output — not the answer itself.

**Monitor ablation:** Run multiple configurations in parallel — vary the monitor model and the prompt. Record monitor model and prompt variant using distinct `layer_idx` values (e.g., -2, -3, -4) or a dedicated `monitor_variant` column in the step-level CSVs.

### Forced Answer Evaluation (vLLM subprocess)

Reload model weights. For each question × each step boundary, construct a prompt cutting off reasoning at that step and forcing immediate answer generation. Batch all prompts, generate one token each, collect logprobs for A/B/C/D. Inject results into step-level CSVs as `layer_idx = -1`. Process exits, VRAM freed.

---

## Phase 6: Difficulty vs. Performativity (Idea 1)

With probe predictions, CoT monitor predictions, and forced answer predictions all in the step-level CSVs, compute the performativity gap per dataset per model. The performativity gap at each timestep is the difference in accuracy between the probe and the CoT monitor at that prefix position.

**Difficulty proxy:** Rank datasets by model accuracy. This is model-relative by design — you are testing whether a model's own subjective difficulty predicts its own performativity, not whether objective task hardness does.

**Null hypothesis:** The performativity gap is significantly lower on hard datasets than on easy ones.

**Test:** Spearman rank correlation between difficulty rank and mean performativity gap (averaged over the full trace) across four datasets and two models.

### Confounds to Control For

**Trace length:** Harder tasks may produce longer reasoning traces, giving the CoT monitor more text to work with and mechanically shrinking the performativity gap. Check whether trace length correlates with difficulty in your data. If it does, compare probe vs. monitor at matched percentile positions as a robustness check.

**Near-chance accuracy:** If a model scores near 25% on a dataset, the probe's four-way classification is also near chance and the performativity gap compresses mechanically, independent of any faithfulness phenomenon. Exclude datasets where a given model scores below 35% accuracy, or flag those data points separately in the analysis.

---

## Phase 7: Temporal Belief Tracking (Idea 4)

Post-hoc analysis on existing step-level CSVs. No additional inference needed.

For each example, track the probe's argmax prediction at every 5% prefix interval and identify two commitment points:

- **Internal commitment point:** First timestep at which the probe's argmax prediction stabilizes to the model's final answer.
- **Expressed commitment point:** First timestep at which the CoT monitor predicts the correct final answer.

The gap between these two is the **per-example performativity event**. This reframes performativity from an aggregate statistic into a per-example event, which is more interpretable.

**Analyses:** 
- Does gap size correlate with dataset difficulty?
- Does gap size correlate with whether the model's final answer is correct?
- Does gap size correlate with trace length?

---

## Phase 8: Calibration and Early Exit (Idea 5)

Post-hoc analysis on existing step-level CSVs. No additional inference needed.

For each dataset, find the earliest timestep at which probe accuracy reaches 97% of its asymptotic value and report token savings at that point. Test whether this early-exit threshold degrades monotonically with difficulty.

**Prediction:** Harder datasets require a larger fraction of the trace before the probe stabilizes, meaning early exit is least safe precisely where compute savings are most desirable. This connects the paper's practical early-exit finding to the difficulty axis.

---

## Phase 9: Training Process Ablations

No additional GPU inference needed — re-use already-saved `.pt` activation files. Pure PyTorch re-training on existing activations.

Systematically vary:
- Number of training epochs
- Batch size
- Learning rate
- Prefix length sampling range

Report sensitivity of the main results (probe accuracy curve shape and performativity gap) to each of these choices. This strengthens the robustness of conclusions and is low-cost since no new activations need to be collected.

---

## Order of Operations

| Phase | What | Dependencies |
|---|---|---|
| 0 | Dataset construction | None |
| 1 | Rollout collection + activation harvesting | Phase 0, infrastructure decision |
| 2 | Replication baseline | Phase 1 |
| 3 | Probe architecture comparison | Phase 2 |
| 4 | Generalizability matrix | Phase 3 |
| 5 | CoT monitor + forced answer evaluation | Phase 1 |
| 6 | Difficulty vs. performativity | Phases 3, 4, 5 |
| 7 | Temporal belief tracking | Phase 6 |
| 8 | Calibration and early exit | Phase 6 |
| 9 | Training process ablations | Phase 3 |

**Parallelism:**
- Phases 5 and 9 can run in parallel with Phase 4.
- Phases 7 and 8 can run in parallel with each other after Phase 6.
