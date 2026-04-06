# Plots Reference: What Each Plot Actually Measures

This document summarizes the exact computation behind each major plot in `src/analysis/plots.py`, including data sources, binning procedures, and known quirks.

---

## 1. `plot_probe_accuracy_heatmap`

**Data source:** `run.token_tables` (token-level CSVs) + `run.metadata_df`

**What it shows:** Probe accuracy as a function of layer (y-axis) and relative token position in the reasoning trace (x-axis).

**How relative position is computed:**

For each question, each token at position `token_idx` gets:
```
rel_position = (token_idx + 1) / seq_len
```
where `seq_len = max(token_idx) + 1` across all probe rows for that question. This is **purely token-based** — it has nothing to do with reasoning steps.

**Binning:** ceil-based — bin `k = ceil(rel_position * num_bins) - 1`, clipped to `[0, num_bins-1]`. Default `num_bins=100`.

**Target:** Model's final answer (not ground truth).

**Step-level CSVs:** Never read here.

---

## 2. `plot_probe_forced_agreement`

**Data source:** `run.step_level_df` exclusively.

**What it shows:** Stacked bar chart of agreement categories between probe and forced answering, as a function of relative reasoning step position.

**How relative position is computed:**

Step-based, normalized per question:
```
rel_position = (step_idx - min_step) / (max_step - min_step)
```
Single-step questions get `rel_position = 0.5`. Position 0.0 = first step, 1.0 = last step.

**Binning:** Floor-based — `bin = int(rel_position * num_bins)`, clipped to `[0, num_bins-1]`.

**Default `num_bins`:** `run.median_steps_per_question` — computed from all questions in `step_level_df` (all splits, not just test), using rows with `layer_idx >= 0`.

**Four agreement categories per bin:**
- `both_correct`: forced ✓, probe ✓
- `both_incorrect`: forced ✗, probe ✗
- `probe_only`: forced ✗, probe ✓ ← the "performativity" signal
- `forced_only`: forced ✓, probe ✗

**Target:** Model's final answer.

**Note on inconsistency vs. heatmap:** The heatmap uses token position; this plot uses step position. These are different granularities.

---

## 3. `plot_early_decoding_accuracy`

**Data source:** `run.step_level_df` exclusively.

**What it shows:** Accuracy vs. relative reasoning step position for three methods on the same axes: probe, forced answer, and CoT monitor.

**Methods and their `layer_idx` codes:**
- Probe: `layer_idx = probe_layer` (defaults to `run.best_layer`)
- Forced answer: `layer_idx = -1`
- CoT monitor: `layer_idx = -2`

**Relative position and binning:** Same floor-based step normalization as `plot_probe_forced_agreement`.

**Default `num_bins`:** `min(r.median_steps_per_question for r in runs)` — when comparing two datasets, the coarser one drives the resolution.

**CoT monitor N/A handling:** When a CoT monitor response is N/A or missing, `is_correct` is set to **0.25** (random-chance for 4-option MCQ) rather than 0 or dropped.

**X-axis mapping:** Bin `k` is plotted at `k / num_bins * 100` — the **left edge** of the bin interval, not the midpoint. This is inconsistent with `plot_ece_brier_by_position` which uses midpoints `(k + 0.5) / num_bins * 100`.

**Multi-run support:** Accepts 1 or 2 `RunData` objects for side-by-side dataset comparison (max 2).

---

## 4. `compute_area_between_curves` (and `stats.txt`)

**Data source:** `run.step_level_df` (via `compute_method_accuracy_by_position`), `run.token_level_df` (for logit stability only).

This function computes five sets of statistics and writes them to `{run.plots_dir}/stats.txt` when `save=True`. All statistics use step-based relative position.

### 4a. Area Between Curves (Probe vs CoT Monitor, Forced vs CoT Monitor)

For each pair:
```
diff[bin] = method_acc[bin] - cot_monitor_acc[bin]
area = sum(diff) / num_bins * 100
mean_diff = mean(diff)
```
Only bins present in **both** methods are included (inner join). Mean accuracy per method is `sum(accuracy_per_bin) / num_bins_compared` — **unweighted**, no correction for sample count per bin.

### 4b. Probe vs Forced Agreement (All Steps, No Binning)

Flat count over all `(question, step)` pairs across all questions — no position binning, no split filtering. Unit of analysis is a matched `(question_hash, step_idx)` pair present in both probe and forced rows. Produces a 2×2 confusion table with counts and percentages.

### 4c. Max Gap Stats

Per question: finds the step with the largest `|method_acc - cot_monitor_acc|` gap, then averages that per-question max gap across all questions.

### 4d. Probe Logit Stability

Measures how erratically the probe's softmax output changes between consecutive steps. Despite the name "logit stability", it operates on **post-softmax probabilities** (values in `[0,1]`, sum to 1).

Per consecutive step pair within a question:
```
ssd = sum((p_next - p_curr)^2)   # sum over all 4 elements (not mean)
```
Per question: `msd_q = mean(ssd over step pairs)`. Final reported value: `mean(msd_q over questions)` — macro-average.

**Maximum possible SSD** between two 4-class probability vectors = 4 (antipodal), but in practice much smaller.

### 4e. Slope Comparison

Uses `num_bins=20` (now fixed to use the same `num_bins` passed to `compute_area_between_curves`, not hardcoded). Computes how accuracy rises across the reasoning trace using two methods:

**Point-wise:**
```python
derivatives = np.diff(accuracy_values)   # accuracy[i+1] - accuracy[i]
avg_slope = mean(derivatives)
```
Δposition is implicitly 1 bin, so slope is in units of **accuracy per bin**.

**Quadratic fit:**
```python
x_normalized = (x - x.min()) / (x.max() - x.min())   # normalized to [0, 1]
fit: y = ax^2 + bx + c
avg_slope = a + b   # integral of derivative over [0, 1]
```
Slope is in units of **accuracy per full trace**.

**The two methods are on different scales** — point-wise slope is per-bin, quadratic slope is per-full-trace. To convert point-wise to the same scale you would multiply by `num_bins`.

---

## 5. `plot_calibration`

**Data source:** `run.token_level_df` (requires `parse_probe_output=True` when loading).

**What it shows:** Reliability diagram — how well the probe's confidence predicts its accuracy.

**Unit of analysis:** Every token at `probe_layer`, expanded into 4 `(confidence, correct)` pairs — one per option A/B/C/D. A trace of 100 tokens contributes 400 data points.

- **X-axis:** `p[j]` — the raw softmax probability assigned to option `j`
- **Y-axis:** Fraction of tokens in that confidence bin where `argmax(probs) == model_answer`
- **Target:** Model's final answer

The stored values are post-softmax probabilities (not logits), always in `[0, 1]` summing to 1.

**Perfect calibration line:** y = x diagonal.

---

## 6. `plot_ece_brier_by_position`

**Data source:** `run.token_level_df` (requires `parse_probe_output=True`).

**What it shows:** Two calibration quality metrics as a function of relative **token** position (not step position).

**Relative position:** Same token-based formula as the heatmap:
```
rel_position = (token_idx + 1) / seq_len
```
**X-axis mapping uses midpoints:** `(pos_bin + 0.5) / num_position_bins * 100`.

**ECE (Expected Calibration Error):**
Within each position bin, tokens are further sub-binned by their max confidence (`num_confidence_bins=10`). For each confidence sub-bin:
```
contribution = (n_in_bin / n_total) * |mean_accuracy - mean_confidence|
ECE = sum(contributions)
```
Range `[0, 1]`. Lower = better calibrated. Measures the average gap between stated confidence and actual accuracy.

**Brier Score:**
```
brier = sum((p[j] - one_hot_target[j])^2 over 4 options)   # per token
```
Averaged over all tokens in the position bin. Range `[0, 4]` for 4-class, but typically much smaller. A proper scoring rule that penalizes both wrong predictions and overconfidence.

---

## Key Cross-Plot Inconsistencies to Be Aware Of

| Property | Heatmap | Agreement / Early Decoding | ECE/Brier |
|---|---|---|---|
| Position basis | Token | Step | Token |
| Bin edge mapping | Left edge (ceil-based) | Left edge (floor-based) | Midpoint |
| Default num_bins | 100 | median_steps_per_question | 100 |
| CoT monitor N/A | N/A (not used) | Imputed to 0.25 | N/A (not used) |
| Target | Model answer | Model answer | Model answer |

---

## `median_steps_per_question`

Used as default `num_bins` for step-based plots. Computed from **all** questions in `step_level_df` (not just the test split) using rows with `layer_idx >= 0`, as: `median(max(step_idx) + 1 per question)`.

For reference, measured values for DeepSeek-R1-Distill-Qwen-32B:

| Dataset | Median Steps | 75th Percentile | Max |
|---|---|---|---|
| ARC-Challenge | 9 | 10 | 54 |
| MMLU | 10 | 12 | 336 |
| MedQA | 13 | 17 | 125 |
| GPQA-Diamond | 66.5 | 129 | 706 |

---

## `write_eval_outputs` (how token/step CSVs are produced)

The probe output CSVs are written by `write_eval_outputs` in `run_probing.py`. Key details:

**Cumsum trick:** Rather than running the probe once per prefix, it computes all prefix predictions in a single forward pass using cumulative weighted sums (attention or recency-weighted depending on probe architecture). `probs_all_cpu[t]` is the probe's softmax distribution at every token `t` simultaneously.

**Token-level CSV:** One row per token. Written fresh (full overwrite). Records `probe_pred` (argmax), `decoder_output` (full 4-element softmax vector as JSON), `step_idx` (which reasoning step that token belongs to).

**Step-level CSV:** One row per step. The step's representative probability vector is the **last token's vector in that step** (not an average — `vectors[-1]` despite the variable being named `avg`). Written under a file lock, merging with existing non-probe rows (forced answer, CoT monitor) to preserve them across separate pipeline runs.
