# Reasoning Theater — Claude Context

## What This Project Is

A research codebase forked from the paper *"Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought"* (arxiv: 2603.05488). The core question: do LLMs genuinely reason during chain-of-thought, or perform "reasoning theater"? **Performativity** is defined as the gap between what internal representations encode vs. what the CoT text expresses.

The user is extending the original paper's codebase with new experiments described in `research_plan.md`.

---

## Research Plan Summary (`research_plan.md`)

The plan extends the paper along three primary axes:

1. **Dataset difficulty ladder** (5 datasets, easiest → hardest):
   - ARC-Easy (`allenai/ai2_arc`, config `ARC-Easy`)
   - MMLU (existing pipeline, ~5700 questions)
   - ARC-Challenge (`allenai/ai2_arc`, config `ARC-Challenge`)
   - MedQA (`openlifescienceai/MedQA-USMLE-4-options-hf`)
   - GPQA-Diamond (existing pipeline, 198 rows — hard ceiling)

2. **Probe generalizability** across datasets (5×5 transfer matrix, diagonal = 0)

3. **Probe architecture comparison**: last-token linear, mean-pool linear, mean-pool MLP, attention probe (paper's method)

**Phase order:**
| Phase | What | Dependencies |
|---|---|---|
| 0 | Dataset construction | None |
| 1 | Rollout collection + activation harvesting | Phase 0 |
| 2 | Replication baseline (MMLU only) | Phase 1 |
| 3 | Probe architecture comparison | Phase 2 |
| 4 | Generalizability matrix | Phase 3 |
| 5 | CoT monitor + forced answer eval | Phase 1 |
| 6 | Difficulty vs. performativity | Phases 3, 4, 5 |
| 7 | Temporal belief tracking (post-hoc) | Phase 6 |
| 8 | Calibration and early exit (post-hoc) | Phase 6 |
| 9 | Training process ablations | Phase 3 |

Phases 5 and 9 can run in parallel with Phase 4. Phases 7 and 8 can run in parallel after Phase 6.

---

## Codebase Structure

```
src/
  data_generation/
    data_gen_config.py      # DataGenerationConfig dataclass (YAML → config)
    datasets.py             # Dataset loading + question formatting
    stage1_responses.py     # vLLM response generation → JSON per question
    stage2_activations.py   # nnsight hidden-state extraction → .pt per question/layer
    utils.py
  analysis/
    experiment_config.py    # ExperimentConfig (YAML schema, nested dataclasses)
    setup_data.py           # Metadata CSV, stratified train/val/test splits
    run_probing.py          # AttentionProbe + LinearProbe training
    run_forced_answering.py # Forced single-token completions at each step
    run_cot_monitor.py      # External LLM CoT monitoring via OpenRouter
    find_inflection_points.py
    inflection_point_analysis.py
    plots.py
    data_loading.py         # RunData dataclass (caches results)
    utils.py                # Token boundary computation, step splitting
streamlit_app/              # Interactive visualization (Plotly + Streamlit)
experiments/                # YAML configs per run
scripts/                    # run_datagen.sh, run_pipeline.sh, run_probe.sh
```

---

## Pipeline Flow

1. **Stage 1** (vLLM subprocess): Generate full reasoning traces → JSON files; process exits, VRAM freed
2. **Stage 2** (nnsight subprocess): Re-run as prefill-only forward pass, extract hidden states → `.pt` files; process exits, VRAM freed
3. **Setup**: Parse JSONs → `predictions_metadata.csv`, stratified train/val/test splits
4. **Probing**: Train probes on activations to predict model's final answer
5. **Forced Answering**: Force single-token completion at each step boundary; results stored as `layer_idx = -1`
6. **CoT Monitor**: External LLM predicts answer from partial trace via OpenRouter; variants use `layer_idx = -2, -3, -4` or `monitor_variant` column
7. **Inflection Detection**: Identify backtracking/realization moments
8. **Plotting**: Heatmaps, calibration curves, agreement analysis

---

## Key Technical Details

- **Activation target (Phase 1 extension):** Hook into the **final layer's post-layernorm residual stream** (`hidden_states[-1]` with `output_hidden_states=True`). Must fire after both residual additions within the final block AND after the final standalone LayerNorm.
- **Token range:** Capture hidden states over reasoning trace tokens only — not prompt tokens, not tokens after closing `</think>` tag.
- **Probe training target:** Model's final answer (not ground truth), cross-entropy loss, random prefix truncation, best checkpoint by val loss.
- **Cumsum trick:** Used for simultaneous predictions at every prefix length without re-running the probe.
- **Data namespacing:** Each dataset gets its own folder for Stage 1 JSONs, Stage 2 `.pt` files, and step-level CSVs.
- **Schema additions:** Step-level CSVs include `dataset`, `probe_architecture`, `monitor_variant` columns from the start.

---

## Tech Stack

- **Python 3.10**, package manager: `uv`
- **Inference:** vLLM 0.14.1
- **Activation extraction:** nnsight ≥ 0.5.15
- **External APIs:** OpenRouter (CoT monitor, inflection detection); `OPENROUTER_API_KEY` in `.env`
- **Config:** YAML-driven; all experiments configured without code changes
- **Visualization:** Streamlit + Plotly (interactive), matplotlib + seaborn (static)
- **Remote storage:** s3fs / fsspec (S3-compatible)

---

## Quality Gate

Phase 2 is a required quality gate: replicate paper's probe accuracy and performativity gap on MMLU before proceeding to Phase 3+. If it fails, the bug is in Stage 2 extraction or probe implementation.