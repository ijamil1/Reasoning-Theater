#!/usr/bin/env python3
"""Phase 3: Probe architecture comparison on MMLU + DeepSeek-R1-Distill-Qwen-32B.

Trains all four probe architectures (attention, recency_linear, recency_mlp,
attention_mlp) for 50 epochs across three batch sizes (32, 64, 128).  For each
(probe_type, batch_size) combination the best checkpoint (by val loss) is
evaluated on the held-out MMLU test set.  Final ranking: mean test accuracy
averaged over batch sizes.

Usage:
    bash scripts/run_phase3.sh experiments/deepseek_r1_qwen_32b/phase3.yaml
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Resolve project root so the script can be run from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.experiment_config import (  # noqa: E402
    CotMonitorConfig,
    DataConfig,
    ExperimentConfig,
    ForcedAnswerConfig,
    ProbeConfig,
    RunConfig,
    SetupConfig,
)
from src.analysis.run_probing import (  # noqa: E402
    AttentionMLPProbe,
    AttentionProbe,
    RecencyWeightedLinearProbe,
    RecencyWeightedMLPProbe,
    compute_normalization_stats,
    evaluate_accuracy_averaged_over_positions,
    load_checkpoint,
    load_metadata,
    load_samples,
    load_split,
    save_normalization_stats,
    train_one_layer,
)
from src.analysis.setup_data import setup_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PROBE_DISPLAY_NAMES = {
    "attention": "Attention",
    "recency_linear": "Recency Linear",
    "recency_mlp": "Recency MLP",
    "attention_mlp": "Attention MLP",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_phase3_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_experiment_config(
    phase3_cfg: dict,
    run_name: str,
    probe_type: str,
    batch_size: int,
    norm_stats_run_root: Optional[Path] = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig for a single (probe_type, batch_size) run."""
    data_raw = phase3_cfg["data"]
    probe_base = phase3_cfg["probe_base"]
    results_raw = phase3_cfg.get("results", {})

    results_dir = Path(results_raw.get("results_dir", "results"))
    seed = int(results_raw.get("seed", 42))

    run = RunConfig(run_name=run_name, results_dir=results_dir, seed=seed)

    data = DataConfig(
        responses_dir=Path(data_raw["responses_dir"]),
        activations_dir=Path(data_raw["activations_dir"]),
        tokenizer_model=data_raw.get(
            "tokenizer_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        ),
        dataset_name=data_raw.get("dataset_name"),
    )

    setup = SetupConfig(
        enabled=True,
        train_val_test_split_ratio=[0.8, 0.1, 0.1],
    )

    probe = ProbeConfig(
        num_layers=int(probe_base["num_layers"]),
        enabled=True,
        train=True,
        eval=False,  # we run eval ourselves via evaluate_accuracy()
        selected_layer=int(probe_base.get("selected_layer", 63)),
        probe_type=probe_type,
        label_type=str(probe_base.get("label_type", "model_ans")),
        batch_size=batch_size,
        learning_rate=float(probe_base.get("learning_rate", 0.005)),
        weight_decay=float(probe_base.get("weight_decay", 0.001)),
        num_epochs=int(probe_base.get("num_epochs", 50)),
        mlp_hidden_dim=int(probe_base.get("mlp_hidden_dim", 32)),
        recency_decay=float(probe_base.get("recency_decay", 0.02)),
        normalize_acts=bool(probe_base.get("normalize_acts", True)),
        disable_tqdm=bool(probe_base.get("disable_tqdm", True)),
        norm_stats_run_root=norm_stats_run_root,
    )

    forced = ForcedAnswerConfig(enabled=False)
    cot_monitor = CotMonitorConfig(enabled=False)

    return ExperimentConfig(
        run=run,
        data=data,
        setup=setup,
        probe=probe,
        forced_answer=forced,
        cot_monitor=cot_monitor,
    )


def instantiate_probe(
    probe_type: str,
    hidden_dim: int,
    output_dim: int,
    cfg: ExperimentConfig,
) -> torch.nn.Module:
    if probe_type == "recency_linear":
        return RecencyWeightedLinearProbe(
            hidden_dim, output_dim, torch.bfloat16, cfg.probe.recency_decay
        )
    if probe_type == "recency_mlp":
        return RecencyWeightedMLPProbe(
            hidden_dim,
            output_dim,
            torch.bfloat16,
            cfg.probe.recency_decay,
            cfg.probe.mlp_hidden_dim,
        )
    if probe_type == "attention_mlp":
        return AttentionMLPProbe(
            hidden_dim, output_dim, torch.bfloat16, cfg.probe.mlp_hidden_dim
        )
    # default: attention
    return AttentionProbe(hidden_dim, output_dim, torch.bfloat16, False, cfg.probe.mlp_hidden_dim)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _mean_acc(
    results: Dict[Tuple[str, int], float],
    probe_type: str,
    batch_sizes: List[int],
) -> float:
    accs = [results.get((probe_type, bs), float("nan")) for bs in batch_sizes]
    valid = [a for a in accs if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


def print_summary_table(
    results: Dict[Tuple[str, int], float],
    probe_types: List[str],
    batch_sizes: List[int],
) -> None:
    col_w = 14
    header = (
        f"{'probe_type':<22}"
        + "".join(f"{'bs='+str(bs):>{col_w}}" for bs in batch_sizes)
        + f"{'mean_acc':>{col_w}}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("Phase 3 Summary — test accuracy averaged over batch sizes")
    print(sep)
    print(header)
    print("-" * len(header))

    mean_accs: Dict[str, float] = {}
    for pt in probe_types:
        mean = _mean_acc(results, pt, batch_sizes)
        mean_accs[pt] = mean
        accs = [results.get((pt, bs), float("nan")) for bs in batch_sizes]
        row = f"{pt:<22}"
        for a in accs:
            row += f"{a*100:>{col_w}.2f}%" if not np.isnan(a) else f"{'N/A':>{col_w}}"
        row += f"{mean*100:>{col_w}.2f}%" if not np.isnan(mean) else f"{'N/A':>{col_w}}"
        print(row)

    print(sep)
    valid_means = {pt: v for pt, v in mean_accs.items() if not np.isnan(v)}
    if valid_means:
        winner = max(valid_means, key=valid_means.__getitem__)
        print(f"\nWinner (highest mean test acc): {winner} ({valid_means[winner]*100:.2f}%)\n")


def save_summary_csv(
    results: Dict[Tuple[str, int], float],
    probe_types: List[str],
    batch_sizes: List[int],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["probe_type"] + [f"bs_{bs}" for bs in batch_sizes] + ["mean_test_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pt in probe_types:
            accs = [results.get((pt, bs), float("nan")) for bs in batch_sizes]
            mean = _mean_acc(results, pt, batch_sizes)
            row: dict = {"probe_type": pt}
            for bs, a in zip(batch_sizes, accs):
                row[f"bs_{bs}"] = f"{a*100:.2f}%" if not np.isnan(a) else "N/A"
            row["mean_test_acc"] = f"{mean*100:.2f}%" if not np.isnan(mean) else "N/A"
            writer.writerow(row)
    logger.info(f"Summary CSV saved to {path}")


def plot_results(
    results: Dict[Tuple[str, int], float],
    probe_types: List[str],
    batch_sizes: List[int],
    plots_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    display_names = [PROBE_DISPLAY_NAMES.get(pt, pt) for pt in probe_types]
    colors = [plt.cm.tab10.colors[i] for i in range(len(probe_types))]
    mean_accs = [_mean_acc(results, pt, batch_sizes) for pt in probe_types]

    # --- Plot 1: mean test accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(display_names, [a * 100 for a in mean_accs], color=colors)
    ax.set_xlabel("Probe Architecture")
    ax.set_ylabel("Mean Test Accuracy (%)")
    ax.set_title(
        "Phase 3: Mean Test Accuracy by Probe Architecture\n"
        "(averaged over batch sizes 32, 64, 128)"
    )
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, mean_accs):
        if not np.isnan(acc):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = plots_dir / f"mean_test_acc.{ext}"
        fig.savefig(p)
        logger.info(f"Saved {p}")
    plt.close(fig)

    # --- Plot 2: accuracy per batch size + mean ---
    x = np.arange(len(batch_sizes) + 1)  # last group = mean
    width = 0.8 / len(probe_types)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (pt, display, color) in enumerate(zip(probe_types, display_names, colors)):
        accs_per_bs = [
            results.get((pt, bs), float("nan")) * 100 for bs in batch_sizes
        ]
        mean_val = _mean_acc(results, pt, batch_sizes) * 100
        values = accs_per_bs + [mean_val]
        offset = (i - len(probe_types) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=display, color=color)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Phase 3: Test Accuracy per Batch Size + Mean")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes] + ["Mean"])
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        p = plots_dir / f"acc_by_batch_size.{ext}"
        fig.savefig(p)
        logger.info(f"Saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: probe architecture comparison on MMLU"
    )
    parser.add_argument("config", type=Path, help="Path to phase3.yaml")
    args = parser.parse_args()

    phase3_cfg = load_phase3_yaml(args.config)
    results_raw = phase3_cfg.get("results", {})
    results_dir = Path(results_raw.get("results_dir", "results"))
    seed = int(results_raw.get("seed", 42))
    run_name_prefix = results_raw.get("run_name_prefix", "phase3")
    summary_csv_path = Path(results_raw.get("summary_csv", "results/phase3_summary.csv"))
    plots_dir = results_dir / "phase3_plots"

    probe_types: List[str] = phase3_cfg["probe_types"]
    batch_sizes: List[int] = phase3_cfg["batch_sizes"]
    layer_idx: int = int(phase3_cfg["probe_base"].get("selected_layer", 63))

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Probe types: {probe_types}")
    logger.info(f"Batch sizes: {batch_sizes}")
    logger.info(f"Total runs: {len(probe_types) * len(batch_sizes)}")

    # ------------------------------------------------------------------
    # Step 1: Setup — generate MMLU split once (idempotent).
    # We use a dedicated "phase3_setup" run so the split and metadata CSV
    # live in a stable location that all probe runs can reference.
    # ------------------------------------------------------------------
    setup_run_name = f"{run_name_prefix}_setup"
    setup_cfg = build_experiment_config(
        phase3_cfg, setup_run_name, "attention", batch_sizes[0]
    )
    logger.info(f"Running setup (generates split if absent): {setup_run_name}")
    setup_data(setup_cfg)

    # Load split and metadata once — they are invariant across all probe runs.
    canonical_split_path: Path = (
        setup_cfg.run.results_dir / setup_cfg.run.run_name / "train_val_test_split.json"
    )
    logger.info(f"Canonical split: {canonical_split_path}")
    split = load_split(canonical_split_path)
    meta_idx_map = load_metadata(setup_cfg.resolved_paths()["metadata"])
    logger.info(
        f"Split loaded: train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}"
    )

    # ------------------------------------------------------------------
    # Step 1b: Compute normalization stats once into the setup run's dir.
    # All 12 grid runs will load from there via norm_stats_run_root.
    # ------------------------------------------------------------------
    shared_norm_stats_root = setup_cfg.resolved_paths()["root"]
    shared_norm_stats_path = shared_norm_stats_root / "normalization_stats.json"
    if shared_norm_stats_path.exists():
        logger.info(f"Normalization stats already exist at {shared_norm_stats_path}, skipping computation")
    else:
        logger.info(f"Computing normalization stats once into setup run dir: {shared_norm_stats_root}")
        norm_stats = compute_normalization_stats(layer_idx, split["train"], setup_cfg)
        save_normalization_stats(setup_cfg, {layer_idx: norm_stats})
        logger.info("Normalization stats saved.")

    # ------------------------------------------------------------------
    # Step 2: Train + eval grid
    # ------------------------------------------------------------------
    grid_results: Dict[Tuple[str, int], float] = {}

    for batch_size in batch_sizes:
        for probe_type in probe_types:
            run_name = f"{run_name_prefix}_{probe_type}_bs{batch_size}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Run: {run_name}")
            logger.info(f"  probe_type={probe_type}  batch_size={batch_size}")

            cfg = build_experiment_config(
                phase3_cfg,
                run_name,
                probe_type,
                batch_size,
                norm_stats_run_root=shared_norm_stats_root,
            )

            # Train for 50 epochs; best checkpoint saved by val loss.
            train_one_layer(layer_idx, cfg, split, meta_idx_map)

            # Evaluate best checkpoint on the test set.
            samples_test = load_samples(
                layer_idx,
                split["test"],
                cfg,
                cfg.probe.label_type,
                is_training_set=False,
            )
            if not samples_test:
                logger.warning(f"No test samples for {run_name} — skipping eval")
                grid_results[(probe_type, batch_size)] = float("nan")
                continue

            hidden_dim = samples_test[0]["activation"].shape[1]
            output_dim = 1 if cfg.probe.label_type == "model_correct" else 4

            model = instantiate_probe(probe_type, hidden_dim, output_dim, cfg)
            load_checkpoint(model, cfg, layer_idx)
            model.to(device)

            test_acc = evaluate_accuracy_averaged_over_positions(
                model, samples_test, device, cfg.probe.label_type
            )
            logger.info(f"Test accuracy: {test_acc * 100:.2f}%")
            grid_results[(probe_type, batch_size)] = test_acc

            # Free GPU memory before the next run.
            del model
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: Summary + plots
    # ------------------------------------------------------------------
    print_summary_table(grid_results, probe_types, batch_sizes)
    save_summary_csv(grid_results, probe_types, batch_sizes, summary_csv_path)
    plot_results(grid_results, probe_types, batch_sizes, plots_dir)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
