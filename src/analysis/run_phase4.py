#!/usr/bin/env python3
"""Phase 4: Probe generalizability matrix.

Trains one probe per dataset (using that dataset's train + val splits) at a
single selected layer, then evaluates each trained probe on every dataset's
test split — including its own training dataset (diagonal = same-domain
sanity check).

The result is a raw accuracy matrix and a corresponding degradation
matrix where degradation[A][B] = accuracy[A][B] - accuracy[A][A].  Diagonal
cells are 0 in the degradation matrix; off-diagonal cells are typically
negative (transfer loss).

Normalization: activations for eval dataset B are normalized using the norm
stats computed from training dataset A's training split.  This tests raw
transfer with zero target-domain adaptation.

Usage:
    bash scripts/run_phase4.sh experiments/deepseek_r1_qwen_32b/phase4.yaml
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    evaluate_accuracy_averaged_over_positions,
    load_checkpoint,
    load_metadata,
    load_samples,
    load_split,
    train_one_layer,
)
from src.analysis.setup_data import setup_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_phase4_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_setup_cfg(
    phase4_cfg: dict,
    dataset: dict,
    run_name_prefix: str,
    results_dir: Path,
    seed: int,
) -> ExperimentConfig:
    """Build ExperimentConfig for the idempotent setup step (split + metadata)."""
    pb = phase4_cfg["probe_base"]
    results_raw = phase4_cfg.get("results", {})
    run = RunConfig(
        run_name=f"{run_name_prefix}_setup_{dataset['name']}",
        results_dir=results_dir,
        seed=seed,
    )
    data = DataConfig(
        responses_dir=Path(dataset["responses_dir"]),
        activations_dir=Path(dataset["activations_dir"]),
        tokenizer_model=results_raw.get(
            "tokenizer_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        ),
        dataset_name=dataset["name"],
    )
    probe = ProbeConfig(num_layers=int(pb["num_layers"]), enabled=False)
    return ExperimentConfig(
        run=run,
        data=data,
        setup=SetupConfig(enabled=True, train_val_test_split_ratio=[0.7, 0.1, 0.2]),
        probe=probe,
        forced_answer=ForcedAnswerConfig(enabled=False),
        cot_monitor=CotMonitorConfig(enabled=False),
    )


def build_train_cfg(
    phase4_cfg: dict,
    dataset: dict,
    run_name_prefix: str,
    results_dir: Path,
    seed: int,
) -> ExperimentConfig:
    """Build ExperimentConfig for training a probe on a given dataset."""
    pb = phase4_cfg["probe_base"]
    results_raw = phase4_cfg.get("results", {})
    selected_layer = int(pb.get("selected_layer", 63))
    run = RunConfig(
        run_name=f"{run_name_prefix}_train_{dataset['name']}",
        results_dir=results_dir,
        seed=seed,
    )
    data = DataConfig(
        responses_dir=Path(dataset["responses_dir"]),
        activations_dir=Path(dataset["activations_dir"]),
        tokenizer_model=results_raw.get(
            "tokenizer_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        ),
        dataset_name=dataset["name"],
    )

    probe = ProbeConfig(
        num_layers=int(pb["num_layers"]),
        enabled=True,
        train=True,
        eval=False,
        selected_layer=selected_layer,
        probe_type=str(pb.get("probe_type", "attention")),
        label_type=str(pb.get("label_type", "model_ans")),
        batch_size=int(pb.get("batch_size", 64)),
        learning_rate=float(pb.get("learning_rate", 0.005)),
        weight_decay=float(pb.get("weight_decay", 0.001)),
        num_epochs=int(pb.get("num_epochs", 10)),
        mlp_hidden_dim=int(pb.get("mlp_hidden_dim", 32)),
        recency_decay=float(pb.get("recency_decay", 0.02)),
        normalize_acts=bool(pb.get("normalize_acts", True)),
        disable_tqdm=bool(pb.get("disable_tqdm", False)),
    )

    return ExperimentConfig(
        run=run,
        data=data,
        setup=SetupConfig(enabled=True, train_val_test_split_ratio=[0.8, 0.1, 0.1]),
        probe=probe,
        forced_answer=ForcedAnswerConfig(enabled=False),
        cot_monitor=CotMonitorConfig(enabled=False),
    )


def build_eval_cfg(
    train_cfg: ExperimentConfig,
    eval_dataset: dict,
) -> ExperimentConfig:
    """Build ExperimentConfig for evaluating on eval_dataset using train_cfg's norm stats.

    The run_name is inherited from train_cfg so that resolved_paths() continues
    to point to the training run's root directory.  reuse_run_root is also set
    to that same root so that load_normalization_stats() loads training dataset
    A's normalization statistics and applies them to eval dataset B's activations.
    """
    data = DataConfig(
        responses_dir=Path(eval_dataset["responses_dir"]),
        activations_dir=Path(eval_dataset["activations_dir"]),
        tokenizer_model=train_cfg.data.tokenizer_model,
        dataset_name=eval_dataset["name"],
    )
    probe = ProbeConfig(
        num_layers=train_cfg.probe.num_layers,
        enabled=True,
        train=False,
        eval=False,
        selected_layer=train_cfg.probe.selected_layer,
        probe_type=train_cfg.probe.probe_type,
        label_type=train_cfg.probe.label_type,
        batch_size=train_cfg.probe.batch_size,
        mlp_hidden_dim=train_cfg.probe.mlp_hidden_dim,
        recency_decay=train_cfg.probe.recency_decay,
        normalize_acts=train_cfg.probe.normalize_acts,
        reuse_run_root=train_cfg.run.root,  # load A's norm stats for B's activations
        recompute_norm_stats=False,
        disable_tqdm=train_cfg.probe.disable_tqdm,
    )
    return ExperimentConfig(
        run=RunConfig(
            run_name=train_cfg.run.run_name,
            results_dir=train_cfg.run.results_dir,
            seed=train_cfg.run.seed,
        ),
        data=data,
        setup=SetupConfig(enabled=False),
        probe=probe,
        forced_answer=ForcedAnswerConfig(enabled=False),
        cot_monitor=CotMonitorConfig(enabled=False),
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
        return AttentionMLPProbe(hidden_dim, output_dim, torch.bfloat16, cfg.probe.mlp_hidden_dim)
    return AttentionProbe(hidden_dim, output_dim, torch.bfloat16, False, cfg.probe.mlp_hidden_dim)


def _peek_hidden_dim(activations_dir: Path, layer_idx: int) -> Optional[int]:
    """Return hidden_dim by peeking at the first activation file found."""
    layer_dir = activations_dir / f"layer_{layer_idx}"
    if not layer_dir.exists():
        return None
    for qhash_dir in layer_dir.iterdir():
        if not qhash_dir.is_dir():
            continue
        for pt_file in qhash_dir.glob("*.pt"):
            try:
                act = torch.load(pt_file, map_location="cpu")
                return int(act.shape[1])
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(val: float, as_pct: bool = True) -> str:
    if np.isnan(val):
        return "N/A"
    return f"{val * 100:.2f}%" if as_pct else f"{val:.4f}"


def print_matrix(
    matrix: Dict[str, Dict[str, float]],
    train_names: List[str],
    eval_names: List[str],
    title: str,
    as_pct: bool = True,
) -> None:
    col_w = 16
    header = f"{'':22}" + "".join(f"{name:>{col_w}}" for name in eval_names)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(title)
    print("Rows = train dataset, Cols = eval dataset")
    print(sep)
    print(header)
    print("-" * len(header))
    for train_name in train_names:
        row = f"{train_name:<22}"
        for eval_name in eval_names:
            val = matrix.get(train_name, {}).get(eval_name, float("nan"))
            row += f"{_fmt(val, as_pct):>{col_w}}"
        print(row)
    print(sep)


def save_matrix_csv(
    matrix: Dict[str, Dict[str, float]],
    train_names: List[str],
    eval_names: List[str],
    path: Path,
    as_pct: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["train_dataset"] + list(eval_names)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for train_name in train_names:
            row: dict = {"train_dataset": train_name}
            for eval_name in eval_names:
                val = matrix.get(train_name, {}).get(eval_name, float("nan"))
                row[eval_name] = _fmt(val, as_pct)
            writer.writerow(row)
    logger.info(f"Matrix CSV saved to {path}")


def plot_matrix(
    matrix: Dict[str, Dict[str, float]],
    train_names: List[str],
    eval_names: List[str],
    title: str,
    plots_dir: Path,
    filename_stem: str,
    cmap: str = "Blues",
    center: Optional[float] = None,
    colorbar_label: str = "Accuracy (%)",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(train_names)
    n_cols = len(eval_names)

    data = np.full((n_rows, n_cols), np.nan)
    for i, train_name in enumerate(train_names):
        for j, eval_name in enumerate(eval_names):
            val = matrix.get(train_name, {}).get(eval_name, float("nan"))
            if not np.isnan(val):
                data[i, j] = val * 100

    valid = data[~np.isnan(data)]
    vmin = float(np.min(valid)) if len(valid) else 0.0
    vmax = float(np.max(valid)) if len(valid) else 100.0

    if center is not None:
        extreme = max(abs(vmin - center), abs(vmax - center), 1.0)
        vmin = center - extreme
        vmax = center + extreme

    fig, ax = plt.subplots(figsize=(max(5, n_cols * 1.5), max(4, n_rows * 1.2)))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label=colorbar_label)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(eval_names, rotation=30, ha="right")
    ax.set_yticklabels(train_names)
    ax.set_xlabel("Eval dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title(title)

    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if not np.isnan(val):
                norm_val = (val - vmin) / (vmax - vmin + 1e-8)
                text_color = "white" if norm_val > 0.55 else "black"
                ax.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center", fontsize=9, color=text_color,
                )

    plt.tight_layout()
    p = plots_dir / f"{filename_stem}.png"
    fig.savefig(p, dpi=150)
    logger.info(f"Saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: probe generalizability matrix"
    )
    parser.add_argument("config", type=Path, help="Path to phase4.yaml")
    args = parser.parse_args()

    phase4_cfg = load_phase4_yaml(args.config)
    results_raw = phase4_cfg.get("results", {})
    results_dir = Path(results_raw.get("results_dir", "results"))
    seed = int(results_raw.get("seed", 42))
    run_name_prefix = results_raw.get("run_name_prefix", "phase4")
    summary_csv_path = Path(results_raw.get("summary_csv", "results/phase4_summary.csv"))
    plots_dir = results_dir / "phase4_plots"

    datasets: List[dict] = phase4_cfg["datasets"]
    eval_dataset_names: List[str] = [d["name"] for d in datasets]
    train_dataset_names: List[str] = [n for n in eval_dataset_names if n != "gpqa_diamond"]
    pb = phase4_cfg["probe_base"]
    layer_idx: int = int(pb.get("selected_layer", 63))
    probe_type: str = str(pb.get("probe_type", "attention"))
    label_type: str = str(pb.get("label_type", "model_ans"))
    output_dim: int = 1 if label_type == "model_correct" else 4

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Train datasets: {train_dataset_names}")
    logger.info(f"Eval datasets: {eval_dataset_names}")
    logger.info(f"Probe type: {probe_type}, Layer: {layer_idx}")
    logger.info(f"Total training runs: {len(train_dataset_names)}, total eval cells: {len(train_dataset_names) * len(eval_dataset_names)}")

    # ------------------------------------------------------------------
    # Step 0: Setup — generate split + metadata for every dataset once.
    # ------------------------------------------------------------------
    splits: Dict[str, dict] = {}
    meta_maps: Dict[str, dict] = {}

    for dataset in datasets:
        name = dataset["name"]
        setup_cfg = build_setup_cfg(phase4_cfg, dataset, run_name_prefix, results_dir, seed)
        logger.info(f"Running setup for dataset: {name}")
        setup_data(setup_cfg)

        split_path = setup_cfg.resolved_paths()["root"] / "train_val_test_split.json"
        splits[name] = load_split(split_path)
        meta_maps[name] = load_metadata(setup_cfg.resolved_paths()["metadata"])
        logger.info(
            f"  {name}: train={len(splits[name]['train'])}, "
            f"val={len(splits[name]['val'])}, test={len(splits[name]['test'])}"
        )
    
    # Infer hidden_dim from a single activation file — same for all datasets
    # since they share the same model.
    hidden_dim = _peek_hidden_dim(Path(datasets[0]["activations_dir"]), layer_idx)
    if hidden_dim is None:
        logger.error(
            f"Could not infer hidden_dim for {datasets[0]['name']} at layer {layer_idx}. "
            f"Skipping entire row."
        )
        return

    logger.info(f"  hidden_dim={hidden_dim}")

    # ------------------------------------------------------------------
    # Step 1 & 2: Nested loop — train on A, evaluate on all B (including A).
    # acc_matrix_raw stores [acc@p25, acc@p50, acc@p75, acc@p90] per cell.
    # ------------------------------------------------------------------
    _FRACS = ["p25", "p50", "p75", "p90"]
    acc_matrix_raw: Dict[str, Dict[str, List[float]]] = {n: {} for n in train_dataset_names}

    for train_ds in datasets:
        if train_ds["name"] not in train_dataset_names:
            continue
        train_name = train_ds["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Outer loop — training probe on: {train_name}")

        train_cfg = build_train_cfg(
            phase4_cfg, train_ds, run_name_prefix, results_dir, seed
        )
        model_path = Path(results_dir) / "models" / f"{run_name_prefix}_train_{train_name}" / f"probe_layer{layer_idx}.pth"
        if not model_path.exists() and train_name != "gpqa_diamond":
            train_one_layer(layer_idx, train_cfg, splits[train_name], meta_maps[train_name])

        # Instantiate probe and load the best checkpoint saved by train_one_layer.
        model = instantiate_probe(probe_type, hidden_dim, output_dim, train_cfg)
        load_checkpoint(model, train_cfg, layer_idx)
        model.to(device)
        model.eval()

        for eval_ds in datasets:
            eval_name = eval_ds["name"]
            logger.info(f"\n  Inner loop - train dataset: {train_name} — eval'ing on dataset: {eval_name}")

            eval_cfg = build_eval_cfg(train_cfg, eval_ds)
            samples = load_samples(
                layer_idx,
                splits[eval_name]["test"],
                eval_cfg,
                label_type,
                is_training_set=False,
            )

            if not samples:
                logger.warning(
                    f"  No test samples for eval_ds={eval_name} — recording NaN"
                )
                acc_matrix_raw[train_name][eval_name] = [float("nan")] * len(_FRACS)
                continue

            per_pos_accs = evaluate_accuracy_averaged_over_positions(
                model, samples, device, label_type
            )
            logger.info(
                f"  Accuracy [{train_name} → {eval_name}]: "
                + ", ".join(f"{f}={a*100:.2f}%" for f, a in zip(_FRACS, per_pos_accs))
            )
            acc_matrix_raw[train_name][eval_name] = per_pos_accs

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: Fan out into 5 accuracy matrices (one per position + avg)
    # ------------------------------------------------------------------
    def _extract_pos_matrix(pos_idx: Optional[int]) -> Dict[str, Dict[str, float]]:
        """Extract a scalar matrix for position pos_idx, or average if None."""
        mat: Dict[str, Dict[str, float]] = {}
        for tn in train_dataset_names:
            mat[tn] = {}
            for en in eval_dataset_names:
                vals = acc_matrix_raw[tn].get(en, [float("nan")] * len(_FRACS))
                if pos_idx is None:
                    valid = [v for v in vals if not np.isnan(v)]
                    mat[tn][en] = float(sum(valid) / len(valid)) if valid else float("nan")
                else:
                    mat[tn][en] = vals[pos_idx] if pos_idx < len(vals) else float("nan")
        return mat

    # Build the 5 accuracy matrices (one per position + avg)
    position_specs = [(f, i) for i, f in enumerate(_FRACS)] + [("avg", None)]
    matrix_sets = []
    for label, pos_idx in position_specs:
        acc_mat = _extract_pos_matrix(pos_idx)
        matrix_sets.append((label, acc_mat))

    # ------------------------------------------------------------------
    # Step 4: Print, save, and plot all 5 matrices.
    # ------------------------------------------------------------------
    for label, acc_mat in matrix_sets:
        print_matrix(
            acc_mat,
            train_dataset_names,
            eval_dataset_names,
            f"Phase 4 — Raw Accuracy Matrix (%) [{label}]",
        )

        acc_csv = summary_csv_path.with_name(f"{summary_csv_path.stem}_{label}.csv")
        save_matrix_csv(acc_mat, train_dataset_names, eval_dataset_names, acc_csv)

        plot_matrix(
            acc_mat,
            train_dataset_names,
            eval_dataset_names,
            f"Phase 4: Transfer Accuracy Matrix (%) [{label}]",
            plots_dir,
            f"accuracy_matrix_{label}",
            cmap="Blues",
        )

    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
