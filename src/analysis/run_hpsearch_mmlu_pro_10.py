"""Hyperparameter grid search for mmlu_pro_10 probe training.

Trains an AttentionProbe at a fixed layer for every combination of
(batch_size, learning_rate, weight_decay), evaluates on the validation set,
and reports a ranked results table. Saves each checkpoint plus a separate
'best' checkpoint.

Usage:
    python -m src.analysis.run_hpsearch_mmlu_pro_10 experiments/deepseek_r1_qwen_32b/hpsearch_mmlu_pro_10.yaml
"""

import argparse
import copy
import csv
import itertools
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from .experiment_config import DataConfig, ExperimentConfig, ProbeConfig, RunConfig, SetupConfig, ForcedAnswerConfig, CotMonitorConfig
from .run_probing import (
    AttentionProbe,
    ProbeDataset,
    collate,
    load_samples,
)
from .setup_data import setup_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HpSearchConfig:
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    learning_rates: List[float] = field(default_factory=lambda: [0.001, 0.005, 0.01])
    weight_decays: List[float] = field(default_factory=lambda: [0.001, 0.01])


def load_config(path: Path):
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return raw


def build_experiment_config(raw: dict, bs: int, lr: float, wd: float) -> ExperimentConfig:
    """Build an ExperimentConfig for one HP combination."""
    run_raw = raw["run"]
    data_raw = raw["data"]
    probe_raw = raw["probe"]

    run = RunConfig(
        run_name=str(run_raw["run_name"]),
        results_dir=Path(run_raw.get("results_dir", "results")),
        seed=int(run_raw.get("seed", 42)),
    )
    data = DataConfig(
        responses_dir=Path(data_raw["responses_dir"]),
        activations_dir=Path(data_raw["activations_dir"]),
        tokenizer_model=data_raw.get("tokenizer_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
        dataset_name=data_raw.get("dataset_name"),
        num_choices=int(data_raw.get("num_choices", 10)),
    )
    probe = ProbeConfig(
        num_layers=int(probe_raw["num_layers"]),
        enabled=True,
        train=True,
        eval=False,
        selected_layer=int(probe_raw.get("selected_layer", 61)),
        probe_type=str(probe_raw.get("probe_type", "attention")),
        label_type=str(probe_raw.get("label_type", "model_ans")),
        batch_size=bs,
        learning_rate=lr,
        weight_decay=wd,
        num_epochs=int(probe_raw.get("num_epochs", 10)),
        mlp_hidden_dim=int(probe_raw.get("mlp_hidden_dim", 32)),
        normalize_acts=bool(probe_raw.get("normalize_acts", True)),
        disable_tqdm=True,
    )
    setup = SetupConfig(enabled=True)
    forced = ForcedAnswerConfig(enabled=False)
    cot_monitor = CotMonitorConfig(enabled=False)

    return ExperimentConfig(run=run, data=data, setup=setup, probe=probe,
                            forced_answer=forced, cot_monitor=cot_monitor)


def train_and_eval(cfg: ExperimentConfig, split: Dict[str, List[str]]) -> Dict:
    """Train probe and return best val loss + val accuracy."""
    layer_idx = cfg.probe.selected_layer
    samples_train = load_samples(layer_idx, split["train"], cfg, cfg.probe.label_type, is_training_set=True)
    samples_val = load_samples(layer_idx, split["val"], cfg, cfg.probe.label_type, is_training_set=False)

    if not samples_train or not samples_val:
        raise RuntimeError("Empty train or val split")

    hidden_dim = samples_train[0]["activation"].shape[1]
    output_dim = cfg.data.num_choices
    model = AttentionProbe(hidden_dim, output_dim, torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        ProbeDataset(samples_train, cfg.probe.label_type, training=True),
        batch_size=cfg.probe.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        ProbeDataset(samples_val, cfg.probe.label_type, training=False),
        batch_size=cfg.probe.batch_size, shuffle=False, collate_fn=collate,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.probe.learning_rate,
                                  weight_decay=cfg.probe.weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.probe.num_epochs):
        model.train()
        for batch_inputs, batch_labels, lengths, _ in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs, lengths)
            loss = criterion(logits.float(), batch_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch_inputs, batch_labels, lengths, _ in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_inputs, lengths)
                loss = criterion(logits.float(), batch_labels.long())
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_count += batch_labels.size(0)
                val_loss_sum += loss.item() * batch_labels.size(0)

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else float("inf")
        val_acc = val_correct / val_count if val_count > 0 else 0.0

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "model_state": best_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HP grid search for mmlu_pro_10 probe.")
    parser.add_argument("config", type=Path, help="Path to hpsearch YAML config")
    args = parser.parse_args()

    raw = load_config(args.config)

    hps_raw = raw.get("hpsearch", {})
    hps = HpSearchConfig(
        batch_sizes=hps_raw.get("batch_sizes", [16, 32, 64]),
        learning_rates=hps_raw.get("learning_rates", [0.001, 0.005, 0.01]),
        weight_decays=hps_raw.get("weight_decays", [0.001, 0.01]),
    )

    # Build a base config just to run setup (split generation)
    base_cfg = build_experiment_config(raw, bs=16, lr=0.001, wd=0.01)
    logger.info("Running setup to generate train/val/test split and metadata")
    setup_run(base_cfg)

    # Load the split
    split_path = base_cfg.resolved_paths()["root"] / "train_val_test_split.json"
    with split_path.open() as f:
        split = json.load(f)
    logger.info(f"Split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

    results_dir = base_cfg.resolved_paths()["root"] / "hpsearch"
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(hps.batch_sizes, hps.learning_rates, hps.weight_decays))
    logger.info(f"Running {len(combos)} HP combinations")

    rows = []
    best_val_loss = float("inf")
    best_combo = None
    best_state = None
    layer_idx = int(raw["probe"].get("selected_layer", 61))

    for bs, lr, wd in combos:
        label = f"bs{bs}_lr{lr}_wd{wd}"
        logger.info(f"=== {label} ===")
        cfg = build_experiment_config(raw, bs=bs, lr=lr, wd=wd)

        result = train_and_eval(cfg, split)

        ckpt_path = results_dir / f"probe_layer{layer_idx}_{label}.pth"
        torch.save(result["model_state"], ckpt_path)
        logger.info(f"  val_loss={result['best_val_loss']:.4f}  val_acc={result['best_val_acc']*100:.2f}%  best_epoch={result['best_epoch']}")

        rows.append({
            "batch_size": bs,
            "learning_rate": lr,
            "weight_decay": wd,
            "best_val_loss": result["best_val_loss"],
            "best_val_acc": result["best_val_acc"],
            "best_epoch": result["best_epoch"],
            "checkpoint": str(ckpt_path),
        })

        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_combo = (bs, lr, wd)
            best_state = result["model_state"]

    # Sort by val loss ascending
    rows.sort(key=lambda r: r["best_val_loss"])

    # Print results table
    print("\n=== HP Search Results (sorted by val loss) ===")
    header = f"{'batch_size':>10}  {'lr':>8}  {'wd':>8}  {'val_loss':>10}  {'val_acc':>8}  {'best_epoch':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['batch_size']:>10}  {r['learning_rate']:>8.4f}  {r['weight_decay']:>8.4f}"
            f"  {r['best_val_loss']:>10.4f}  {r['best_val_acc']*100:>7.2f}%  {r['best_epoch']:>10}"
        )

    # Save CSV
    csv_path = results_dir / "hpsearch_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "learning_rate", "weight_decay",
                                               "best_val_loss", "best_val_acc", "best_epoch", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Results saved to {csv_path}")

    # Save best checkpoint
    if best_state is not None:
        best_path = results_dir / f"probe_layer{layer_idx}_best.pth"
        torch.save(best_state, best_path)
        bs, lr, wd = best_combo
        logger.info(f"Best config: batch_size={bs}, lr={lr}, wd={wd}  val_loss={best_val_loss:.4f}")
        logger.info(f"Best checkpoint saved to {best_path}")
        print(f"\nBest checkpoint: {best_path}")
        print(f"Best config: batch_size={bs}, lr={lr}, weight_decay={wd}, val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
