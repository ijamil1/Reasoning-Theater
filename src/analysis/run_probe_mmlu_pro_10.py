"""Train an AttentionProbe for mmlu_pro_10 with fixed hyperparameters.

Each run appends one row to probe_results.csv so results across different
hyperparameter settings can be compared over time.

Usage:
    python -m src.analysis.run_probe_mmlu_pro_10 experiments/deepseek_r1_qwen_32b/probe_mmlu_pro_10.yaml
"""

import argparse
import csv
import json
import logging
import sys
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
from .setup_data import setup_data as setup_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FIELDS = [
    "batch_size", "learning_rate", "weight_decay", "num_epochs",
    "best_val_loss", "best_val_acc", "best_epoch", "checkpoint",
]


def load_config(path: Path):
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    return raw


def build_experiment_config(raw: dict) -> ExperimentConfig:
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
        batch_size=int(probe_raw.get("batch_size", 32)),
        learning_rate=float(probe_raw.get("learning_rate", 0.001)),
        weight_decay=float(probe_raw.get("weight_decay", 0.001)),
        num_epochs=int(probe_raw.get("num_epochs", 20)),
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
        logger.info(f"  epoch {epoch+1}/{cfg.probe.num_epochs}  val_loss={avg_val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "model_state": best_state,
    }


def append_result(csv_path: Path, row: Dict) -> None:
    """Append one result row to the CSV, writing a header if the file is new."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AttentionProbe for mmlu_pro_10.")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    args = parser.parse_args()

    raw = load_config(args.config)
    cfg = build_experiment_config(raw)

    logger.info("Running setup to generate train/val/test split and metadata")
    setup_run(cfg)

    split_path = cfg.resolved_paths()["root"] / "train_val_test_split.json"
    with split_path.open() as f:
        split = json.load(f)
    logger.info(f"Split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

    layer_idx = cfg.probe.selected_layer
    bs, lr, wd = cfg.probe.batch_size, cfg.probe.learning_rate, cfg.probe.weight_decay
    logger.info(f"Training: layer={layer_idx}, bs={bs}, lr={lr}, wd={wd}, epochs={cfg.probe.num_epochs}")

    result = train_and_eval(cfg, split)

    results_dir = cfg.resolved_paths()["root"] / "probe_runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    label = f"bs{bs}_lr{lr}_wd{wd}"
    ckpt_path = results_dir / f"probe_layer{layer_idx}_{label}.pth"
    torch.save(result["model_state"], ckpt_path)

    row = {
        "batch_size": bs,
        "learning_rate": lr,
        "weight_decay": wd,
        "num_epochs": cfg.probe.num_epochs,
        "best_val_loss": result["best_val_loss"],
        "best_val_acc": result["best_val_acc"],
        "best_epoch": result["best_epoch"],
        "checkpoint": str(ckpt_path),
    }
    csv_path = cfg.resolved_paths()["root"] / "probe_results.csv"
    append_result(csv_path, row)

    logger.info(f"Best: val_loss={result['best_val_loss']:.4f}  val_acc={result['best_val_acc']*100:.2f}%  epoch={result['best_epoch']}")
    logger.info(f"Checkpoint: {ckpt_path}")
    logger.info(f"Results appended to: {csv_path}")


if __name__ == "__main__":
    main()
