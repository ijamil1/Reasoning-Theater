"""Train a probe for mmlu_pro_10 with fixed hyperparameters.

Each run appends one row to probe_results.csv so results across different
hyperparameter settings can be compared over time.

Supported probe_type values (set in YAML under probe.probe_type):
  attention             — AttentionProbe (original paper architecture)
  causal_self_attention — CausalSelfAttentionProbe
  answer_choice         — AnswerChoiceProbe (requires data.model_checkpoint_dir)

Usage:
    python -m src.analysis.run_probe_mmlu_pro_10 experiments/deepseek_r1_qwen_32b/probe_mmlu_pro_10.yaml
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .experiment_config import DataConfig, ExperimentConfig, ProbeConfig, RunConfig, SetupConfig, ForcedAnswerConfig, CotMonitorConfig
from .run_probing import (
    AttentionProbe,
    AnswerChoiceProbe,
    CausalSelfAttentionProbe,
    ProbeDataset,
    collate,
    load_samples,
    load_samples_with_choices,
)
from .setup_data import setup_data as setup_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FIELDS = [
    "probe_type", "batch_size", "learning_rate", "weight_decay", "num_epochs",
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
        model_checkpoint_dir=Path(data_raw["model_checkpoint_dir"]) if data_raw.get("model_checkpoint_dir") else None,
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


def build_probe(probe_type: str, hidden_dim: int, output_dim: int, dtype: torch.dtype) -> torch.nn.Module:
    if probe_type == "attention":
        return AttentionProbe(hidden_dim, output_dim, dtype)
    elif probe_type == "causal_self_attention":
        return CausalSelfAttentionProbe(hidden_dim, output_dim, dtype)
    elif probe_type == "answer_choice":
        return AnswerChoiceProbe(hidden_dim, output_dim, dtype)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type!r}. Choose from: attention, causal_self_attention, answer_choice")


def load_embed_weights(cfg: ExperimentConfig) -> torch.Tensor:
    """Load embed_tokens.weight from model safetensors shards (CPU only)."""
    checkpoint_dir = cfg.data.model_checkpoint_dir
    if checkpoint_dir is None:
        raise ValueError("data.model_checkpoint_dir must be set in the YAML config for answer_choice probe")

    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors package required for answer_choice probe. Install with: pip install safetensors")

    shard_files = sorted(checkpoint_dir.glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No model*.safetensors found in {checkpoint_dir}")

    key = "model.embed_tokens.weight"
    for shard_path in shard_files:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if key in f.keys():
                logger.info(f"Loaded {key} from {shard_path.name}")
                return f.get_tensor(key)

    raise KeyError(f"'{key}' not found in any shard under {checkpoint_dir}")


def precompute_choice_embeddings(
    samples: List[Dict],
    tokenizer,
    embed_weight: torch.Tensor,
    num_choices: int,
) -> None:
    """Mean-pool input embeddings for each answer choice and store in sample meta.

    Adds meta["choice_embeddings"]: Tensor [num_choices, H] (bfloat16, CPU).
    Samples missing or short answer_choices lists are filled with zeros.
    """
    H = embed_weight.shape[1]
    n = len(samples)
    log_every = max(1, n // 10)
    for idx, sample in enumerate(samples):
        if idx % log_every == 0:
            logger.info(f"  Precomputing choice embeddings: {idx}/{n} ({100*idx//n}%)")
        choices = sample["meta"].get("answer_choices", [])
        choice_embs = []
        for i in range(num_choices):
            if i < len(choices):
                token_ids = tokenizer.encode(choices[i], add_special_tokens=False)
                if token_ids:
                    vecs = embed_weight[token_ids]          # [n_tokens, H]
                    emb = vecs.mean(dim=0).to(torch.bfloat16)
                else:
                    emb = torch.zeros(H, dtype=torch.bfloat16)
            else:
                emb = torch.zeros(H, dtype=torch.bfloat16)
            choice_embs.append(emb)
        sample["meta"]["choice_embeddings"] = torch.stack(choice_embs)  # [num_choices, H]


def collate_with_choices(batch):
    """Like collate, but also stacks choice_embeddings from meta."""
    activations, labels, metas = zip(*batch)
    max_len = max(act.shape[0] for act in activations)
    hidden_dim = activations[0].shape[1]
    dtype = activations[0].dtype
    padded = torch.zeros(len(activations), max_len, hidden_dim, dtype=dtype)
    lengths = []
    for i, act in enumerate(activations):
        seq_len = act.shape[0]
        padded[i, :seq_len] = act
        lengths.append(seq_len)
    label_tensor = torch.stack([torch.tensor(lbl) for lbl in labels])
    choice_embs = torch.stack([m["choice_embeddings"] for m in metas])  # [batch, num_choices, H]
    return padded, label_tensor, lengths, metas, choice_embs


def train_and_eval(cfg: ExperimentConfig, split: Dict[str, List[str]]) -> Dict:
    """Train probe and return best val loss + val accuracy."""
    layer_idx = cfg.probe.selected_layer
    probe_type = cfg.probe.probe_type
    is_answer_choice = probe_type == "answer_choice"

    loader_fn = load_samples_with_choices if is_answer_choice else load_samples
    samples_train = loader_fn(layer_idx, split["train"], cfg, cfg.probe.label_type, is_training_set=True)
    samples_val = loader_fn(layer_idx, split["val"], cfg, cfg.probe.label_type, is_training_set=False)

    if not samples_train or not samples_val:
        raise RuntimeError("Empty train or val split")

    hidden_dim = samples_train[0]["activation"].shape[1]
    output_dim = cfg.data.num_choices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_probe(probe_type, hidden_dim, output_dim, torch.bfloat16)
    model.to(device)

    if is_answer_choice:
        logger.info("Precomputing answer choice embeddings from model input embedding matrix...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_model)
        embed_weight = load_embed_weights(cfg)
        precompute_choice_embeddings(samples_train, tokenizer, embed_weight, output_dim)
        precompute_choice_embeddings(samples_val, tokenizer, embed_weight, output_dim)
        del embed_weight  # free CPU memory
        collate_fn = collate_with_choices
    else:
        collate_fn = collate

    train_loader = DataLoader(
        ProbeDataset(samples_train, cfg.probe.label_type, training=True),
        batch_size=cfg.probe.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ProbeDataset(samples_val, cfg.probe.label_type, training=False),
        batch_size=cfg.probe.batch_size, shuffle=False, collate_fn=collate_fn,
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
        for batch in train_loader:
            if is_answer_choice:
                batch_inputs, batch_labels, lengths, _, choice_embs = batch
                choice_embs = choice_embs.to(device)
            else:
                batch_inputs, batch_labels, lengths, _ = batch
                choice_embs = None
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs, lengths) if choice_embs is None else model(batch_inputs, lengths, choice_embs)
            loss = criterion(logits.float(), batch_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                if is_answer_choice:
                    batch_inputs, batch_labels, lengths, _, choice_embs = batch
                    choice_embs = choice_embs.to(device)
                else:
                    batch_inputs, batch_labels, lengths, _ = batch
                    choice_embs = None
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_inputs, lengths) if choice_embs is None else model(batch_inputs, lengths, choice_embs)
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
    parser = argparse.ArgumentParser(description="Train probe for mmlu_pro_10.")
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
    probe_type = cfg.probe.probe_type
    logger.info(f"Training: probe={probe_type}, layer={layer_idx}, bs={bs}, lr={lr}, wd={wd}, epochs={cfg.probe.num_epochs}")

    result = train_and_eval(cfg, split)

    results_dir = cfg.resolved_paths()["root"] / "probe_runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    label = f"{probe_type}_bs{bs}_lr{lr}_wd{wd}"
    ckpt_path = results_dir / f"probe_layer{layer_idx}_{label}.pth"
    torch.save(result["model_state"], ckpt_path)

    row = {
        "probe_type": probe_type,
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
