import argparse
import csv
import fcntl
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .experiment_config import ExperimentConfig
from .setup_data import STEP_HEADERS, TOKEN_HEADERS
from .utils import (
    DELIMITERS,
    compute_reasoning_token_start,
    is_gpt_oss_model,
    load_forced_step_boundaries,
    split_response_text,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


CHOICE_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
CHOICE_TO_IDX_10 = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9}


class AttentionProbe(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, mlp: bool = False, mlp_hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = mlp
        self.q = torch.nn.Linear(in_features, 1, dtype=dtype)
        if self.mlp:
            self.v_up = torch.nn.Linear(in_features, mlp_hidden_dim, dtype=dtype)
            self.v_relu = torch.nn.ReLU()
            self.v_down = torch.nn.Linear(mlp_hidden_dim, out_features, dtype=dtype)
        else:
            self.v = torch.nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, lengths: Sequence[int] | None = None) -> torch.Tensor:
        attn_logits = self.q(x).squeeze(-1)
        if lengths is not None:
            mask = torch.zeros_like(attn_logits)
            for i, length in enumerate(lengths):
                if length > 0:
                    mask[i, :length] = 1
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
        if self.mlp:
            v_up_out = self.v_up(x)
            aggregated = torch.sum(attn_weights.unsqueeze(-1) * v_up_out, dim=1)
            activated = self.v_relu(aggregated)
            pooled = self.v_down(activated)
        else:
            values = self.v(x)
            pooled = torch.sum(attn_weights.unsqueeze(-1) * values, dim=1)
        return pooled


class RecencyWeightedLinearProbe(torch.nn.Module):
    """Linear probe using exponentially decaying recency-weighted pooling.

    Token t in a prefix of length L gets weight exp(-decay * (L-1-t)), so
    the most recent token always has weight 1 (unnormalized) and earlier
    tokens decay geometrically into the past.
    """
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, decay: float = 0.02) -> None:
        super().__init__()
        self.decay = decay
        self.linear = torch.nn.Linear(in_features, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, lengths: Sequence[int] | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        t_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)  # [S]
        if lengths is not None:
            lengths_t = x.new_tensor(lengths, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        else:
            lengths_t = torch.full((batch_size, 1), seq_len, dtype=torch.float32, device=x.device)
        # log_w[b, t] = -decay * (L_b - 1 - t); max per row = 0 (at t = L_b - 1)
        log_w = -self.decay * (lengths_t - 1 - t_idx)           # [B, S]
        log_w = log_w.masked_fill(t_idx >= lengths_t, float("-inf"))  # zero out padding
        w = torch.exp(log_w)                                     # [B, S]
        w = w / w.sum(dim=1, keepdim=True)                       # [B, S], normalised
        pooled = (w.unsqueeze(-1) * x.float()).sum(dim=1)        # [B, H]
        return self.linear(pooled.to(x.dtype))


class RecencyWeightedMLPProbe(torch.nn.Module):
    """Two-layer MLP probe using exponentially decaying recency-weighted pooling."""
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, decay: float = 0.02, mlp_hidden_dim: int = 32) -> None:
        super().__init__()
        self.decay = decay
        self.up = torch.nn.Linear(in_features, mlp_hidden_dim, dtype=dtype)
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.Linear(mlp_hidden_dim, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, lengths: Sequence[int] | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        t_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)  # [S]
        if lengths is not None:
            lengths_t = x.new_tensor(lengths, dtype=torch.float32).unsqueeze(1)  # [B, 1]
        else:
            lengths_t = torch.full((batch_size, 1), seq_len, dtype=torch.float32, device=x.device)
        log_w = -self.decay * (lengths_t - 1 - t_idx)           # [B, S]
        log_w = log_w.masked_fill(t_idx >= lengths_t, float("-inf"))
        w = torch.exp(log_w)                                     # [B, S]
        w = w / w.sum(dim=1, keepdim=True)                       # [B, S], normalised
        pooled = (w.unsqueeze(-1) * x.float()).sum(dim=1)        # [B, H]
        return self.down(self.relu(self.up(pooled.to(x.dtype))))


class AttentionMLPProbe(torch.nn.Module):
    """Attention probe where attention is the pooling mechanism and MLP is the read-out.

    Differs from AttentionProbe(mlp=True): here the MLP is applied to the
    attention-pooled hidden state (post-aggregation), not to each token's
    value projection before aggregation.
    """
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype, mlp_hidden_dim: int = 32) -> None:
        super().__init__()
        self.q = torch.nn.Linear(in_features, 1, dtype=dtype)
        self.up = torch.nn.Linear(in_features, mlp_hidden_dim, dtype=dtype)
        self.relu = torch.nn.ReLU()
        self.down = torch.nn.Linear(mlp_hidden_dim, out_features, dtype=dtype)

    def forward(self, x: torch.Tensor, lengths: Sequence[int] | None = None) -> torch.Tensor:
        attn_logits = self.q(x).squeeze(-1)
        if lengths is not None:
            mask = torch.zeros_like(attn_logits)
            for i, length in enumerate(lengths):
                if length > 0:
                    mask[i, :length] = 1
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)
        return self.down(self.relu(self.up(pooled)))


class ProbeDataset(Dataset):
    def __init__(self, samples: Sequence[Dict], label_type: str, training: bool = False) -> None:
        self.samples = list(samples)
        self.label_type = label_type
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        activation = self.samples[idx]["activation"]
        # During training, randomly truncate sequences to prevent memorization
        # Each sample is randomly truncated to a random length between 1 and the full sequence length.
        if self.training:
            seq_len = activation.shape[0]
            end_idx = random.randint(1, seq_len)
            activation = activation[:end_idx]
        return activation, self.samples[idx]["label"], self.samples[idx]["meta"]


def collate(batch):
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
    return padded, label_tensor, lengths, metas


def load_split(split_path: Path) -> Dict[str, List[str]]:
    with split_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(metadata_path: Path) -> Dict[str, int]:
    csv.field_size_limit(sys.maxsize)
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["question_hash"]: int(row["question_idx"]) for row in reader}


def compute_normalization_stats(layer_idx: int, train_hashes: Sequence[str], cfg: ExperimentConfig) -> Dict[str, float]:
    """Compute mean and std for a layer across all training samples.

    Streams one activation file at a time to avoid accumulating all tensors in
    RAM.  Uses the sum / sum-of-squares identity so only three scalars need to
    be kept between files.
    """
    logger.info(f"Computing normalization stats for layer {layer_idx} on {len(train_hashes)} training samples")
    count = 0
    running_sum = 0.0
    running_sum_sq = 0.0

    for idx, qhash in enumerate(train_hashes):
        if (idx + 1) % 500 == 0:
            logger.info(f"  Processing {idx+1}/{len(train_hashes)} samples for stats")
        activation_dir = cfg.data.activations_dir / f"layer_{layer_idx}" / qhash
        if not activation_dir.exists():
            continue
        activation_files = [f for f in os.listdir(activation_dir) if f.endswith(".pt")]
        if not activation_files:
            continue
        activation_path = activation_dir / activation_files[0]
        activation = torch.load(activation_path, map_location="cpu").float()
        count += activation.numel()
        running_sum += activation.sum().item()
        running_sum_sq += (activation ** 2).sum().item()

    if count == 0:
        raise RuntimeError(f"No activations found for layer {layer_idx} in training set")

    mean = running_sum / count
    std = (running_sum_sq / count - mean ** 2) ** 0.5

    if std == 0.0:
        raise RuntimeError(f"Zero std for layer {layer_idx} activations - cannot normalize")

    logger.info(f"Layer {layer_idx} stats: mean={mean:.6f}, std={std:.6f}")
    return {"mean": mean, "std": std}


def load_normalization_stats(cfg: ExperimentConfig) -> Dict[int, Dict[str, float]]:
    """Load normalization stats from file, or from norm_stats_run_root/reuse_run_root if specified."""
    stats_path = cfg.resolved_paths()["normalization_stats"]

    if cfg.probe.norm_stats_run_root and cfg.probe.normalize_acts and not cfg.probe.recompute_norm_stats:
        shared_stats_path = Path(cfg.probe.norm_stats_run_root).resolve() / "normalization_stats.json"
        if shared_stats_path.exists():
            logger.info(f"Loading normalization stats from norm_stats_run_root: {shared_stats_path}")
            with shared_stats_path.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
            return {int(k): v for k, v in stats.items()}
        else:
            raise FileNotFoundError(
                f"normalize_acts=True and norm_stats_run_root={cfg.probe.norm_stats_run_root}, "
                f"but normalization_stats.json not found at {shared_stats_path}"
            )

    if cfg.probe.reuse_run_root and cfg.probe.normalize_acts and not cfg.probe.recompute_norm_stats:
        reuse_root = Path(cfg.probe.reuse_run_root).resolve()
        reuse_stats_path = reuse_root / "normalization_stats.json"
        if reuse_stats_path.exists():
            logger.info(f"Loading normalization stats from reuse_run_root: {reuse_stats_path}")
            with reuse_stats_path.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
            return {int(k): v for k, v in stats.items()}
        else:
            raise FileNotFoundError(
                f"normalize_acts=True and reuse_run_root={cfg.probe.reuse_run_root}, "
                f"but normalization_stats.json not found at {reuse_stats_path}"
            )
    
    lock_path = stats_path.with_suffix(".lock")
    if stats_path.exists():
        logger.info(f"Loading normalization stats from {stats_path}")
        with lock_path.open("w") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                with stats_path.open("r", encoding="utf-8") as handle:
                    stats = json.load(handle)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return {int(k): v for k, v in stats.items()}

    return {}


def save_normalization_stats(cfg: ExperimentConfig, stats: Dict[int, Dict[str, float]]) -> None:
    """Save normalization stats to file, merging with existing stats under file lock."""
    stats_path = cfg.resolved_paths()["normalization_stats"]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stats_path.with_suffix(".lock")

    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            existing_stats = {}
            if stats_path.exists():
                try:
                    with stats_path.open("r", encoding="utf-8") as handle:
                        existing_json = json.load(handle)
                    existing_stats = {int(k): v for k, v in existing_json.items()}
                except (json.JSONDecodeError, ValueError, KeyError):
                    logger.warning(f"Could not parse existing normalization stats, overwriting")
                    existing_stats = {}

            merged_stats = {**existing_stats, **stats}
            stats_json = {str(k): v for k, v in merged_stats.items()}
            with stats_path.open("w", encoding="utf-8") as handle:
                json.dump(stats_json, handle, indent=2)
            logger.info(f"Saved normalization stats to {stats_path} (layers: {sorted(merged_stats.keys())})")
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def load_samples(layer_idx: int, hashes: Sequence[str], cfg: ExperimentConfig, label_type: str, is_training_set: bool = False) -> List[Dict]:
    logger.info(f"Loading samples for layer {layer_idx}, label_type={label_type}, num_hashes={len(hashes)}")
    samples = []
    skipped_missing = 0
    skipped_invalid_label = 0
    skipped_no_activation = 0
    total_hashes = len(hashes)
    log_interval = max(100, total_hashes // 20)  # Log every 5% or every 100, whichever is larger
    
    norm_stats = None
    if cfg.probe.normalize_acts:
        all_stats = load_normalization_stats(cfg)
        if layer_idx in all_stats:
            norm_stats = all_stats[layer_idx]
            logger.info(f"Using normalization stats for layer {layer_idx}: mean={norm_stats['mean']:.6f}, std={norm_stats['std']:.6f}")
        elif is_training_set and cfg.probe.train:
            logger.info(f"Normalization stats not found for layer {layer_idx}, computing from training set")
            norm_stats = compute_normalization_stats(layer_idx, hashes, cfg)
            all_stats[layer_idx] = norm_stats
            save_normalization_stats(cfg, all_stats)
        else:
            raise RuntimeError(
                    f"normalize_acts=True but no stats found for layer {layer_idx} and not loading training set. "
                    f"Stats must be pre-computed or reuse_run_root must be set."
                )
    
    tokenizer = None
    if cfg.probe.eval:
        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_model)
    tokens_reconstructed = 0
    tokens_mismatch_count = 0
    max_mismatch_warnings = 3  # Only warn for first few mismatches
    
    for idx, qhash in enumerate(hashes):
        if (idx + 1) % log_interval == 0 or idx == 0:
            logger.info(f"  Processing hash {idx+1}/{total_hashes} (loaded {len(samples)}, skipped {skipped_missing + skipped_no_activation})")
        response_path = cfg.data.responses_dir / f"{qhash}.json"
        activation_dir = cfg.data.activations_dir / f"layer_{layer_idx}" / qhash
        if not response_path.exists() or not activation_dir.exists():
            skipped_missing += 1
            continue
        with response_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        activation_files = [f for f in os.listdir(activation_dir) if f.endswith(".pt")]
        if not activation_files:
            skipped_no_activation += 1
            continue
        activation_path = activation_dir / activation_files[0]
        activation = torch.load(activation_path)
        
        if norm_stats is not None:
            activation = (activation - norm_stats["mean"]) / norm_stats["std"]
        
        model_ans = payload.get("parsed_answer")
        correct_ans = payload.get("correct_answer")
        active_choice_map = CHOICE_TO_IDX_10 if cfg.data.num_choices == 10 else CHOICE_TO_IDX
        valid_choices = set(active_choice_map.keys())
        if label_type == "model_ans":
            if model_ans not in valid_choices:
                skipped_invalid_label += 1
                if skipped_invalid_label <= max_mismatch_warnings:
                    logger.warning(f"Skipping {qhash}: invalid model_ans '{model_ans}'")
                continue
            label = active_choice_map[model_ans]
        elif label_type == "correct_ans":
            if correct_ans not in valid_choices:
                skipped_invalid_label += 1
                if skipped_invalid_label <= max_mismatch_warnings:
                    logger.warning(f"Skipping {qhash}: invalid correct_ans '{correct_ans}'")
                continue
            label = active_choice_map[correct_ans]
        elif label_type == "model_correct":
            if model_ans not in valid_choices or correct_ans not in valid_choices:
                skipped_invalid_label += 1
                if skipped_invalid_label <= max_mismatch_warnings:
                    logger.warning(f"Skipping {qhash}: invalid model_correct labels model_ans='{model_ans}', correct_ans='{correct_ans}'")
                continue
            label = int(model_ans == correct_ans)
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")

        if cfg.probe.randomize_labels:
            label = random.randint(0, cfg.data.num_choices - 1)

        response_tokens = payload.get("complete_rollout_tokens") or []

        if not response_tokens and cfg.probe.eval:
            system_prompt = payload.get("system_prompt", "")
            formatted_question = payload.get("formatted_question", "")
            reasoning_text = payload.get("reasoning", "")
            content_text = payload.get("full_response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_question},
                {"role": "assistant", "content": reasoning_text + content_text}
            ]
            chat_template_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            response_tokens = tokenizer.encode(chat_template_text, add_special_tokens=False)
            tokens_reconstructed += 1
            
            activation_seq_len = activation.shape[0]
            if len(response_tokens) != activation_seq_len:
                tokens_mismatch_count += 1
                if tokens_mismatch_count <= max_mismatch_warnings:
                    logger.warning(
                        f"Token length mismatch for {qhash}: "
                        f"reconstructed={len(response_tokens)}, activation={activation_seq_len}, "
                        f"difference={activation_seq_len - len(response_tokens)}"
                    )
        
        samples.append(
            {
                "activation": activation,
                "label": label,
                "meta": {
                    "question_hash": qhash,
                    "response_tokens": response_tokens,
                    "response_strs": payload.get("complete_rollout_tokenized", []),
                    "question_idx": payload.get("question_idx"),
                    "system_prompt": payload.get("system_prompt", ""),
                    "formatted_question": payload.get("formatted_question", ""),
                },
            }
        )
    
    if tokens_reconstructed > 0:
        logger.info(f"Reconstructed tokens for {tokens_reconstructed} samples")
    if tokens_mismatch_count > 0:
        logger.warning(f"Token length mismatches detected in {tokens_mismatch_count} samples")
    if skipped_invalid_label > 0:
        logger.warning(f"Skipped {skipped_invalid_label} samples due to invalid labels")
    logger.info(f"Loaded {len(samples)} samples (skipped {skipped_missing} missing paths, {skipped_no_activation} no activation files)")
    return samples


def load_checkpoint(model: AttentionProbe, cfg: ExperimentConfig, layer_idx: int) -> None:
    """Load checkpoint into model from checkpoint path, reuse_run_root, or default."""
    ckpt = cfg.probe.checkpoint
    if ckpt is None:
        if cfg.probe.reuse_run_root is not None:
            reuse_root = Path(cfg.probe.reuse_run_root).resolve()
            ckpt = reuse_root.parent / "models" / reuse_root.name / f"probe_layer{layer_idx}.pth"
            logger.info(f"Using reuse_run_root: {reuse_root}")
        else:
            ckpt = cfg.resolved_paths()["models"] / f"probe_layer{layer_idx}.pth"
    else:
        ckpt = Path(ckpt).resolve()
    
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    
    logger.info(f"Loading checkpoint from {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    logger.info("Checkpoint loaded successfully")


def train_one_layer(layer_idx: int, cfg: ExperimentConfig, split: Dict[str, List[str]], meta_idx_map: Dict[str, int]) -> None:
    logger.info(f"=== Training probe for layer {layer_idx} ===")
    logger.info(f"Config: train={cfg.probe.train}, eval={cfg.probe.eval}, label_type={cfg.probe.label_type}")
    logger.info(f"Split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    
    samples_train = []
    samples_val = []
    samples_test = []
    source_samples = None
    if cfg.probe.train:
        samples_train = load_samples(layer_idx, split["train"], cfg, cfg.probe.label_type, is_training_set=True)
        samples_val = load_samples(layer_idx, split["val"], cfg, cfg.probe.label_type, is_training_set=False)
        source_samples = samples_train

    if cfg.probe.eval:
        samples_test = load_samples(layer_idx, split["test"], cfg, cfg.probe.label_type, is_training_set=False)
        source_samples = samples_test
    
    if cfg.probe.train and (not samples_train):
        raise RuntimeError(f"No samples for layer {layer_idx}: train={len(samples_train)}")
    if cfg.probe.eval and not samples_test:
        raise RuntimeError(f"No eval samples for layer {layer_idx}: test={len(samples_test)}")

    if source_samples is None:
        raise RuntimeError(f"No samples available for layer {layer_idx}")
    hidden_dim = source_samples[0]["activation"].shape[1]
    output_dim = 1 if cfg.probe.label_type == "model_correct" else cfg.data.num_choices
    logger.info(f"Model architecture: hidden_dim={hidden_dim}, output_dim={output_dim}, probe_type={cfg.probe.probe_type}")

    probe_type = cfg.probe.probe_type
    if probe_type == "recency_linear":
        model = RecencyWeightedLinearProbe(hidden_dim, output_dim, torch.bfloat16, cfg.probe.recency_decay)
    elif probe_type == "recency_mlp":
        model = RecencyWeightedMLPProbe(hidden_dim, output_dim, torch.bfloat16, cfg.probe.recency_decay, cfg.probe.mlp_hidden_dim)
    elif probe_type == "attention_mlp":
        model = AttentionMLPProbe(hidden_dim, output_dim, torch.bfloat16, cfg.probe.mlp_hidden_dim)
    else:  # "attention" (default)
        model = AttentionProbe(hidden_dim, output_dim, torch.bfloat16, False, cfg.probe.mlp_hidden_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    if cfg.probe.train:
        if cfg.probe.reuse_run_root is not None:
            logger.info("Fine-tuning mode: loading checkpoint before training")
            load_checkpoint(model, cfg, layer_idx)
        else:
            logger.info("Training from scratch (no checkpoint specified)")
        
        logger.info(f"Starting training: {cfg.probe.num_epochs} epochs, batch_size={cfg.probe.batch_size}, lr={cfg.probe.learning_rate}")
        train_loader = DataLoader(ProbeDataset(samples_train, cfg.probe.label_type, training=True), batch_size=cfg.probe.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(ProbeDataset(samples_val, cfg.probe.label_type, training=True), batch_size=cfg.probe.batch_size, shuffle=False, collate_fn=collate)
        logger.info(f"DataLoaders created: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
        
        if cfg.probe.label_type == "model_correct":
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.probe.learning_rate, weight_decay=cfg.probe.weight_decay)
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        
        for epoch in range(cfg.probe.num_epochs):
            logger.info(f"Epoch {epoch+1}/{cfg.probe.num_epochs}")
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_count = 0
            for batch_idx, (batch_inputs, batch_labels, lengths, _) in enumerate(train_loader):
                batch_inputs = batch_inputs.to(device=device)
                batch_labels = batch_labels.to(device=device)
                logits = model(batch_inputs, lengths)
                if cfg.probe.label_type == "model_correct":
                    logits_flat = logits.view(-1)
                    targets = batch_labels.view(-1).float()
                    loss = criterion(logits_flat, targets)
                    preds = (torch.sigmoid(logits_flat) > 0.5).to(targets.dtype)
                    train_correct += (preds == targets).sum().item()
                    count = targets.numel()
                else:
                    loss = criterion(logits.float(), batch_labels.long())
                    preds = torch.argmax(logits, dim=1)
                    train_correct += (preds == batch_labels).sum().item()
                    count = batch_labels.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * count
                train_count += count
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, loss={loss.item():.4f}")
            
            avg_train_loss = train_loss / train_count if train_count > 0 else 0.0
            train_acc = (train_correct / train_count * 100.0) if train_count > 0 else 0.0
            logger.info(f"  Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.2f}%")
            
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_count = 0
            with torch.no_grad():
                for batch_inputs, batch_labels, lengths, _ in val_loader:
                    batch_inputs = batch_inputs.to(device=device)
                    batch_labels = batch_labels.to(device=device)
                    logits = model(batch_inputs, lengths)
                    if cfg.probe.label_type == "model_correct":
                        logits_flat = logits.view(-1)
                        targets = batch_labels.view(-1).float()
                        loss = criterion(logits_flat, targets)
                        preds = (torch.sigmoid(logits_flat) > 0.5).to(targets.dtype)
                        val_correct += (preds == targets).sum().item()
                        count = targets.numel()
                    else:
                        loss = criterion(logits.float(), batch_labels.long())
                        preds = torch.argmax(logits, dim=1)
                        val_correct += (preds == batch_labels).sum().item()
                        count = batch_labels.size(0)
                    val_loss += loss.item() * count
                    val_count += count
            
            avg_val_loss = val_loss / val_count if val_count > 0 else float("inf")
            val_acc = (val_correct / val_count * 100.0) if val_count > 0 else 0.0
            logger.info(f"  Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.2f}%")
            
            if val_count and val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                logger.info(f"  New best val loss: {avg_val_loss:.4f}, val acc: {val_acc:.2f}% (epoch {best_epoch})")
        
        if best_state is not None:
            logger.info(f"Loading best model from epoch {best_epoch}")
            model.load_state_dict(best_state)
        else:
            logger.warning("No best state found, using final model")
        
        models_dir = cfg.resolved_paths()["models"]
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"probe_layer{layer_idx}.pth"
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved successfully")
    else:
        logger.info("Skipping training, loading checkpoint for evaluation")
        load_checkpoint(model, cfg, layer_idx)

    if cfg.probe.eval:
        logger.info(f"Starting evaluation on {len(samples_test)} test samples")
        test_loader = DataLoader(ProbeDataset(samples_test, cfg.probe.label_type, training=False), batch_size=cfg.probe.batch_size, shuffle=False, collate_fn=collate)
        logger.info(f"Test DataLoader created: {len(test_loader)} batches")
        write_eval_outputs(layer_idx, model, test_loader, cfg, meta_idx_map)
        logger.info("Evaluation complete")

    model.cpu()
    torch.cuda.empty_cache()
    logger.info("GPU memory released")


def write_eval_outputs(layer_idx: int, model: torch.nn.Module, loader: DataLoader, cfg: ExperimentConfig, meta_idx_map: Dict[str, int]) -> None:
    logger.info(f"Writing evaluation outputs for layer {layer_idx}")
    paths = cfg.resolved_paths()
    logger.info(f"Loading tokenizer: {cfg.data.tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_model)
    device = next(model.parameters()).device
    weight_dtype = next(model.parameters()).dtype
    logger.info(f"Model device: {device}")
    model.eval()

    use_forced_boundaries = is_gpt_oss_model(cfg)
    forced_boundaries = {}
    if use_forced_boundaries:
        forced_boundaries = load_forced_step_boundaries(cfg)
        if forced_boundaries:
            logger.info(f"Using forced step boundaries for GPT-OSS model ({len(forced_boundaries)} questions)")
        else:
            logger.warning("GPT-OSS model but no forced step boundaries found, falling back to delimiter-based splitting")
            use_forced_boundaries = False

    dataset = loader.dataset
    total_samples = len(dataset)
    logger.info(f"Processing {total_samples} samples individually (no padding)")
    
    questions_processed = 0
    correct_predictions = 0
    total_predictions = 0
    per_question_accuracies = []
    with torch.no_grad():
        for global_idx in range(total_samples):
            if (global_idx + 1) % 100 == 0 or global_idx == 0:
                logger.info(f"Processing sample {global_idx + 1}/{total_samples}")
            
            activation, label, meta = dataset[global_idx]
            full_seq_len = activation.shape[0]
            
            if full_seq_len == 0:
                logger.warning(f"Skipping sample {global_idx}: seq_len is 0")
                continue
            
            qhash = meta["question_hash"]
            question_idx = meta_idx_map.get(qhash, -1)
            response_tokens = meta.get("response_tokens", [])
            response_strs = meta.get("response_strs", [])
            
            if not response_tokens:
                logger.warning(f"Skipping {qhash}: no response_tokens")
                continue
            
            activation_device = activation.to(device=device, dtype=weight_dtype)

            if isinstance(model, (RecencyWeightedLinearProbe, RecencyWeightedMLPProbe)):
                # Recursive recency-weighted cumsum: weighted_sum[t] = alpha*weighted_sum[t-1] + h[t]
                alpha = math.exp(-model.decay)
                act_fp32 = activation_device.to(torch.float32)
                weighted_sum = torch.zeros(act_fp32.shape[1], dtype=torch.float32, device=device)
                weight_sum = 0.0
                pooled_list = []
                for t in range(full_seq_len):
                    weighted_sum = alpha * weighted_sum + act_fp32[t]
                    weight_sum = alpha * weight_sum + 1.0
                    pooled_t = (weighted_sum / (weight_sum + 1e-8)).unsqueeze(0).to(weight_dtype)
                    if isinstance(model, RecencyWeightedMLPProbe):
                        out_t = model.down(model.relu(model.up(pooled_t))).to(torch.float32)
                    else:
                        out_t = model.linear(pooled_t).to(torch.float32)
                    pooled_list.append(out_t)
                pooled = torch.cat(pooled_list, dim=0)
            elif isinstance(model, AttentionMLPProbe):
                # Attention cumsum for pooling, then MLP read-out on pooled hidden state
                q_logits = model.q(activation_device).squeeze(-1)
                q_fp32 = q_logits.to(torch.float32)
                act_fp32 = activation_device.to(torch.float32)
                max_global = torch.max(q_fp32)
                exp_logits = torch.exp(q_fp32 - max_global)
                denom = torch.cumsum(exp_logits, dim=0)
                weighted = torch.cumsum(exp_logits.unsqueeze(-1) * act_fp32, dim=0)
                eps = 1e-8
                pooled_raw = (weighted / (denom.unsqueeze(-1) + eps)).to(weight_dtype)
                pooled = model.down(model.relu(model.up(pooled_raw))).to(torch.float32)
            else:
                # AttentionProbe (with optional per-token MLP value projection)
                q_logits = model.q(activation_device).squeeze(-1)
                if model.mlp:
                    value_hidden = model.v_up(activation_device)
                    value_hidden = model.v_relu(value_hidden)
                    v_proj = model.v_down(value_hidden)
                else:
                    v_proj = model.v(activation_device)

                q_fp32 = q_logits.to(torch.float32)
                v_fp32 = v_proj.to(torch.float32)

                max_global = torch.max(q_fp32)
                exp_logits = torch.exp(q_fp32 - max_global)
                denom = torch.cumsum(exp_logits, dim=0)
                weighted = torch.cumsum(exp_logits.unsqueeze(-1) * v_fp32, dim=0)
                eps = 1e-8
                pooled = weighted / (denom.unsqueeze(-1) + eps)
            
            if cfg.probe.label_type == "model_correct":
                logits_all = pooled.squeeze(-1)
                probs_all = torch.sigmoid(logits_all)
                probs_all_cpu = probs_all.to(torch.float32).cpu()
                preds_all_cpu = (probs_all > 0.5).to(torch.long).cpu()
            else:
                probs_all = torch.softmax(pooled, dim=-1)
                probs_all_cpu = probs_all.to(torch.float32).cpu()
                preds_all_cpu = torch.argmax(probs_all, dim=-1).cpu()
            
            system_prompt = meta.get("system_prompt", "")
            formatted_question = meta.get("formatted_question", "")
            reasoning_start = compute_reasoning_token_start(system_prompt, formatted_question, tokenizer)

            reasoning_tokens = response_tokens[reasoning_start:] if reasoning_start < len(response_tokens) else response_tokens

            qhash_boundaries = forced_boundaries.get(qhash) if use_forced_boundaries else None

            if qhash_boundaries:
                num_forced_steps = len(qhash_boundaries)

                full_reasoning_text = tokenizer.decode(reasoning_tokens, skip_special_tokens=True)

                boundary_token_indices = [0]
                for boundary in qhash_boundaries:
                    if boundary == 0:
                        continue
                    if boundary >= len(full_reasoning_text):
                        boundary_token_indices.append(len(reasoning_tokens))
                    else:
                        prefix_text = full_reasoning_text[:boundary]
                        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                        boundary_token_indices.append(len(prefix_tokens))

                step_texts = [""]
                prev_pos = 0
                for boundary in qhash_boundaries:
                    if boundary > 0:
                        step_texts.append(full_reasoning_text[prev_pos:boundary])
                        prev_pos = boundary
                if prev_pos < len(full_reasoning_text):
                    step_texts.append(full_reasoning_text[prev_pos:])
                    boundary_token_indices.append(len(reasoning_tokens))

                total_steps = len(boundary_token_indices)

                token_to_step = {}
                last_context_token_idx = reasoning_start - 1 if reasoning_start > 0 else -1
                for t_idx in range(len(response_tokens)):
                    if t_idx < last_context_token_idx:
                        token_to_step[t_idx] = -1
                    elif t_idx == last_context_token_idx and last_context_token_idx >= 0:
                        token_to_step[t_idx] = 0
                    elif t_idx < reasoning_start:
                        token_to_step[t_idx] = -1
                    else:
                        reasoning_t_idx = t_idx - reasoning_start
                        step_idx = 0
                        for i in range(1, len(boundary_token_indices)):
                            if reasoning_t_idx < boundary_token_indices[i]:
                                step_idx = i
                                break
                            step_idx = i + 1
                        token_to_step[t_idx] = min(step_idx, total_steps - 1)
            else:
                step_end_indices, step_token_sequences = split_response_text(reasoning_tokens, tokenizer, DELIMITERS)
                step_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in step_token_sequences] or [""]

                token_to_step = {}
                for t_idx in range(len(response_tokens)):
                    if t_idx < reasoning_start:
                        token_to_step[t_idx] = -1
                    else:
                        reasoning_t_idx = t_idx - reasoning_start
                        ptr = 0
                        while ptr < len(step_end_indices) and reasoning_t_idx >= step_end_indices[ptr]:
                            ptr += 1
                        token_to_step[t_idx] = min(ptr, len(step_end_indices) - 1) if step_end_indices else 0

            question_correct = 0
            question_total = 0
            for token_idx in range(full_seq_len):
                pred = int(preds_all_cpu[token_idx].item())
                is_correct = (pred == label)
                if is_correct:
                    correct_predictions += 1
                    question_correct += 1
                total_predictions += 1
                question_total += 1

            if question_total > 0:
                per_question_accuracies.append(question_correct / question_total)

            token_csv = paths["token_level"] / f"{qhash}.csv"
            step_csv = paths["step_level"] / f"{qhash}.csv"
            questions_processed += 1

            rows_written = 0
            token_csv.parent.mkdir(parents=True, exist_ok=True)
            with token_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=TOKEN_HEADERS, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                for token_idx in range(full_seq_len):
                    if cfg.probe.label_type == "model_correct":
                        prob = float(probs_all_cpu[token_idx].item())
                        prob_vector = [1.0 - prob, prob]
                        prediction_display = "correct" if prob > 0.5 else "incorrect"
                    else:
                        probs_token = probs_all_cpu[token_idx]
                        prob_vector = [float(val) for val in probs_token.tolist()]
                        pred_idx = int(preds_all_cpu[token_idx].item())
                        prediction_display = ["A", "B", "C", "D"][pred_idx] if pred_idx < 4 else str(pred_idx)
                    if token_idx < len(response_tokens):
                        token_id = response_tokens[token_idx]
                        token_piece = tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False)
                        if isinstance(token_piece, str):
                            token_text = tokenizer.convert_tokens_to_string([token_piece]) or token_piece
                        else:
                            token_text = str(token_piece)
                    elif token_idx < len(response_strs):
                        token_text = str(response_strs[token_idx])
                    else:
                        token_text = ""
                    step_idx = token_to_step.get(token_idx, -1)
                    writer.writerow(
                        {
                            "question_hash": str(qhash),
                            "question_idx": str(question_idx),
                            "layer_idx": str(layer_idx),
                            "token_idx": int(token_idx),
                            "step_idx": str(step_idx),
                            "token_text": str(token_text) if token_text else "",
                            "decoder_type": "probe",
                            "decoder_pred": str(prediction_display),
                            "decoder_output": json.dumps(prob_vector),
                        }
                    )
                    rows_written += 1

            if rows_written == 0:
                logger.warning(f"No rows written for {qhash} (seq_len={full_seq_len}, response_tokens={len(response_tokens)})")

            step_outputs = {}
            for token_idx in range(min(full_seq_len, len(response_tokens))):
                step_idx = token_to_step.get(token_idx, -1)
                if step_idx < 0:
                    continue
                if cfg.probe.label_type == "model_correct":
                    prob = float(probs_all_cpu[token_idx].item())
                    prob_vector = [1.0 - prob, prob]
                else:
                    prob_vector = [float(val) for val in probs_all_cpu[token_idx].tolist()]
                step_outputs.setdefault(step_idx, []).append(prob_vector)

            step_csv.parent.mkdir(parents=True, exist_ok=True)
            new_step_rows = []
            for step_idx, step_text in enumerate(step_texts):
                vectors = step_outputs.get(step_idx) or []
                if not vectors:
                    continue
                avg = torch.tensor(vectors[-1], dtype=torch.float32)
                avg_list = avg.tolist()
                if cfg.probe.label_type == "model_correct":
                    prob = float(avg_list[1]) if len(avg_list) > 1 else float(avg_list[0])
                    prediction_display = "correct" if prob > 0.5 else "incorrect"
                else:
                    pred_idx = int(torch.argmax(avg).item())
                    prediction_display = ["A", "B", "C", "D"][pred_idx] if pred_idx < 4 else str(pred_idx)
                new_step_rows.append(
                    {
                        "question_hash": str(qhash),
                        "question_idx": str(question_idx),
                        "layer_idx": str(layer_idx),
                        "step_idx": str(step_idx),
                        "step_text": str(step_text) if step_text else "",
                        "decoder_type": "probe",
                        "decoder_pred": str(prediction_display),
                        "decoder_output": json.dumps(avg_list),
                        "forced_completion": "",
                    }
                )

            lock_path = step_csv.with_suffix(".lock")
            with lock_path.open("w") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    existing_rows = []
                    if step_csv.exists():
                        with step_csv.open("r", newline="", encoding="utf-8") as rhandle:
                            reader = csv.DictReader(rhandle)
                            existing_rows = [row for row in reader if row.get("decoder_type") != "probe"]
                    merged_rows = existing_rows + new_step_rows
                    merged_rows.sort(key=lambda r: (int(r["step_idx"]), int(r["layer_idx"])))
                    with step_csv.open("w", newline="", encoding="utf-8") as whandle:
                        writer = csv.DictWriter(whandle, fieldnames=STEP_HEADERS, quoting=csv.QUOTE_MINIMAL)
                        writer.writeheader()
                        writer.writerows(merged_rows)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    
    if total_predictions > 0:
        test_acc_micro = (correct_predictions / total_predictions) * 100.0
        logger.info(f"Test accuracy (micro): {test_acc_micro:.2f}% ({correct_predictions}/{total_predictions})")
    else:
        logger.warning("No predictions made, cannot calculate test accuracy")

    if per_question_accuracies:
        test_acc_macro = sum(per_question_accuracies) / len(per_question_accuracies) * 100.0
        logger.info(f"Test accuracy (macro): {test_acc_macro:.2f}% (averaged over {len(per_question_accuracies)} questions)")

    logger.info(f"Evaluation outputs written: processed {questions_processed} questions")


def evaluate_accuracy(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_type: str = "model_ans",
) -> float:
    """Return classification accuracy on data_loader without writing any CSVs."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, batch_labels, lengths, _ in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_inputs, lengths)
            if label_type == "model_correct":
                logits_flat = logits.view(-1)
                targets = batch_labels.view(-1).float()
                preds = (torch.sigmoid(logits_flat) > 0.5).to(targets.dtype)
                correct += (preds == targets).sum().item()
                total += targets.numel()
            else:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)
    return correct / total if total > 0 else 0.0


def _predict_all_positions(model: torch.nn.Module, activation_device: torch.Tensor) -> torch.Tensor:
    """Return logits at every prefix position without writing any CSVs.

    Uses the same cumsum / recursive trick as write_eval_outputs.
    Returns a float32 tensor of shape [T, C].
    """
    weight_dtype = next(model.parameters()).dtype
    eps = 1e-8

    if isinstance(model, (RecencyWeightedLinearProbe, RecencyWeightedMLPProbe)):
        alpha = math.exp(-model.decay)
        act_fp32 = activation_device.to(torch.float32)
        T, H = act_fp32.shape
        weighted_sum = torch.zeros(H, dtype=torch.float32, device=activation_device.device)
        weight_sum = 0.0
        outs = []
        for t in range(T):
            weighted_sum = alpha * weighted_sum + act_fp32[t]
            weight_sum = alpha * weight_sum + 1.0
            pooled_t = (weighted_sum / (weight_sum + eps)).unsqueeze(0).to(weight_dtype)
            if isinstance(model, RecencyWeightedMLPProbe):
                out_t = model.down(model.relu(model.up(pooled_t)))
            else:
                out_t = model.linear(pooled_t)
            outs.append(out_t.to(torch.float32))
        return torch.cat(outs, dim=0)  # [T, C]

    if isinstance(model, AttentionMLPProbe):
        q_fp32 = model.q(activation_device).squeeze(-1).to(torch.float32)
        act_fp32 = activation_device.to(torch.float32)
        exp_logits = torch.exp(q_fp32 - q_fp32.max())
        denom = torch.cumsum(exp_logits, dim=0)
        weighted = torch.cumsum(exp_logits.unsqueeze(-1) * act_fp32, dim=0)
        pooled_raw = (weighted / (denom.unsqueeze(-1) + eps)).to(weight_dtype)
        return model.down(model.relu(model.up(pooled_raw))).to(torch.float32)  # [T, C]

    # AttentionProbe (mlp=False or mlp=True)
    q_fp32 = model.q(activation_device).squeeze(-1).to(torch.float32)
    if model.mlp:
        v_proj = model.v_down(model.v_relu(model.v_up(activation_device)))
    else:
        v_proj = model.v(activation_device)
    v_fp32 = v_proj.to(torch.float32)
    exp_logits = torch.exp(q_fp32 - q_fp32.max())
    denom = torch.cumsum(exp_logits, dim=0)
    weighted = torch.cumsum(exp_logits.unsqueeze(-1) * v_fp32, dim=0)
    return (weighted / (denom.unsqueeze(-1) + eps))  # [T, C]


_EVAL_FRACTIONS = [0.25, 0.50, 0.75, 0.90]


def evaluate_accuracy_averaged_over_positions(
    model: torch.nn.Module,
    samples: List[Dict],
    device: torch.device,
    label_type: str = "model_ans",
) -> List[float]:
    """Return per-position accuracies (averaged over samples) for four relative positions.

    For each sample the probe is evaluated at the token positions closest to
    25 %, 50 %, 75 %, and 90 % of the full reasoning trace using the cumsum /
    recursive trick.  Returns one accuracy per percentile, each averaged across
    all questions.  The caller is responsible for averaging across positions.
    """
    model.eval()
    weight_dtype = next(model.parameters()).dtype
    n = len(_EVAL_FRACTIONS)
    correct = [0] * n
    total = [0] * n

    with torch.no_grad():
        for sample in samples:
            activation = sample["activation"].to(device=device, dtype=weight_dtype)
            label = sample["label"]
            T = activation.shape[0]
            if T == 0:
                continue

            logits_all = _predict_all_positions(model, activation)  # [T, C]
            preds = torch.argmax(logits_all, dim=-1)                 # [T]

            for i, frac in enumerate(_EVAL_FRACTIONS):
                idx = min(int(frac * T), T - 1)
                correct[i] += int(preds[idx].item() == label)
                total[i] += 1

    return [correct[i] / total[i] if total[i] > 0 else float("nan") for i in range(n)]


def run_probing(cfg: ExperimentConfig, layer_override: int | None = None) -> None:
    logger.info("=== Starting probe training/evaluation ===")
    paths = cfg.resolved_paths()
    logger.info(f"Run root: {paths['root']}")
    
    split_path = paths["root"] / "train_val_test_split.json"
    logger.info(f"Loading split from {split_path}")
    split = load_split(split_path)

    logger.info(f"Loading metadata from {paths['metadata']}")
    meta_idx_map = load_metadata(paths["metadata"])
    logger.info(f"Loaded metadata for {len(meta_idx_map)} questions")

    logger.info(f"Setting random seed: {cfg.run.seed}")
    random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)
    torch.cuda.manual_seed_all(cfg.run.seed)

    if layer_override is not None:
        layer = layer_override
        logger.info(f"Using layer override: {layer}")
    elif cfg.probe.selected_layer != -1:
        layer = cfg.probe.selected_layer
        logger.info(f"Using selected_layer from config: {layer}")
    else:
        raise RuntimeError("Provide --layer or set probe.selected_layer in the config (use -1 for all layers)")

    train_one_layer(layer, cfg, split, meta_idx_map)
    logger.info("=== Probe training/evaluation complete ===")


def main():
    parser = argparse.ArgumentParser(description="Run probe training/eval for early decoders")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML")
    parser.add_argument("--layer", type=int, help="Override to run a single layer")
    args = parser.parse_args()
    
    logger.info(f"Loading config from {args.config}")
    cfg = ExperimentConfig.from_yaml(args.config)
    logger.info(f"Config loaded: run_name={cfg.run.run_name}, results_dir={cfg.run.results_dir}")
    
    try:
        run_probing(cfg, layer_override=args.layer)
    except Exception as e:
        logger.exception(f"Error during probe training/evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
