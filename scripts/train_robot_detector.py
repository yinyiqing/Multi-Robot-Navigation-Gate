#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TD3"))

from robot_perception.dataset import list_shards, load_shard
from robot_perception.metrics import detection_metrics, select_validation_threshold
from robot_perception.models import LocalRobotDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Train the single-frame lidar robot detector.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--offset-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def batch_indices(labels, negative_ratio, rng):
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    negative_count = min(len(negatives), max(len(positives) * negative_ratio, 16))
    if negative_count < len(negatives):
        negatives = rng.choice(negatives, size=negative_count, replace=False)
    indices = np.concatenate((positives, negatives))
    rng.shuffle(indices)
    return indices


def train_epoch(model, paths, optimizer, args, rng):
    model.train()
    losses = []
    shuffled_paths = list(paths)
    rng.shuffle(shuffled_paths)
    for path in shuffled_paths:
        shard = load_shard(path)
        labels = shard["labels"].astype(np.float32)
        indices = batch_indices(labels, args.negative_ratio, rng)
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start : start + args.batch_size]
            if len(batch) == 0:
                continue
            patches = torch.from_numpy(shard["patches"][batch].astype(np.float32)).to(args.device)
            targets = torch.from_numpy(labels[batch]).to(args.device)
            offsets = torch.from_numpy(
                shard["center_offsets"][batch].astype(np.float32)
            ).to(args.device)
            logits, predicted_offsets = model(patches)
            classification_loss = F.binary_cross_entropy_with_logits(logits, targets)
            positive_mask = targets > 0.5
            offset_loss = (
                F.smooth_l1_loss(predicted_offsets[positive_mask], offsets[positive_mask])
                if torch.any(positive_mask)
                else logits.sum() * 0.0
            )
            loss = classification_loss + args.offset_loss_weight * offset_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, paths, batch_size, device):
    model.eval()
    probabilities = []
    labels = []
    visible_count = 0
    offset_errors = []
    for path in paths:
        shard = load_shard(path)
        shard_labels = shard["labels"].astype(np.uint8)
        visible_count += int(shard["visible_robot_count"])
        for start in range(0, len(shard_labels), batch_size):
            stop = start + batch_size
            patches = torch.from_numpy(
                shard["patches"][start:stop].astype(np.float32)
            ).to(device)
            logits, predicted_offsets = model(patches)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            batch_labels = shard_labels[start:stop]
            labels.append(batch_labels)
            positive = batch_labels == 1
            if np.any(positive):
                truth = shard["center_offsets"][start:stop][positive].astype(np.float32)
                prediction = predicted_offsets.cpu().numpy()[positive]
                offset_errors.extend(np.linalg.norm(prediction - truth, axis=1).tolist())
    if not probabilities:
        raise ValueError("validation shards contain no candidate patches")
    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
        visible_count,
        float(np.mean(offset_errors)) if offset_errors else None,
    )


def save_checkpoint(path, model, optimizer, epoch, args, metrics, offset_mae):
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": {"input_channels": 3, "hidden_dim": 128},
            "threshold": metrics.threshold,
            "validation_metrics": metrics.to_dict(),
            "validation_offset_mae_m": offset_mae,
            "training_config": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "negative_ratio": args.negative_ratio,
                "offset_loss_weight": args.offset_loss_weight,
                "seed": args.seed,
            },
        },
        path,
    )


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.negative_ratio < 1:
        raise ValueError("epochs, batch size, and negative ratio must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    train_paths = list_shards(args.train_dir)
    validation_paths = list_shards(args.validation_dir)
    model = LocalRobotDetector().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_paths, optimizer, args, rng)
        probabilities, labels, visible_count, offset_mae = evaluate(
            model, validation_paths, args.batch_size, args.device
        )
        metrics = select_validation_threshold(probabilities, labels, visible_count)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": metrics.to_dict(),
            "validation_offset_mae_m": offset_mae,
        }
        history.append(record)
        save_checkpoint(
            args.output_dir / "latest.pt",
            model,
            optimizer,
            epoch,
            args,
            metrics,
            offset_mae,
        )
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                args,
                metrics,
                offset_mae,
            )
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
            handle.write("\n")
        print(
            "epoch=%d loss=%.5f threshold=%.2f precision=%.3f recall=%.3f "
            "fpr=%.3f proposal_recall=%.3f offset_mae=%s pass=%s"
            % (
                epoch,
                train_loss,
                metrics.threshold,
                metrics.precision,
                metrics.recall,
                metrics.fpr,
                metrics.proposal_recall,
                "n/a" if offset_mae is None else "%.3f" % offset_mae,
                metrics.meets_entry_criteria,
            )
        )


if __name__ == "__main__":
    main()
