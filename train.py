"""Train and evaluate OCSFlow on one benchmark scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from configuration import build_model, load_config, resolve_runtime_config
from data.dataset_loader import load_dataset
from data.preprocessing import normalize_train_pixels
from data.split import stratified_random_split
from evaluation.evaluator import evaluate_model
from training.seed import set_reproducible_seed
from training.trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OCSFlow")
    parser.add_argument("--dataset", required=True, choices=("IP", "PU", "KSC", "HC"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "ocsflow.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.dataset, args.seed)
    set_reproducible_seed(args.seed)
    dataset = load_dataset(args.dataset, config["data_root"])
    masks = stratified_random_split(
        dataset.ground_truth,
        train_per_class=int(config["split"]["train_per_class"]),
        val_per_class=int(config["split"]["val_per_class"]),
        seed=args.seed,
    )
    normalized, _ = normalize_train_pixels(
        dataset.image,
        masks.train_mask,
        eps=float(config["normalization"]["eps"]),
    )
    config = resolve_runtime_config(
        config,
        image_shape=tuple(int(value) for value in normalized.shape),
        num_classes=dataset.num_classes,
    )
    device_name = str(config.get("device", "auto"))
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    image = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    ground_truth = torch.from_numpy(dataset.ground_truth).long().to(device)
    train_mask = torch.from_numpy(masks.train_mask).bool().to(device)
    val_mask = torch.from_numpy(masks.val_mask).bool().to(device)
    test_mask = torch.from_numpy(masks.test_mask).bool().to(device)
    model = build_model(config)
    training_result = train_model(
        model, image, ground_truth, train_mask, val_mask, config
    )
    sampling_generator = torch.Generator(device=device)
    sampling_generator.manual_seed(args.seed + 7919)
    metrics = evaluate_model(
        model,
        image,
        ground_truth,
        train_mask,
        test_mask,
        noise=training_result["evaluation_noise"],
        remote_sampling_generator=sampling_generator,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
