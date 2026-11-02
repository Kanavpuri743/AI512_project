import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset import CLASS_NAMES, IGNORE_INDEX, NightCityDataset
from engine import run_epoch
from model import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NightCity segmentation model.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(
    dataset: NightCityDataset, limit: Optional[int]
) -> torch.utils.data.Dataset:
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("Sample limits must be positive.")
    return Subset(dataset, range(min(limit, len(dataset))))


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Epochs and batch size must be positive.")
    if args.learning_rate <= 0 or args.num_workers < 0:
        raise ValueError("Learning rate must be positive and workers non-negative.")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("Height and width must be positive multiples of 32.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = NightCityDataset(
        args.data_root, "train", size=(args.height, args.width)
    )
    val_dataset = NightCityDataset(
        args.data_root, "val", size=(args.height, args.width)
    )
    train_loader = DataLoader(
        limit_dataset(train_dataset, args.max_train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        limit_dataset(val_dataset, args.max_val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Model(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4
    )
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_from}")
        checkpoint = torch.load(
            args.resume_from, map_location=device, weights_only=True
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"Unsupported resume checkpoint: {args.resume_from}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                checkpoint.get("val_metrics", {}).get("loss", float("inf")),
            )
        )
        print(f"Resuming from epoch {start_epoch - 1}: {args.resume_from}")

    print(
        f"Device={device}; train={len(train_loader.dataset)}; "
        f"val={len(val_loader.dataset)}; size={args.height}x{args.width}; "
        f"epochs={args.epochs}; lr={args.learning_rate}"
    )

    log_path = args.output_dir / "training_log.csv"
    log_mode = "a" if args.resume_from is not None and log_path.exists() else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=(
                "epoch",
                "train_loss",
                "train_miou",
                "val_loss",
                "val_miou",
                "learning_rate",
            ),
        )
        if log_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            train_stats = run_epoch(
                model, train_loader, optimizer, device, len(CLASS_NAMES)
            )
            val_stats = run_epoch(
                model, val_loader, None, device, len(CLASS_NAMES)
            )

            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_miou": train_stats["miou"],
                "val_loss": val_stats["loss"],
                "val_miou": val_stats["miou"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            log_file.flush()

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_stats,
                "val_metrics": val_stats,
                "best_val_loss": min(best_val_loss, val_stats["loss"]),
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "class_names": CLASS_NAMES,
                "ignore_index": IGNORE_INDEX,
            }
            torch.save(checkpoint, args.output_dir / "last_checkpoint.pt")
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                torch.save(checkpoint, args.output_dir / "best_checkpoint.pt")

            print(
                f"Epoch {epoch}/{args.epochs} "
                f"train_loss={train_stats['loss']:.4f} "
                f"train_mIoU={train_stats['miou']:.4f} "
                f"val_loss={val_stats['loss']:.4f} "
                f"val_mIoU={val_stats['miou']:.4f}"
            )

    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
