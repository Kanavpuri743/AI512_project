import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image

from dataset import (
    CLASS_NAMES,
    IGNORE_INDEX,
    RAW_CLASS_IDS,
    NightCityDataset,
)
from engine import metrics_from_confusion, update_confusion_from_prediction
from model import Model
from visualize_dataset import PALETTE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiclass NightCity inference and evaluation."
    )
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "outputs" / "best_checkpoint.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "predictions"
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def load_weights(model: Model, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict)


def profile_model(
    model: Model, height: int, width: int, device: torch.device
) -> tuple[int, int]:
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError(
            "FLOPs profiling requires THOP. Install it in this environment with "
            "`python -m pip install thop`."
        ) from error

    sample = torch.zeros(1, 3, height, width, device=device)
    model.eval()
    with torch.inference_mode():
        macs, _ = profile(model, inputs=(sample,), verbose=False)
    return int(macs), sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative.")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("Height and width must be positive multiples of 32.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("Sample limit must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NightCityDataset(
        args.data_root, args.split, size=(args.height, args.width)
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = Model(num_classes=len(CLASS_NAMES)).to(device)
    load_weights(model, args.checkpoint, device)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_macs, parameter_count = profile_model(
        model, args.height, args.width, device
    )

    confusion = torch.zeros(
        len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64, device=device
    )
    elapsed_seconds = 0.0
    num_images = 0
    sample_offset = 0

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            logits = model(images)
            predictions = logits.argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds += time.perf_counter() - start

            update_confusion_from_prediction(
                confusion, predictions, targets, len(CLASS_NAMES), IGNORE_INDEX
            )

            for batch_index in range(predictions.shape[0]):
                sample_index = sample_offset + batch_index
                if isinstance(dataset, Subset):
                    source_path = dataset.dataset.samples[dataset.indices[sample_index]][0]
                else:
                    source_path = dataset.samples[sample_index][0]

                prediction = predictions[batch_index].cpu().numpy()
                raw_ids = np.asarray(RAW_CLASS_IDS, dtype=np.uint8)[prediction]
                label_path = args.output_dir / f"{source_path.stem}_pred_labelIds.png"
                color_path = args.output_dir / f"{source_path.stem}_pred_color.png"
                Image.fromarray(raw_ids, mode="L").save(label_path)
                Image.fromarray(PALETTE[prediction]).save(color_path)
                num_images += 1
            sample_offset += predictions.shape[0]

    metrics = metrics_from_confusion(confusion)
    metrics["split"] = args.split
    metrics["num_images"] = num_images
    metrics["checkpoint"] = str(args.checkpoint.resolve())
    metrics["parameter_count"] = parameter_count
    metrics["thop_macs"] = profile_macs
    metrics["thop_estimated_flops"] = profile_macs * 2
    metrics["flops_note"] = (
        "THOP estimate computed as 2 x MACs; functional operations in "
        "CustomMamba may not be fully counted."
    )
    metrics["input_resolution"] = [args.height, args.width]
    metrics["device"] = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    metrics["inference_seconds"] = elapsed_seconds
    metrics["latency_ms_per_image"] = (
        elapsed_seconds * 1000 / num_images if num_images else float("nan")
    )
    metrics["fps"] = num_images / elapsed_seconds if elapsed_seconds else float("nan")
    metrics["class_names"] = list(CLASS_NAMES)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(json_safe(metrics), indent=2), encoding="utf-8"
    )
    print(f"Device: {device}")
    print(f"Evaluated images: {num_images}")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"Mean pixel accuracy: {metrics['mean_pixel_accuracy']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predicted label and color maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
