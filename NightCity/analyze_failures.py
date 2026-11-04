import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = {
    19: "traffic_light",
    25: "rider",
    27: "truck",
    28: "bus",
    32: "motorcycle",
    33: "bicycle",
    22: "terrain",
}
RAW_CLASS_IDS = (
    7,
    8,
    11,
    12,
    13,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
)
CLASS_COLORS = np.asarray(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)
CLASS_COLOR_BY_ID = {
    raw_id: CLASS_COLORS[index] for index, raw_id in enumerate(RAW_CLASS_IDS)
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Select low-IoU validation examples for chosen classes and save "
            "input/ground-truth/prediction comparison panels."
        )
    )
    parser.add_argument(
        "--data-root", type=Path, default=project_root / "NightCity" / "data"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_root / "nightcity_baseline_300_evaluation",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "failure_case_candidates"
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-class-pixels", type=int, default=10)
    return parser.parse_args()


def colorize_label(label: np.ndarray, size: tuple) -> Image.Image:
    colored = np.full((*label.shape, 3), 32, dtype=np.uint8)
    for raw_id, color in CLASS_COLOR_BY_ID.items():
        colored[label == raw_id] = color
    return Image.fromarray(colored).resize(size, Image.Resampling.NEAREST)


def make_panel(candidate: Dict[str, object], output_path: Path) -> None:
    image_path = candidate["image_path"]
    label_path = candidate["label_path"]
    prediction_path = candidate["prediction_path"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    ground_truth = np.asarray(Image.open(label_path).convert("L"))
    prediction = np.asarray(Image.open(prediction_path).convert("L"))

    panel_size = (512, 256)
    panels = [
        image.resize(panel_size, Image.Resampling.BILINEAR),
        colorize_label(ground_truth, panel_size),
        colorize_label(prediction, panel_size),
    ]
    title = (
        f"{candidate['class']} | {candidate['image']} | "
        f"IoU={candidate['image_iou']:.3f} | "
        f"GT pixels={candidate['gt_pixels']}"
    )
    canvas = Image.new("RGB", (1536, 290), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, fill="black")
    for index, (panel, panel_title) in enumerate(
        zip(panels, ("Input", "Ground truth", "Prediction"))
    ):
        x = index * panel_size[0]
        draw.text((x + 8, 24), panel_title, fill="black")
        canvas.paste(panel, (x, 34))
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.min_class_pixels <= 0:
        raise ValueError("--top-k and --min-class-pixels must be positive.")

    image_dir = args.data_root / "NightCity-images" / "images" / args.split
    label_dir = args.data_root / "NightCity-label" / "label" / args.split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not args.predictions.is_dir():
        raise FileNotFoundError(
            f"Prediction directory not found: {args.predictions}"
        )

    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    valid_ids = np.asarray(RAW_CLASS_IDS, dtype=np.uint8)
    candidates: Dict[int, List[Dict[str, object]]] = {
        raw_id: [] for raw_id in CLASS_NAMES
    }

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}_labelIds.png"
        prediction_path = (
            args.predictions / f"{image_path.stem}_pred_labelIds.png"
        )
        if not label_path.is_file():
            raise FileNotFoundError(f"Matching label not found: {label_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Prediction not found for {image_path.name}: {prediction_path}"
            )

        with Image.open(label_path) as source:
            ground_truth = np.asarray(source.convert("L"))
        with Image.open(prediction_path) as source:
            prediction = np.asarray(source.convert("L"))
        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"Ground-truth/prediction size mismatch for {image_path.name}: "
                f"{ground_truth.shape} vs {prediction.shape}"
            )

        valid_pixels = np.isin(ground_truth, valid_ids)
        for raw_id, class_name in CLASS_NAMES.items():
            gt_mask = (ground_truth == raw_id) & valid_pixels
            gt_pixels = int(gt_mask.sum())
            if gt_pixels < args.min_class_pixels:
                continue

            pred_mask = (prediction == raw_id) & valid_pixels
            union = int((gt_mask | pred_mask).sum())
            intersection = int((gt_mask & pred_mask).sum())
            image_iou = intersection / union if union else float("nan")

            wrong_predictions = prediction[gt_mask & (prediction != raw_id)]
            wrong_predictions = wrong_predictions[
                np.isin(wrong_predictions, valid_ids)
            ]
            dominant_wrong = "none"
            if wrong_predictions.size:
                values, counts = np.unique(wrong_predictions, return_counts=True)
                dominant_wrong = str(int(values[counts.argmax()]))

            candidates[raw_id].append(
                {
                    "class": class_name,
                    "raw_id": raw_id,
                    "image": image_path.name,
                    "gt_pixels": gt_pixels,
                    "image_iou": image_iou,
                    "most_common_wrong_raw_id": dominant_wrong,
                    "image_path": image_path,
                    "label_path": label_path,
                    "prediction_path": prediction_path,
                }
            )

    selected: List[Dict[str, object]] = []
    for class_candidates in candidates.values():
        selected.extend(
            sorted(
                class_candidates,
                key=lambda item: (item["image_iou"], item["image"]),
            )[: args.top_k]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in selected:
        filename = (
            f"{candidate['class']}_{Path(candidate['image']).stem}.png"
        )
        make_panel(candidate, args.output_dir / filename)

    summary_path = args.output_dir / "candidate_summary.csv"
    fields = (
        "class",
        "raw_id",
        "image",
        "gt_pixels",
        "image_iou",
        "most_common_wrong_raw_id",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        for candidate in selected:
            writer.writerow({field: candidate[field] for field in fields})

    print(f"Scanned {len(image_paths)} {args.split} images.")
    print(f"Selected {len(selected)} candidate examples.")
    print(f"Comparison panels: {args.output_dir}")
    print(f"Candidate summary: {summary_path}")
    print(
        "Scene conditions such as glare or low illumination are not detected "
        "automatically; review the saved panels manually."
    )


if __name__ == "__main__":
    main()
