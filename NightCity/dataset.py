from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)

# Working Cityscapes-style raw IDs inferred from the observed NightCity labels.
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
IGNORE_INDEX = 255


class NightCityDataset(Dataset):
    """Load NightCity RGB images and map raw label IDs to 19 class indices."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        size: Optional[Tuple[int, int]] = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split {split!r}; expected 'train' or 'val'.")

        self.root = Path(root)
        self.split = split
        self.size = size

        image_dir = self.root / "NightCity-images" / "images" / split
        label_dir = self.root / "NightCity-label" / "label" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

        image_paths = sorted(image_dir.glob("*.png"))
        label_paths = {
            path.name.removesuffix("_labelIds.png"): path
            for path in label_dir.glob("*_labelIds.png")
        }
        image_stems = {path.stem for path in image_paths}
        missing_labels = sorted(image_stems - label_paths.keys())
        orphaned_labels = sorted(label_paths.keys() - image_stems)
        if missing_labels or orphaned_labels:
            raise ValueError(
                f"Unmatched files in {split}: "
                f"images without labels={missing_labels[:10]}, "
                f"labels without images={orphaned_labels[:10]}"
            )
        if not image_paths:
            raise ValueError(f"No PNG images found in {image_dir}")

        self.samples = [(path, label_paths[path.stem]) for path in image_paths]
        self._class_lookup = np.full(256, IGNORE_INDEX, dtype=np.uint8)
        for class_index, raw_id in enumerate(RAW_CLASS_IDS):
            self._class_lookup[raw_id] = class_index

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            with Image.open(label_path) as label_file:
                label = label_file.convert("L")

                if image.size != label.size:
                    raise ValueError(
                        f"Image and label dimensions differ for {image_path.name}: "
                        f"{image.size} vs {label.size}"
                    )

                if self.size is not None:
                    height, width = self.size
                    image = image.resize(
                        (width, height), resample=Image.Resampling.BILINEAR
                    )
                    label = label.resize(
                        (width, height), resample=Image.Resampling.NEAREST
                    )

                image_array = np.asarray(image, dtype=np.uint8).copy()
                raw_label = np.asarray(label, dtype=np.uint8).copy()

        observed_ids = np.unique(raw_label)
        known_ids = np.asarray((*RAW_CLASS_IDS, 0, 3), dtype=np.uint8)
        unknown_ids = np.setdiff1d(observed_ids, known_ids)
        if unknown_ids.size:
            raise ValueError(
                f"Unexpected raw label IDs in {label_path}: "
                f"{unknown_ids.tolist()}"
            )

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        label_tensor = torch.from_numpy(self._class_lookup[raw_label].copy()).long()
        return image_tensor, label_tensor
