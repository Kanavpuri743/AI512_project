import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset import CLASS_NAMES, IGNORE_INDEX, NightCityDataset


PALETTE = np.asarray(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a NightCity image, mapped label, and overlay preview."
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "previews" / "nightcity_preview.png",
    )
    args = parser.parse_args()

    dataset = NightCityDataset(Path(__file__).parent / "data", args.split)
    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"Index {args.index} is outside split {args.split!r} "
            f"with {len(dataset)} samples."
        )

    image_tensor, target_tensor = dataset[args.index]
    image = (
        image_tensor.permute(1, 2, 0).mul(255).byte().numpy()
    )
    target = target_tensor.numpy()

    label_rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    valid = target != IGNORE_INDEX
    label_rgb[valid] = PALETTE[target[valid]]
    label_rgb[~valid] = (32, 32, 32)

    overlay = (
        image.astype(np.float32) * 0.55
        + label_rgb.astype(np.float32) * 0.45
    ).clip(0, 255).astype(np.uint8)

    panels = [
        Image.fromarray(image),
        Image.fromarray(label_rgb),
        Image.fromarray(overlay),
    ]
    panel_width, panel_height = panels[0].size
    header_height = 36
    legend_height = 126
    gap = 8
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + gap * 4, panel_height + header_height + legend_height),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    titles = ("RGB image", "Mapped labels (dark gray = ignored)", "Overlay")

    for column, (panel, title) in enumerate(zip(panels, titles)):
        x = gap + column * (panel_width + gap)
        draw.text((x, 8), title, fill=(0, 0, 0))
        canvas.paste(panel, (x, header_height))

    legend_top = header_height + panel_height + 8
    column_width = (canvas.width - 2 * gap) // 4
    for class_index, class_name in enumerate(CLASS_NAMES):
        column = class_index % 4
        row = class_index // 4
        x = gap + column * column_width
        y = legend_top + row * 22
        draw.rectangle((x, y + 2, x + 16, y + 18), fill=tuple(PALETTE[class_index]))
        draw.text((x + 22, y + 2), class_name, fill=(0, 0, 0))
    draw.rectangle(
        (gap + 3 * column_width, legend_top + 5 * 22 + 2,
         gap + 3 * column_width + 16, legend_top + 5 * 22 + 18),
        fill=(32, 32, 32),
    )
    draw.text(
        (gap + 3 * column_width + 22, legend_top + 5 * 22 + 2),
        "ignored",
        fill=(0, 0, 0),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Preview saved to: {args.output.resolve()}")
    print(f"Sample: {dataset.samples[args.index][0].name} ({args.split})")
    print(f"Image and label size: {panel_width} x {panel_height}")


if __name__ == "__main__":
    main()
