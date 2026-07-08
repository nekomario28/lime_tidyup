#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (255, 60, 60),
    (60, 180, 255),
    (70, 220, 90),
    (255, 190, 60),
    (190, 90, 255),
    (255, 90, 190),
    (80, 255, 220),
]


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"image path must contain an images directory: {image_path}") from exc
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def draw_label(image_path: Path, output_path: Path) -> None:
    label_path = label_path_for(image_path)
    if not label_path.exists():
        raise FileNotFoundError(f"label not found: {label_path}")

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = ImageFont.load_default()

    for line_no, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue

        values = line.split()
        if (len(values) - 5) % 3 != 0:
            raise ValueError(f"invalid YOLO pose label at {label_path}:{line_no}")

        cls = values[0]
        cx, cy, bw, bh = map(float, values[1:5])
        keypoints = list(map(float, values[5:]))

        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=2)
        draw.text((x1 + 4, y1 + 4), f"class {cls}", fill=(255, 255, 255), font=font)

        for idx in range(0, len(keypoints), 3):
            point_no = idx // 3
            x_norm, y_norm, visible = keypoints[idx : idx + 3]
            if visible <= 0:
                continue

            x = x_norm * width
            y = y_norm * height
            color = COLORS[point_no % len(COLORS)]
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)
            draw.text((x + 7, y - 7), str(point_no), fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw YOLO pose labels on images.")
    parser.add_argument(
        "source",
        nargs="?",
        default="datasets/Cube_Pose_Estimation.v1-test1.yolov8/test/images",
        help="Image file or image directory. Default: test/images",
    )
    parser.add_argument(
        "--out",
        default="pose_label_viz",
        help="Output directory. Default: pose_label_viz",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Maximum number of images when source is a directory. Use 0 for all. Default: 12",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.out)
    images = iter_images(source)
    if args.limit and source.is_dir():
        images = images[: args.limit]

    if not images:
        raise SystemExit(f"no images found: {source}")

    for image_path in images:
        rel_name = image_path.name
        out_path = output_dir / rel_name
        draw_label(image_path, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
