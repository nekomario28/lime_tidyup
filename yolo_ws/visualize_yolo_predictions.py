#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


COLORS = [
    (255, 60, 60),
    (60, 180, 255),
    (70, 220, 90),
    (255, 190, 60),
    (190, 90, 255),
    (255, 90, 190),
    (80, 255, 220),
]


def iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in source.rglob("*") if path.suffix.lower() in suffixes)


def draw_result(image_path: Path, result, output_path: Path, kpt_conf: float) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    boxes = [] if result.boxes is None else result.boxes.xyxy.tolist()
    classes = [] if result.boxes is None else result.boxes.cls.tolist()
    confs = [] if result.boxes is None else result.boxes.conf.tolist()
    keypoints_xy = [] if result.keypoints is None else result.keypoints.xy.tolist()
    keypoints_data = [] if result.keypoints is None else result.keypoints.data.tolist()

    for det_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 255), width=2)
        label = f"{int(classes[det_idx])} {confs[det_idx]:.2f}" if det_idx < len(confs) else str(det_idx)
        draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))

        if det_idx >= len(keypoints_xy):
            continue

        for point_no, point in enumerate(keypoints_xy[det_idx]):
            point_score = 1.0
            if det_idx < len(keypoints_data) and point_no < len(keypoints_data[det_idx]):
                if len(keypoints_data[det_idx][point_no]) >= 3:
                    point_score = keypoints_data[det_idx][point_no][2]
            if point_score < kpt_conf:
                continue

            x, y = point
            color = COLORS[point_no % len(COLORS)]
            radius = 5
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)
            draw.text((x + 7, y - 7), str(point_no), fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw YOLO pose predictions on images.")
    parser.add_argument(
        "source",
        nargs="?",
        default="datasets/Cube_Pose_Estimation.v1-test1.yolov8/test/images",
        help="Image file or image directory. Default: test/images",
    )
    parser.add_argument("--model", default="runs/pose/train-2/weights/best.pt", help="YOLO pose model path.")
    parser.add_argument("--out", default="pose_predict_viz", help="Output directory. Default: pose_predict_viz")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence. Default: 0.3")
    parser.add_argument("--kpt-conf", type=float, default=0.0, help="Keypoint confidence threshold. Default: 0.0")
    parser.add_argument("--limit", type=int, default=12, help="Max images for directory input. Use 0 for all.")
    args = parser.parse_args()

    source = Path(args.source)
    images = iter_images(source)
    if args.limit and source.is_dir():
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"no images found: {source}")

    model = YOLO(args.model)
    for image_path in images:
        result = model.predict(source=str(image_path), conf=args.conf, save=False, verbose=False)[0]
        output_path = Path(args.out) / image_path.name
        draw_result(image_path, result, output_path, args.kpt_conf)
        shape = None if result.keypoints is None else tuple(result.keypoints.xy.shape)
        print(f"{output_path} keypoints={shape}")


if __name__ == "__main__":
    main()
