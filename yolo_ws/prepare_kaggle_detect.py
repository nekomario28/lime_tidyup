#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def bbox_from_foreground(image_path: Path, threshold: int) -> tuple[float, float, float, float]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    mask = np.asarray(image).max(axis=2) > threshold
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError(f"no foreground found: {image_path}")

    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1
    cx = min_x + box_w / 2
    cy = min_y + box_h / 2
    return cx / width, cy / height, box_w / width, box_h / height


def read_filenames(labels_csv: Path) -> list[str]:
    with labels_csv.open(newline="") as f:
        rows = csv.DictReader(f)
        return [row["filename"] for row in rows]


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Kaggle Rubix cube images into YOLO bbox labels.")
    parser.add_argument("--source", default="datasets/Kaggle/training/training")
    parser.add_argument("--out", default="datasets/Kaggle/yolo_detect")
    parser.add_argument("--val-stride", type=int, default=10, help="Every Nth image goes to val.")
    parser.add_argument("--threshold", type=int, default=12)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    args = parser.parse_args()

    source = Path(args.source)
    image_dir = source / "images"
    labels_csv = source / "labels.csv"
    out = Path(args.out)

    filenames = read_filenames(labels_csv)
    counts = {"train": 0, "val": 0}
    for idx, filename in enumerate(filenames):
        split = "val" if idx % args.val_stride == 0 else "train"
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        cx, cy, bw, bh = bbox_from_foreground(image_path, args.threshold)
        link_or_copy(image_path, out / "images" / split / filename, args.copy)

        label_path = out / "labels" / split / f"{Path(filename).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        counts[split] += 1

    data_path = out
    if data_path.parts and data_path.parts[0] == "yolo_ws":
        data_path = Path(*data_path.parts[1:])

    (out / "data.yaml").write_text(
        f"path: {data_path.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        "nc: 1\n"
        "names: ['cube']\n"
    )
    print(f"wrote {out}: train={counts['train']} val={counts['val']}")


if __name__ == "__main__":
    main()
