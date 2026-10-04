import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def parse_composite_label(filename: str) -> Tuple[str, str]:
    base = os.path.splitext(filename)[0]
    matched_cat = ""
    for cat in sorted(CATEGORIES, key=len, reverse=True):
        if base.startswith(cat + "_"):
            matched_cat = cat
            remainder = base[len(cat) + 1:]
            break

    if not matched_cat:
        return "unknown", base

    if remainder.startswith("good"):
        return matched_cat, "good"

    parts = remainder.rsplit("_", 1)
    defect = parts[0] if len(parts) == 2 and parts[1].isdigit() else remainder
    return matched_cat, defect


def load_yolo_boxes(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    boxes: List[List[float]] = []
    class_ids: List[int] = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes, class_ids

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            w = max(0.001, min(0.999, w))
            h = max(0.001, min(0.999, h))
            boxes.append([cx, cy, w, h])
            class_ids.append(cid)
    return boxes, class_ids


def save_yolo_boxes(label_path: Path, boxes: List[List[float]], class_ids: List[int]) -> None:
    lines = [f"{int(cid)} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for cid, b in zip(class_ids, boxes)]
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def build_transforms(category: str) -> A.Compose:
    bbox_params = A.BboxParams(format="yolo", label_fields=["category_ids"], min_visibility=0.3)
    transforms: List[A.BasicTransform] = [A.HorizontalFlip(p=0.5)]

    if category != "transistor":
        transforms.append(A.VerticalFlip(p=0.5))
        transforms.append(A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-10, 10), p=0.7))
    else:
        transforms.append(A.Affine(scale=(0.97, 1.03), translate_percent=(-0.02, 0.02), rotate=(-5, 5), p=0.5))

    transforms.extend([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])

    return A.Compose(transforms, bbox_params=bbox_params)


def augment_offline_dataset(source_dir: Path, output_dir: Path, target: int = 40, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    src_img_dir = source_dir / "images"
    src_lbl_dir = source_dir / "labels"
    out_img_dir = output_dir / "images"
    out_lbl_dir = output_dir / "labels"

    os.makedirs(out_img_dir / "train", exist_ok=True)
    os.makedirs(out_lbl_dir / "train", exist_ok=True)

    for split in ["val", "test"]:
        dst_i = out_img_dir / split
        dst_l = out_lbl_dir / split
        if dst_i.exists():
            shutil.rmtree(dst_i)
        if dst_l.exists():
            shutil.rmtree(dst_l)
        shutil.copytree(src_img_dir / split, dst_i)
        shutil.copytree(src_lbl_dir / split, dst_l)

    classes_src = source_dir / "classes.txt"
    if classes_src.exists():
        shutil.copyfile(classes_src, output_dir / "classes.txt")

    train_files = sorted([f for f in os.listdir(src_img_dir / "train") if f.endswith((".png", ".jpg"))])
    groups: Dict[str, List[str]] = {}
    good_files: List[str] = []

    for fname in train_files:
        cat, defect = parse_composite_label(fname)
        if defect == "good":
            good_files.append(fname)
        else:
            group_key = f"{cat}_{defect}"
            groups.setdefault(group_key, []).append(fname)

    for fname in good_files:
        shutil.copyfile(src_img_dir / "train" / fname, out_img_dir / "train" / fname)
        lbl_name = Path(fname).stem + ".txt"
        lbl_p = src_lbl_dir / "train" / lbl_name
        if lbl_p.exists():
            shutil.copyfile(lbl_p, out_lbl_dir / "train" / lbl_name)
        else:
            (out_lbl_dir / "train" / lbl_name).touch()

    for fname_list in groups.values():
        for fname in fname_list:
            shutil.copyfile(src_img_dir / "train" / fname, out_img_dir / "train" / fname)
            lbl_name = Path(fname).stem + ".txt"
            shutil.copyfile(src_lbl_dir / "train" / lbl_name, out_lbl_dir / "train" / lbl_name)

    stats = []

    for group_key, flist in sorted(groups.items()):
        cat = group_key.split("_")[0]
        curr_count = len(flist)
        needed = max(0, target - curr_count)
        transform = build_transforms(cat)

        valid_flist = [
            f for f in flist
            if (src_lbl_dir / "train" / f"{Path(f).stem}.txt").exists()
            and (src_lbl_dir / "train" / f"{Path(f).stem}.txt").stat().st_size > 0
        ]
        if not valid_flist:
            valid_flist = list(flist)

        aug_idx = 0
        attempt = 0
        max_attempts = needed * 20

        while aug_idx < needed and attempt < max_attempts:
            src_fname = valid_flist[attempt % len(valid_flist)]
            attempt += 1
            stem = Path(src_fname).stem
            img_path = src_img_dir / "train" / src_fname
            lbl_path = src_lbl_dir / "train" / f"{stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            boxes, cids = load_yolo_boxes(lbl_path)
            if not boxes:
                continue

            try:
                res = transform(image=img, bboxes=boxes, category_ids=cids)
                aug_img = res["image"]
                aug_boxes = [list(b) for b in res["bboxes"]]
                aug_cids = [int(c) for c in res["category_ids"]]
            except Exception:
                continue

            if not aug_boxes or len(aug_boxes) != len(boxes):
                continue

            aug_idx += 1
            out_stem = f"{stem}_aug_{aug_idx}"
            cv2.imwrite(str(out_img_dir / "train" / f"{out_stem}.png"), aug_img)
            save_yolo_boxes(out_lbl_dir / "train" / f"{out_stem}.txt", aug_boxes, aug_cids)

        stats.append({
            "group": group_key,
            "category": cat,
            "original_train": curr_count,
            "augmented_added": aug_idx,
            "final_train": curr_count + aug_idx
        })

    report_path = output_dir / "augment_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "category", "original_train", "augmented_added", "final_train"])
        writer.writeheader()
        writer.writerows(stats)

    yaml_content = f"""path: {output_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 48
names: {load_class_names(classes_src)}
"""
    yaml_path = Path("configs/data/mvtec_70_15_15_augmented.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def load_class_names(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline augmentation for MVTec AD defect groups.")
    parser.add_argument("--source_dir", type=str, default="data/processed/split_70_15_15")
    parser.add_argument("--output_dir", type=str, default="data/processed/split_70_15_15_augmented")
    parser.add_argument("--target", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    augment_offline_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        target=args.target,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
