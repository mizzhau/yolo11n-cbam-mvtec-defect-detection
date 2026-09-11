"""
split_data.py

Stratified split for MVTec AD dataset (YOLO format) into train/val/test.

Supports multiple split ratios (e.g., 80/10/10, 70/15/15, 60/20/20) in one
run.  Each ratio produces a separate output folder under --output_dir with
the standard YOLO directory layout:

    split_{ratio}/
      images/{train,val,test}/
      labels/{train,val,test}/
      classes.txt
      split_report.csv

Also generates a dataset YAML config in configs/data/ for each ratio.

Key design decisions:
  - Stratify by composite label = category + defect_type (or "good").
  - Merge good_train + good_test into a single "good" pool per category.
  - Copy files (never move) to preserve the original data.
  - Handle class-size edge cases: classes with <= 2 images go to train only.
  - Reproducible with seed (default 42).

Usage:
    python src/data/split_data.py \\
      --input_images data/interim/convert_output/images \\
      --input_labels data/processed/obj_train_data \\
      --output_dir data/processed \\
      --ratios 80/10/10 70/15/15 60/20/20 \\
      --seed 42
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_composite_label(filename: str) -> str:
    """Extract composite label from MVTec filename.

    Examples:
        bottle_broken_large_000.png  -> bottle_broken_large
        pill_good_train_005.png      -> pill_good
        metal_nut_scratch_012.png    -> metal_nut_scratch
        cable_good_test_003.png      -> cable_good

    The last ``_NNN`` is the numeric index which is stripped.  If the
    defect_type part contains ``good_train`` or ``good_test``, both are
    collapsed to just ``good`` so that a single stratification group per
    category exists for normal images.
    """
    base = os.path.splitext(filename)[0]
    # Strip trailing numeric index: ``_000``, ``_003a`` etc.
    base = re.sub(r"_\d+[a-z]?$", "", base)
    # Merge good_train / good_test -> good
    base = re.sub(r"_good_train$", "_good", base)
    base = re.sub(r"_good_test$", "_good", base)
    return base


def load_class_names(path: str) -> list[str]:
    """Read class names from obj.names or classes.txt (one per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def verify_split(
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    total_expected: int,
) -> list[str]:
    """Return a list of error messages (empty == all good)."""
    errors: list[str] = []
    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    # No overlap
    tv = train_set & val_set
    if tv:
        errors.append(f"Overlap train-val: {len(tv)} files")
    tt = train_set & test_set
    if tt:
        errors.append(f"Overlap train-test: {len(tt)} files")
    vt = val_set & test_set
    if vt:
        errors.append(f"Overlap val-test: {len(vt)} files")

    # Total preserved
    total = len(train_set) + len(val_set) + len(test_set)
    if total != total_expected:
        errors.append(
            f"Count mismatch: {total} != {total_expected} expected"
        )

    return errors


def write_split_report(
    path: str,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
) -> None:
    """Write per-group distribution CSV."""
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {
        "train": 0, "val": 0, "test": 0, "total": 0,
    })
    for f in train_files:
        g = parse_composite_label(f)
        groups[g]["train"] += 1
        groups[g]["total"] += 1
    for f in val_files:
        g = parse_composite_label(f)
        groups[g]["val"] += 1
        groups[g]["total"] += 1
    for f in test_files:
        g = parse_composite_label(f)
        groups[g]["test"] += 1
        groups[g]["total"] += 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "composite_label", "train", "val", "test", "total",
            "train_pct", "val_pct", "test_pct",
        ])
        for label in sorted(groups):
            row = groups[label]
            t = row["total"] or 1
            writer.writerow([
                label,
                row["train"], row["val"], row["test"], row["total"],
                f"{row['train']/t:.1%}",
                f"{row['val']/t:.1%}",
                f"{row['test']/t:.1%}",
            ])


def generate_yaml(
    yaml_path: str,
    split_dir: str,
    class_names: list[str],
    ratio_tag: str,
) -> None:
    """Write a YOLO dataset YAML config."""
    abs_split = os.path.abspath(split_dir).replace("\\", "/")
    lines = [
        f"# MVTec AD - YOLO Dataset Config",
        f"# Split ratio: {ratio_tag} | Seed: auto",
        f"# Auto-generated by src/data/split_data.py",
        f"",
        f"path: {abs_split}",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        f"",
        f"nc: {len(class_names)}",
        f"names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    lines.append("")

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def stratified_split(
    filenames: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified split respecting composite labels.

    Classes with <= 2 samples are placed entirely in train (cannot stratify).
    """
    labels = [parse_composite_label(f) for f in filenames]
    counts = Counter(labels)

    # Separate rare classes (can't stratify with < 3 samples)
    min_samples = 3
    main_files: list[str] = []
    main_labels: list[str] = []
    rare_files: list[str] = []

    for fname, label in zip(filenames, labels):
        if counts[label] < min_samples:
            rare_files.append(fname)
        else:
            main_files.append(fname)
            main_labels.append(label)

    if rare_files:
        rare_labels = set(parse_composite_label(f) for f in rare_files)
        print(f"  [WARN] {len(rare_files)} files in rare classes "
              f"({rare_labels}) -> forced to train")

    # Step 1: split off test
    train_val_files, test_files = train_test_split(
        main_files,
        test_size=test_ratio,
        stratify=main_labels,
        random_state=seed,
    )

    # Step 2: split off val from remainder
    tv_labels = [parse_composite_label(f) for f in train_val_files]
    adjusted_val = val_ratio / (train_ratio + val_ratio)
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=adjusted_val,
        stratify=tv_labels,
        random_state=seed,
    )

    # Add rare files to train
    train_files.extend(rare_files)

    return sorted(train_files), sorted(val_files), sorted(test_files)


def copy_files(
    filenames: list[str],
    split_name: str,
    input_images: str,
    input_labels: str,
    output_dir: str,
) -> None:
    """Copy image + label pairs into output_dir/images|labels/{split}/."""
    img_out = os.path.join(output_dir, "images", split_name)
    lbl_out = os.path.join(output_dir, "labels", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for fname in filenames:
        basename = os.path.splitext(fname)[0]
        # Copy image
        src_img = os.path.join(input_images, fname)
        if os.path.exists(src_img):
            shutil.copyfile(src_img, os.path.join(img_out, fname))
        else:
            print(f"  [WARN] Image not found: {src_img}")

        # Copy label
        lbl_name = basename + ".txt"
        src_lbl = os.path.join(input_labels, lbl_name)
        if os.path.exists(src_lbl):
            shutil.copyfile(src_lbl, os.path.join(lbl_out, lbl_name))
        else:
            print(f"  [WARN] Label not found: {src_lbl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: Optional[list[str]] = None) -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Stratified split MVTec AD dataset for YOLO training.",
    )
    parser.add_argument(
        "--input_images",
        required=True,
        help="Directory containing .png images",
    )
    parser.add_argument(
        "--input_labels",
        required=True,
        help="Directory containing .txt YOLO labels (from CVAT export)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory (split folders created inside)",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["70/15/15"],
        help="Split ratios as 'train/val/test', e.g. '80/10/10 70/15/15'",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--configs_dir",
        default="configs/data",
        help="Directory for generated YAML configs",
    )
    opts = parser.parse_args(args)

    # Collect all image filenames that have a matching label
    all_images = sorted(
        f for f in os.listdir(opts.input_images)
        if f.lower().endswith(".png")
    )
    all_label_basenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(opts.input_labels)
        if f.endswith(".txt")
    }
    matched = [
        f for f in all_images
        if os.path.splitext(f)[0] in all_label_basenames
    ]
    print(f"Found {len(matched)} matched image-label pairs")
    print(f"  Images dir: {opts.input_images}")
    print(f"  Labels dir: {opts.input_labels}")

    # Quick stats
    labels = [parse_composite_label(f) for f in matched]
    counts = Counter(labels)
    good_count = sum(c for l, c in counts.items() if l.endswith("_good"))
    defect_count = sum(c for l, c in counts.items() if not l.endswith("_good"))
    print(f"  Good: {good_count} | Defect: {defect_count} "
          f"| Groups: {len(counts)}")
    print()

    # Locate class names file
    names_file = os.path.join(opts.input_labels, "..", "obj.names")
    if not os.path.exists(names_file):
        names_file = os.path.join(
            opts.input_images, "..", "classes.txt"
        )
    class_names = load_class_names(names_file)
    print(f"Loaded {len(class_names)} class names from {names_file}")
    print()

    # Process each ratio
    for ratio_str in opts.ratios:
        parts = ratio_str.split("/")
        if len(parts) != 3:
            print(f"[ERROR] Invalid ratio format: {ratio_str}")
            continue
        tr, va, te = int(parts[0]), int(parts[1]), int(parts[2])
        total_pct = tr + va + te
        train_r = tr / total_pct
        val_r = va / total_pct
        test_r = te / total_pct

        ratio_tag = f"{tr}_{va}_{te}"
        split_dir = os.path.join(opts.output_dir, f"split_{ratio_tag}")

        print(f"{'='*60}")
        print(f"Split: {ratio_str} -> {split_dir}")
        print(f"{'='*60}")

        # Perform split
        train_f, val_f, test_f = stratified_split(
            matched, train_r, val_r, test_r, opts.seed,
        )
        print(f"  Train: {len(train_f)} | Val: {len(val_f)} "
              f"| Test: {len(test_f)} | Total: "
              f"{len(train_f)+len(val_f)+len(test_f)}")

        # Verify
        errors = verify_split(train_f, val_f, test_f, len(matched))
        if errors:
            for e in errors:
                print(f"  [FAIL] {e}")
            sys.exit(1)
        else:
            print("  [PASS] No overlap, count preserved")

        # Copy files
        print("  Copying files...")
        copy_files(
            train_f, "train",
            opts.input_images, opts.input_labels, split_dir,
        )
        copy_files(
            val_f, "val",
            opts.input_images, opts.input_labels, split_dir,
        )
        copy_files(
            test_f, "test",
            opts.input_images, opts.input_labels, split_dir,
        )

        # Copy classes.txt
        classes_dst = os.path.join(split_dir, "classes.txt")
        shutil.copy2(names_file, classes_dst)
        print(f"  Copied class names -> {classes_dst}")

        # Write split report
        report_path = os.path.join(split_dir, "split_report.csv")
        write_split_report(report_path, train_f, val_f, test_f)
        print(f"  Split report -> {report_path}")

        # Generate YAML config
        yaml_path = os.path.join(opts.configs_dir, f"mvtec_{ratio_tag}.yaml")
        generate_yaml(yaml_path, split_dir, class_names, ratio_str)
        print(f"  YAML config -> {yaml_path}")
        print()

    print("All splits completed successfully!")


if __name__ == "__main__":
    main()
