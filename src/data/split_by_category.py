"""
split_by_category.py

Reorganize the FLAT output of mvtec_mask_to_yolo.py (images/, labels/,
classes.txt, dataset-yolo_summary.csv all in one folder) into one
subfolder per MVTec category, each with its own images/ + labels/ +
classes.txt. Useful for uploading to Roboflow one category-batch at a
time (so each batch can be tagged by category).

The class ID mapping (classes.txt) is GLOBAL across the whole dataset
(built by scanning all 15 categories), so the copied classes.txt is
identical in every category folder -- do not regenerate it per category.

Usage:
    python split_by_category.py \
        --flat_root D:\Dataset__KLTN\mvtec_yolo \
        --mvtec_root D:\Dataset__KLTN\mvtec_anomaly_detection \
        --dest_root D:\Dataset__KLTN\mvtec_yolo_by_category
"""

import argparse
import os
import shutil


def find_categories(mvtec_root):
    """Same rule as mvtec_mask_to_yolo.py: every subfolder of mvtec_root is a category."""
    return sorted(
        d for d in os.listdir(mvtec_root)
        if os.path.isdir(os.path.join(mvtec_root, d))
    )


def main():
    parser = argparse.ArgumentParser(
        description="Split the flat mvtec_mask_to_yolo.py output into per-category folders."
    )
    parser.add_argument("--flat_root", type=str, required=True,
                         help="output_root you passed to mvtec_mask_to_yolo.py "
                              "(must contain images/, labels/, classes.txt).")
    parser.add_argument("--mvtec_root", type=str, required=True,
                         help="Original MVTec AD root, used to derive the category list.")
    parser.add_argument("--dest_root", type=str, required=True,
                         help="Folder to create the per-category images/labels/classes.txt in.")
    args = parser.parse_args()

    categories = find_categories(args.mvtec_root)
    print(f"Categories ({len(categories)}): {categories}")

    src_images = os.path.join(args.flat_root, "images")
    src_labels = os.path.join(args.flat_root, "labels")
    classes_file = os.path.join(args.flat_root, "classes.txt")
    summary_file = os.path.join(args.flat_root, "dataset-yolo_summary.csv")

    all_images = sorted(f for f in os.listdir(src_images) if f.lower().endswith(".png"))

    # Safety check: every image must match exactly one category prefix,
    # so no image silently ends up in the wrong folder or gets dropped.
    counts = {cat: 0 for cat in categories}
    unmatched = []
    for fname in all_images:
        matches = [cat for cat in categories if fname.startswith(cat + "_")]
        if len(matches) != 1:
            unmatched.append((fname, matches))
        else:
            counts[matches[0]] += 1

    if unmatched:
        print(f"[ERROR] {len(unmatched)} files did not match exactly one category prefix:")
        for fname, matches in unmatched[:20]:
            print(f"  {fname} -> matches {matches}")
        raise SystemExit(1)

    print("Prefix match check OK. Per-category image counts:")
    for cat, n in counts.items():
        print(f"  {cat:12s} {n}")

    os.makedirs(args.dest_root, exist_ok=True)
    for cat in categories:
        os.makedirs(os.path.join(args.dest_root, cat, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.dest_root, cat, "labels"), exist_ok=True)
        shutil.copy2(classes_file, os.path.join(args.dest_root, cat, "classes.txt"))

    if os.path.exists(summary_file):
        shutil.copy2(summary_file, os.path.join(args.dest_root, "dataset-yolo_summary.csv"))

    copied_images = 0
    copied_labels = 0
    for fname in all_images:
        cat = next(c for c in categories if fname.startswith(c + "_"))
        stem = os.path.splitext(fname)[0]
        shutil.copy2(os.path.join(src_images, fname), os.path.join(args.dest_root, cat, "images", fname))
        copied_images += 1

        label_name = stem + ".txt"
        src_label_path = os.path.join(src_labels, label_name)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, os.path.join(args.dest_root, cat, "labels", label_name))
            copied_labels += 1
        else:
            print(f"[WARN] missing label for {fname}")

    print(f"\nDone. Copied {copied_images} images, {copied_labels} labels into: {args.dest_root}")


if __name__ == "__main__":
    main()
