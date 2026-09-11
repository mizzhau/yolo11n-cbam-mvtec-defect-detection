"""
mvtec_mask_to_yolo.py

Convert MVTec AD binary defect masks to YOLO bounding box annotations.

Pipeline (adapted from matthewkenely/mask-to-annotation, with fixes):
  1. Read each ground-truth mask as GRAYSCALE (avoids the 3-channel color
     branch in the original component_labelling(), which is meant for
     colored multi-class masks, not MVTec's plain binary masks).
  2. Split each mask into connected components (cv2.connectedComponentsWithStats)
     so that multiple disjoint defect regions in one mask become separate boxes.
  3. Get the bounding rect of each component.
  4. Filter out boxes that are too small (noise) using a minimum-area ratio.
  5. Normalize box coordinates to YOLO format (class_id x_center y_center w h).
  6. Build a deterministic class mapping: scan ALL defect-type folder names
     across every category, sort alphabetically, and assign index 0,1,2,...
     Each defect box gets the integer class_id from this mapping.
  7. Also copy "good" (defect-free) images with an EMPTY label file, so the
     model learns what a normal / no-defect image looks like.
  8. Write out a dataset-yolo_summary.csv with per-category / per-defect-type counts,
     so you can inspect class balance before training.

Expected MVTec AD folder structure (after extracting the .tar.xz):

    mvtec_root/
      bottle/
        train/
          good/000.png ...
        test/
          good/000.png ...
          broken_large/000.png ...
          broken_small/000.png ...
        ground_truth/
          broken_large/000_mask.png ...
          broken_small/000_mask.png ...
      cable/
        ...
      ... (15 categories total)

Output structure produced by this script:

    output_root/
      images/
        bottle_broken_large_000.png
        bottle_good_test_000.png
        ...
      labels/
        bottle_broken_large_000.txt
        bottle_good_test_000.txt      (empty file -> no defect)
        ...
      dataset-yolo_summary.csv

You still need to do the train/val/test split afterwards (not included here
on purpose, so you can apply the stratified split strategy discussed earlier).

Usage:
    python mvtec_mask_to_yolo.py --mvtec_root /path/to/mvtec_ad --output_root /path/to/output
"""

import argparse
import csv
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------------

# Minimum area of a connected component, as a fraction of total image area.
# Components smaller than this are treated as annotation noise and dropped.
# Tune this per your own visual inspection (start here, adjust if needed).
MIN_AREA_RATIO = 0.0005  # 0.05% of image area

# Class mapping is now built dynamically by scanning all defect-type folders
# across every category, sorting alphabetically, and assigning 0,1,2,...
# See collect_all_defect_types() and build_class_mapping().

# Light denoising applied before contour detection, mirrors the original
# repo's use of Gaussian blur + erosion to clean up jagged mask edges.
BLUR_KERNEL = (7, 7)
ERODE_KERNEL = np.ones((3, 3), np.uint8)


# ----------------------------------------------------------------------------
# Core conversion logic
# ----------------------------------------------------------------------------

def clean_binary_mask(mask):
    """
    Threshold + blur + erode a grayscale mask to reduce jagged edges and
    stray noise pixels before connected-component analysis.
    mask: single-channel (grayscale) numpy array.
    """
    # Ensure strictly binary (0 / 255) before further processing.
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Light blur to smooth jagged edges (anti-aliasing), then erode back
    # down so the blur doesn't inflate the defect region.
    blurred = cv2.GaussianBlur(binary, BLUR_KERNEL, sigmaX=1, sigmaY=1)
    eroded = cv2.erode(blurred, ERODE_KERNEL, iterations=1)

    # Re-threshold after blur/erode so the mask is clean binary again.
    _, cleaned = cv2.threshold(eroded, 127, 255, cv2.THRESH_BINARY)
    return cleaned


def mask_to_boxes(mask, min_area_ratio=MIN_AREA_RATIO):
    """
    Convert a single-channel binary mask into a list of (x, y, w, h) pixel
    bounding boxes, one per disjoint defect region (connected component).

    Returns: list of (x, y, w, h) tuples in pixel coordinates.
    """
    h_img, w_img = mask.shape[:2]
    total_area = h_img * w_img
    min_area = min_area_ratio * total_area

    cleaned = clean_binary_mask(mask)

    # connectedComponentsWithStats requires a single-channel binary image.
    # Label 0 is always the background, so components start at label 1.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    boxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        # Filter out noise: components with pixel area below the threshold.
        # (stats' CC_STAT_AREA is pixel count of the component, a good proxy
        # for real defect size, better than just using box area.)
        if area < min_area:
            continue

        boxes.append((int(x), int(y), int(w), int(h)))

    return boxes


def boxes_to_yolo_lines(boxes, img_w, img_h, class_label, class_to_id):
    """
    Convert pixel (x, y, w, h) boxes into normalized YOLO format lines:
    "class_id x_center y_center width height", all normalized to [0, 1].

    class_label: string name of the defect type (e.g. "broken_large").
    class_to_id: dict mapping defect-type names to integer class indices.

    Output lines use the integer class_id, e.g.:
        5 0.622778 0.614444 0.465556 0.591111
    """
    class_id = class_to_id[class_label]
    lines = []
    for (x, y, w, h) in boxes:
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
    return lines


# ----------------------------------------------------------------------------
# MVTec AD directory walking
# ----------------------------------------------------------------------------

def find_categories(mvtec_root):
    """Return a sorted list of category folder names (e.g. bottle, cable, ...)."""
    return sorted(
        d for d in os.listdir(mvtec_root)
        if os.path.isdir(os.path.join(mvtec_root, d))
    )


def process_defect_images(category, mvtec_root, output_images_dir, output_labels_dir, summary,
                           class_to_id, min_area_ratio=MIN_AREA_RATIO):
    """
    Process all defective test images for one category: match each test image
    to its ground-truth mask, convert mask -> boxes -> YOLO label file, and
    copy the image into the flat output structure.
    """
    test_dir = os.path.join(mvtec_root, category, "test")
    gt_dir = os.path.join(mvtec_root, category, "ground_truth")

    if not os.path.isdir(test_dir) or not os.path.isdir(gt_dir):
        print(f"  [skip] {category}: missing test/ or ground_truth/ folder")
        return

    defect_types = sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and d != "good"
    )

    for defect_type in defect_types:
        img_dir = os.path.join(test_dir, defect_type)
        mask_dir = os.path.join(gt_dir, defect_type)

        if not os.path.isdir(mask_dir):
            print(f"  [skip] {category}/{defect_type}: no matching ground_truth folder")
            continue

        image_files = sorted(
            f for f in os.listdir(img_dir) if f.lower().endswith(".png")
        )

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]  # e.g. "000"
            mask_file = f"{base_name}_mask.png"
            mask_path = os.path.join(mask_dir, mask_file)
            image_path = os.path.join(img_dir, image_file)

            if not os.path.exists(mask_path):
                print(f"  [warn] missing mask for {image_path}, skipping")
                continue

            # Read mask as single-channel grayscale (avoids the color-mask
            # branch meant for multi-colored masks, which MVTec AD is not).
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  [warn] failed to read mask {mask_path}, skipping")
                continue

            img = cv2.imread(image_path)
            if img is None:
                print(f"  [warn] failed to read image {image_path}, skipping")
                continue
            img_h, img_w = img.shape[:2]

            boxes = mask_to_boxes(mask, min_area_ratio=min_area_ratio)
            yolo_lines = boxes_to_yolo_lines(boxes, img_w, img_h, defect_type, class_to_id)

            # Unique flat filename: category_defecttype_basename
            out_stem = f"{category}_{defect_type}_{base_name}"
            out_image_path = os.path.join(output_images_dir, out_stem + ".png")
            out_label_path = os.path.join(output_labels_dir, out_stem + ".txt")

            shutil.copy2(image_path, out_image_path)
            with open(out_label_path, "w") as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")

            summary[(category, defect_type)]["images"] += 1
            summary[(category, defect_type)]["boxes"] += len(yolo_lines)


def process_good_images(category, mvtec_root, output_images_dir, output_labels_dir, summary):
    """
    Copy defect-free ("good") images from both train/ and test/ splits, each
    paired with an EMPTY label file (no objects), so the model also sees
    negative examples during training.
    """
    for split in ("train", "test"):
        good_dir = os.path.join(mvtec_root, category, split, "good")
        if not os.path.isdir(good_dir):
            continue

        image_files = sorted(
            f for f in os.listdir(good_dir) if f.lower().endswith(".png")
        )

        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(good_dir, image_file)

            out_stem = f"{category}_good_{split}_{base_name}"
            out_image_path = os.path.join(output_images_dir, out_stem + ".png")
            out_label_path = os.path.join(output_labels_dir, out_stem + ".txt")

            shutil.copy2(image_path, out_image_path)
            # Empty label file = "no objects in this image" in YOLO format.
            open(out_label_path, "w").close()

            summary[(category, f"good_{split}")]["images"] += 1
            summary[(category, f"good_{split}")]["boxes"] += 0


# ----------------------------------------------------------------------------
# Summary reporting
# ----------------------------------------------------------------------------

def write_summary_csv(summary, output_root):
    summary_path = os.path.join(output_root, "dataset-yolo_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "defect_type", "num_images", "num_boxes"])
        for (category, defect_type), counts in sorted(summary.items()):
            writer.writerow([category, defect_type, counts["images"], counts["boxes"]])
    print(f"\nSummary written to: {summary_path}")


def write_classes_file(output_root, class_to_id):
    """Write classes.txt where line i = name of class with id i."""
    # Invert mapping: id → name, then write in index order.
    id_to_class = {v: k for k, v in class_to_id.items()}
    classes_path = os.path.join(output_root, "classes.txt")
    with open(classes_path, "w") as f:
        for idx in range(len(id_to_class)):
            f.write(id_to_class[idx] + "\n")
    print(f"Classes file written to: {classes_path}  ({len(id_to_class)} classes)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def collect_all_defect_types(mvtec_root, categories):
    """
    Scan every category's test/ folder (excluding 'good') to collect the
    complete, deduplicated set of defect-type names across the whole dataset.
    Returns a sorted list of unique defect-type strings.
    """
    defect_types = set()
    for category in categories:
        test_dir = os.path.join(mvtec_root, category, "test")
        if not os.path.isdir(test_dir):
            continue
        for d in os.listdir(test_dir):
            if os.path.isdir(os.path.join(test_dir, d)) and d != "good":
                defect_types.add(d)
    return sorted(defect_types)


def build_class_mapping(defect_types):
    """
    Build a deterministic name → integer mapping from a sorted list of
    defect-type names.  E.g. {'bent': 0, 'bent_lead': 1, ...}
    """
    return {name: idx for idx, name in enumerate(defect_types)}


def main():
    parser = argparse.ArgumentParser(
        description="Convert MVTec AD binary masks to YOLO bounding box annotations."
    )
    parser.add_argument("--mvtec_root", type=str, required=True,
                         help="Path to the root folder of the extracted MVTec AD dataset "
                              "(the folder containing bottle/, cable/, etc.)")
    parser.add_argument("--output_root", type=str, required=True,
                         help="Path to the folder where images/ and labels/ will be created.")
    parser.add_argument("--min_area_ratio", type=float, default=MIN_AREA_RATIO,
                         help="Minimum connected-component area as a fraction of image area; "
                              "components smaller than this are treated as noise and dropped.")
    args = parser.parse_args()

    output_images_dir = os.path.join(args.output_root, "images")
    output_labels_dir = os.path.join(args.output_root, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    categories = find_categories(args.mvtec_root)
    print(f"Found {len(categories)} categories: {categories}")

    # ---- Build deterministic class mapping (sorted alphabetically) ----
    all_defect_types = collect_all_defect_types(args.mvtec_root, categories)
    class_to_id = build_class_mapping(all_defect_types)
    print(f"\nClass mapping ({len(class_to_id)} defect types):")
    for name, idx in class_to_id.items():
        print(f"  {idx:3d} -> {name}")

    summary = defaultdict(lambda: {"images": 0, "boxes": 0})

    for category in categories:
        print(f"\nProcessing category: {category}")
        process_defect_images(category, args.mvtec_root, output_images_dir, output_labels_dir, summary,
                               class_to_id=class_to_id, min_area_ratio=args.min_area_ratio)
        process_good_images(category, args.mvtec_root, output_images_dir, output_labels_dir, summary)

    write_summary_csv(summary, args.output_root)
    write_classes_file(args.output_root, class_to_id)

    total_images = sum(v["images"] for v in summary.values())
    total_boxes = sum(v["boxes"] for v in summary.values())
    print(f"\nDone. Total images: {total_images}, total boxes: {total_boxes}")
    print(f"Images saved to: {output_images_dir}")
    print(f"Labels saved to: {output_labels_dir}")
    print("\nNext steps (not done by this script):")
    print("  1. Visually spot-check a sample of images/labels (overlay boxes on images).")
    print("  2. Perform the stratified train/val/test split per category + defect type.")
    print("  3. Feed the resulting folders into YOLO11n training.")


if __name__ == "__main__":
    main()
