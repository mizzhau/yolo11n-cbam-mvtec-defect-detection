"""
verify_yolo_boxes.py

Quick visual QA tool: draws the converted YOLO bounding boxes on top of the
original image, so you can visually confirm the boxes actually line up with
the real defect regions before trusting the full-dataset conversion.

Usage:
    python verify_yolo_boxes.py --output_root /path/to/output --stem bottle_broken_large_000
    (omit --stem to get a random sample instead)
"""

import argparse
import os
import random

import cv2


def load_class_names(classes_file):
    """Read classes.txt and return a list where index i = class name for id i."""
    with open(classes_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def draw_boxes(image_path, label_path, save_path, class_names=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label file found: {label_path}")
        return

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"'{os.path.basename(label_path)}' is empty -> no defect (this is expected for 'good' images).")

    for line in lines:
        parts = line.split()
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:])

        # Lookup human-readable name from classes.txt if available
        if class_names and 0 <= class_id < len(class_names):
            display_label = f"{class_names[class_id]} ({class_id})"
        else:
            display_label = f"class {class_id}"

        x_center_px = x_center * w
        y_center_px = y_center * h
        box_w_px = box_w * w
        box_h_px = box_h * h

        x1 = int(x_center_px - box_w_px / 2)
        y1 = int(y_center_px - box_h_px / 2)
        x2 = int(x_center_px + box_w_px / 2)
        y2 = int(y_center_px + box_h_px / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, display_label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(save_path, img)
    print(f"Saved verification image to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, required=True,
                         help="The --output_root you passed to mvtec_mask_to_yolo.py")
    parser.add_argument("--stem", type=str, default=None,
                         help="Specific file stem to check (without extension). "
                              "If omitted, picks a random image.")
    parser.add_argument("--save_dir", type=str, default="verify_output",
                         help="Folder to save the annotated preview images into.")
    args = parser.parse_args()

    images_dir = os.path.join(args.output_root, "images")
    labels_dir = os.path.join(args.output_root, "labels")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load class names from classes.txt for human-readable overlay
    classes_file = os.path.join(args.output_root, "classes.txt")
    class_names = None
    if os.path.exists(classes_file):
        class_names = load_class_names(classes_file)
        print(f"Loaded {len(class_names)} class names from {classes_file}")
    else:
        print(f"[warn] classes.txt not found at {classes_file}, will show numeric IDs only.")

    if args.stem:
        stems = [args.stem]
    else:
        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
        stems = [os.path.splitext(random.choice(all_images))[0]]

    for stem in stems:
        image_path = os.path.join(images_dir, stem + ".png")
        label_path = os.path.join(labels_dir, stem + ".txt")
        save_path = os.path.join(args.save_dir, stem + "_verify.png")
        draw_boxes(image_path, label_path, save_path, class_names=class_names)


if __name__ == "__main__":
    main()
