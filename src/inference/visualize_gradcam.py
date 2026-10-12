import argparse
import os
import sys
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models.cbam import register_cbam_to_ultralytics
from src.inference.gradcam_core import compute_gradcam, stratified_sample_images


def generate_gradcam(
    weights_path: str,
    test_file: str,
    output_dir: str,
    img_size: int = 640,
    max_images: int = 30,
    device: str = ""
) -> None:
    """Tạo bản đồ nhiệt Grad-CAM CHỈ SỬ DỤNG tập TEST (test set)."""
    w_p = Path(weights_path)
    if not w_p.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy trọng số tại {w_p}. Bỏ qua Grad-CAM.")
        return

    test_p = Path(test_file)
    if not test_p.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy file test set tại {test_p}. Bỏ qua Grad-CAM.")
        return

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if "cbam" in str(w_p).lower():
        register_cbam_to_ultralytics()

    from ultralytics import YOLO

    device_obj = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    yolo_model = YOLO(weights_path)
    model = yolo_model.model.to(device_obj)
    model.eval()

    with open(test_p, "r", encoding="utf-8") as f:
        all_lines = [
            line.strip() for line in f if line.strip() and not line.strip().startswith("#")
        ]

    dataset_root = test_p.parent
    valid_test_images: List[Path] = []
    for line in all_lines:
        p = Path(line)
        if not p.is_absolute():
            p = dataset_root / line
        if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            valid_test_images.append(p)

    selected_images = stratified_sample_images(valid_test_images, max_images=max_images)

    print("=" * 70)
    print(" BẮT ĐẦU TRỰC QUAN HÓA GRAD-CAM (CHỈ DÙNG TEST SET)")
    print(f" - Trọng số:       {weights_path}")
    print(f" - Test set file:  {test_file} (Tổng {len(valid_test_images)} ảnh test)")
    print(f" - Số ảnh xử lý:   {len(selected_images)}")
    print(f" - Thư mục lưu:    {out_p}")
    print("=" * 70)

    for idx, img_path in enumerate(selected_images, 1):
        orig_bgr = cv2.imread(str(img_path))
        if orig_bgr is None:
            continue

        overlay, heatmap, _ = compute_gradcam(
            model=model,
            image_bgr=orig_bgr,
            img_size=img_size,
            device=device_obj
        )

        combined = np.hstack([orig_bgr, overlay])
        save_name = f"{img_path.stem}_gradcam.png"
        cv2.imwrite(str(out_p / save_name), combined)

    print(f"✅ Hoàn tất lưu toàn bộ Grad-CAM vào: {out_p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trực quan hóa Grad-CAM cho mô hình YOLO trên Test Set."
    )
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file best.pt")
    parser.add_argument(
        "--test_file", type=str, required=True, help="Đường dẫn file danh sách test set (.txt)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Thư mục xuất ảnh Grad-CAM")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    generate_gradcam(
        weights_path=args.weights,
        test_file=args.test_file,
        output_dir=args.output_dir,
        img_size=args.imgsz,
        max_images=args.max_images,
        device=args.device
    )


if __name__ == "__main__":
    main()
