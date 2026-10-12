import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
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


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Tính Intersection over Union giữa hai bounding box."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return (inter_area / union_area) if union_area > 0 else 0.0


def read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, List[float]]]:
    """Đọc nhãn YOLO dạng [cls_id, cx, cy, w, h] chuyển sang pixel [x1, y1, x2, y2]."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = [float(x) for x in parts[1:5]]
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                boxes.append((cls_id, [x1, y1, x2, y2]))
    return boxes


def run_visual_analysis(
    weights_path: str,
    test_file: str,
    output_dir: str,
    img_size: int = 640,
    max_images: int = 30,
    conf_thresh: float = 0.25,
    device: str = ""
) -> None:
    """Khởi chạy bộ trực quan hóa: Grad-CAM, Side-by-Side và Failure Cases."""
    w_p = Path(weights_path)
    if not w_p.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy file trọng số {w_p}")
        return

    test_p = Path(test_file)
    if not test_p.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy file test set {test_p}")
        return

    base_out = Path(output_dir)
    dir_gradcam = base_out / "grad_cam"
    dir_side_by_side = base_out / "side_by_side"
    dir_failure = base_out / "failure_cases"

    dir_gradcam.mkdir(parents=True, exist_ok=True)
    dir_side_by_side.mkdir(parents=True, exist_ok=True)
    dir_failure.mkdir(parents=True, exist_ok=True)

    if "cbam" in str(w_p).lower():
        register_cbam_to_ultralytics()

    from ultralytics import YOLO

    device_obj = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    yolo = YOLO(weights_path)
    model = yolo.model.to(device_obj)
    model.eval()

    with open(test_p, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    dataset_root = test_p.parent
    valid_images: List[Path] = []
    for line in lines:
        p = Path(line)
        if not p.is_absolute():
            p = dataset_root / line
        if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            valid_images.append(p)

    selected_images = stratified_sample_images(valid_images, max_images=max_images)

    print("=" * 75)
    print(" KHỞI CHẠY BỘ TRỰC QUAN HÓA TOÀN DIỆN (CHỈ DÙNG TEST SET)")
    print(f" - Trọng số:            {weights_path}")
    print(f" - Test set file:       {test_file} (Tổng {len(valid_images)} ảnh)")
    print(f" - Số ảnh chọn lọc:     {len(selected_images)} (Bao phủ 15 categories)")
    print(f" - Thư mục tổng hợp:    {base_out}")
    print(f"   ├── 1. Grad-CAM:     {dir_gradcam}")
    print(f"   ├── 2. Side-by-Side: {dir_side_by_side}")
    print(f"   └── 3. Failure:      {dir_failure}")
    print("=" * 75)

    failure_records = []
    names = getattr(yolo, "names", {})

    for idx, img_path in enumerate(selected_images, 1):
        orig_bgr = cv2.imread(str(img_path))
        if orig_bgr is None:
            continue
        h_orig, w_orig = orig_bgr.shape[:2]

        label_candidates = [
            img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
            dataset_root / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
            Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        ]
        label_file = next((cand for cand in label_candidates if cand.exists()), label_candidates[0])
        gt_boxes = read_yolo_labels(label_file, w_orig, h_orig)

        target_cls = gt_boxes[0][0] if gt_boxes else None
        overlay, heatmap, _ = compute_gradcam(
            model=model,
            image_bgr=orig_bgr,
            img_size=img_size,
            target_cls=target_cls,
            device=device_obj
        )
        gradcam_pair = np.hstack([orig_bgr, overlay])
        cv2.imwrite(str(dir_gradcam / f"{img_path.stem}_gradcam.png"), gradcam_pair)

        pred_results = yolo.predict(
            source=str(img_path), conf=conf_thresh, imgsz=img_size, verbose=False
        )
        det_boxes = []
        if pred_results and len(pred_results) > 0 and pred_results[0].boxes is not None:
            boxes_obj = pred_results[0].boxes
            for i in range(len(boxes_obj)):
                xyxy = boxes_obj.xyxy[i].cpu().numpy().tolist()
                cls = int(boxes_obj.cls[i].item())
                conf = float(boxes_obj.conf[i].item())
                det_boxes.append((cls, conf, xyxy))

        gt_canvas = orig_bgr.copy()
        pred_canvas = orig_bgr.copy()

        cv2.putText(
            gt_canvas, "GROUND TRUTH", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )
        for cls_id, box in gt_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(gt_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            c_name = names.get(cls_id, str(cls_id))
            cv2.putText(
                gt_canvas, c_name, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        model_tag = Path(weights_path).parent.parent.name
        cv2.putText(
            pred_canvas, f"PREDICTION ({model_tag})", (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2
        )
        for cls_id, conf, box in det_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(pred_canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            c_name = names.get(cls_id, str(cls_id))
            label_text = f"{c_name} {conf:.2f}"
            cv2.putText(
                pred_canvas, label_text, (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        side_by_side = np.hstack([gt_canvas, pred_canvas])
        cv2.imwrite(str(dir_side_by_side / f"{img_path.stem}_comparison.png"), side_by_side)

        is_failure = False
        reason = ""

        if len(gt_boxes) > 0 and len(det_boxes) == 0:
            is_failure = True
            reason = "False Negative (Sót lỗi hoàn toàn)"
        elif len(gt_boxes) == 0 and len(det_boxes) > 0:
            is_failure = True
            reason = "False Positive (Cảnh báo nhầm ảnh bình thường)"
        elif len(gt_boxes) > 0 and len(det_boxes) > 0:
            best_iou = 0.0
            for _, gt_b in gt_boxes:
                for _, _, det_b in det_boxes:
                    best_iou = max(best_iou, compute_iou(gt_b, det_b))
            if best_iou < 0.3:
                is_failure = True
                reason = f"Low IoU Localization Error (IoU cực đại = {best_iou:.2f} < 0.30)"

        if is_failure:
            failure_canvas = side_by_side.copy()
            cv2.putText(
                failure_canvas, f"[FAIL: {reason}]", (15, h_orig - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )
            cv2.imwrite(str(dir_failure / f"fail_{img_path.stem}.png"), failure_canvas)
            failure_records.append({"image": img_path.name, "reason": reason})

    report_p = dir_failure / "failure_report.md"
    with open(report_p, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Phân Tích Ca Lỗi Biên (Failure Cases Analysis)\n\n")
        f.write(f"- **Mô hình kiểm định:** `{weights_path}`\n")
        f.write(f"- **Tập dữ liệu:** Chỉ lấy mẫu từ Test Set (`{test_file}`)\n")
        f.write(
            f"- **Tổng số ca lỗi biên phát hiện:** {len(failure_records)} / "
            f"{len(selected_images)} ảnh kiểm thử\n\n"
        )
        f.write("| Tên Ảnh | Dạng Lỗi Biên (Error Type) | Nhận Định |\n")
        f.write("|:---|:---|:---|\n")
        for rec in failure_records:
            f.write(
                f"| `{rec['image']}` | **{rec['reason']}** | "
                f"Kiểm tra chi tiết tại `failure_cases/fail_{Path(rec['image']).stem}.png` |\n"
            )
        if not failure_records:
            f.write(
                "\n> 🎉 Xuất sắc: Không phát hiện ca lỗi biên nghiêm trọng "
                "trong tập kiểm thử được lấy mẫu!\n"
            )

    print("\n✅ HOÀN TẤT BỘ TRỰC QUAN HÓA TOÀN DIỆN:")
    print(f" - Grad-CAM lưu tại:       {dir_gradcam}")
    print(f" - Side-by-Side lưu tại:   {dir_side_by_side}")
    print(f" - Failure Cases lưu tại:  {dir_failure} (Báo cáo: {report_p})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bộ trực quan hóa trên Test Set (Grad-CAM, Side-by-Side, Failure Cases)."
    )
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file best.pt")
    parser.add_argument(
        "--test_file", type=str, required=True, help="Đường dẫn danh sách test set (.txt)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Thư mục tổng hợp xuất trực quan hóa"
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    run_visual_analysis(
        weights_path=args.weights,
        test_file=args.test_file,
        output_dir=args.output_dir,
        img_size=args.imgsz,
        max_images=args.max_images,
        conf_thresh=args.conf,
        device=args.device
    )


if __name__ == "__main__":
    main()
