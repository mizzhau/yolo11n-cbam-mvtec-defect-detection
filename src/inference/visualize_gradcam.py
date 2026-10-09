import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import cv2
import numpy as np
import torch
from PIL import Image

from src.models.cbam import register_cbam_to_ultralytics


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Tự động tìm layer attention hoặc layer tích hợp đặc trưng cuối cùng trước Detect."""
    target_layer = None
    for layer in reversed(list(model.model)):
        layer_name = type(layer).__name__
        if layer_name in ["CBAM", "C2PSA", "SPPF", "C3k2"]:
            target_layer = layer
            break
    if target_layer is None:
        target_layer = model.model[-2]
    return target_layer


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

    target_layer = find_target_layer(model)

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module: torch.nn.Module, inp: tuple, out: torch.Tensor) -> None:
        activations.append(out)

    def backward_hook(module: torch.nn.Module, grad_in: tuple, grad_out: tuple) -> None:
        gradients.append(grad_out[0])

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Đọc danh sách ảnh trong test set
    with open(test_p, "r", encoding="utf-8") as f:
        all_lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    dataset_root = test_p.parent
    valid_test_images: List[Path] = []
    for line in all_lines:
        p = Path(line)
        if not p.is_absolute():
            p = dataset_root / line
        if p.exists() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            valid_test_images.append(p)

    # Ưu tiên lấy mẫu đa dạng theo category từ test set
    category_map = {}
    for img_p in valid_test_images:
        cat = img_p.parent.parent.name if img_p.parent.name != "test" else img_p.parent.parent.parent.name
        if not cat or cat == ".":
            cat = img_p.parts[-3] if len(img_p.parts) >= 3 else "default"
        if cat not in category_map:
            category_map[cat] = []
        category_map[cat].append(img_p)

    selected_images: List[Path] = []
    for cat, imgs in category_map.items():
        selected_images.extend(imgs[:2])
        if len(selected_images) >= max_images:
            break

    if not selected_images:
        selected_images = valid_test_images[:max_images]

    print("=" * 70)
    print(" BẮT ĐẦU TRỰC QUAN HÓA GRAD-CAM (CHỈ DÙNG TEST SET)")
    print(f" - Trọng số:       {weights_path}")
    print(f" - Test set file:  {test_file} (Tổng {len(valid_test_images)} ảnh test)")
    print(f" - Số ảnh xử lý:   {len(selected_images)}")
    print(f" - Thư mục lưu:    {out_p}")
    print(f" - Target layer:   {type(target_layer).__name__}")
    print("=" * 70)

    for idx, img_path in enumerate(selected_images, 1):
        orig_bgr = cv2.imread(str(img_path))
        if orig_bgr is None:
            continue
        h_orig, w_orig = orig_bgr.shape[:2]

        # Tiền xử lý theo chuẩn YOLO 640x640
        resized_rgb = cv2.cvtColor(cv2.resize(orig_bgr, (img_size, img_size)), cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(resized_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device_obj)
        input_tensor.requires_grad = True

        activations.clear()
        gradients.clear()

        preds = model(input_tensor)
        target_score = preds[0].max() if isinstance(preds, (list, tuple)) else preds.max()
        model.zero_grad()
        target_score.backward()

        if not activations or not gradients:
            continue

        act = activations[-1].detach()
        grad = gradients[-1].detach()

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam_resized = cv2.resize(cam, (w_orig, h_orig))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

        # Ghép ảnh gốc bên cạnh ảnh overlay nhiệt
        combined = np.hstack([orig_bgr, overlay])

        save_name = f"{img_path.stem}_gradcam.png"
        cv2.imwrite(str(out_p / save_name), combined)

    fwd_handle.remove()
    bwd_handle.remove()
    print(f"✅ Hoàn tất lưu toàn bộ Grad-CAM vào: {out_p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trực quan hóa Grad-CAM cho mô hình YOLO trên Test Set.")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file best.pt")
    parser.add_argument("--test_file", type=str, required=True, help="Đường dẫn file danh sách test set (.txt)")
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
