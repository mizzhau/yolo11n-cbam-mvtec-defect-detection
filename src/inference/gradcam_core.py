from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

MVTEC_CATEGORIES: List[str] = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Letterbox chuẩn YOLO giữ tỷ lệ khung hình với viền xám."""
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = float(new_shape[1] - new_unpad[0]) / 2
    dh = float(new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(new_shape[0] - new_unpad[1] - top))
    left = int(round(dw - 0.1))
    right = int(round(new_shape[1] - new_unpad[0] - left))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def unletterbox_heatmap(
    cam: np.ndarray,
    orig_shape: Tuple[int, int],
    pad: Tuple[float, float]
) -> np.ndarray:
    """Loại bỏ viền padding và nội suy bản đồ CAM về kích thước gốc (h, w)."""
    h_orig, w_orig = orig_shape
    h_cam, w_cam = cam.shape[:2]
    dw, dh = pad

    top = max(0, int(round(dh - 0.1)))
    bottom = min(h_cam, h_cam - int(round(dh - 0.1)))
    left = max(0, int(round(dw - 0.1)))
    right = min(w_cam, w_cam - int(round(dw - 0.1)))

    cropped = cam[top:bottom, left:right]
    if cropped.size == 0:
        return np.zeros((h_orig, w_orig), dtype=np.float32)

    return cv2.resize(cropped, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


def extract_category(image_path: Path) -> str:
    """Trích xuất category MVTec AD từ đường dẫn file."""
    stem_lower = image_path.stem.lower()
    for cat in sorted(MVTEC_CATEGORIES, key=len, reverse=True):
        if stem_lower.startswith(cat + "_"):
            return cat

    parts_lower = [p.lower() for p in image_path.parts]
    for cat in sorted(MVTEC_CATEGORIES, key=len, reverse=True):
        if cat in parts_lower:
            return cat

    return image_path.parent.name if image_path.parent.name != "images" else "unknown"


def stratified_sample_images(image_paths: List[Path], max_images: int = 30) -> List[Path]:
    """Lấy mẫu phân tầng đảm bảo bao phủ tối đa 15 categories của MVTec AD."""
    if not image_paths or max_images <= 0:
        return image_paths[:max_images] if max_images > 0 else image_paths

    cat_to_images: Dict[str, List[Path]] = {}
    for p in image_paths:
        cat = extract_category(p)
        cat_to_images.setdefault(cat, []).append(p)

    selected: List[Path] = []
    n_cats = len(cat_to_images)
    per_cat = max(1, max_images // n_cats)

    for cat in sorted(cat_to_images.keys()):
        imgs = cat_to_images[cat]
        selected.extend(imgs[:per_cat])
        if len(selected) >= max_images:
            return selected[:max_images]

    remaining = [p for p in image_paths if p not in selected]
    need = max_images - len(selected)
    selected.extend(remaining[:need])
    return selected


def find_target_layers(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module, str]]:
    """Xác định các layer nguồn cấp cho Detect head (P3, P4, P5) và CBAM nếu có."""
    detect_module = model.model[-1]
    source_indices = getattr(detect_module, "f", [])
    if isinstance(source_indices, int):
        source_indices = [source_indices]

    targets: List[Tuple[int, torch.nn.Module, str]] = []
    for idx in source_indices:
        layer = model.model[idx]
        name = f"DetectSource_L{idx}_{type(layer).__name__}"
        targets.append((idx, layer, name))

    for idx, layer in enumerate(model.model):
        if type(layer).__name__ == "CBAM":
            targets.append((idx, layer, f"CBAM_L{idx}"))

    if not targets:
        targets.append((len(model.model) - 2, model.model[-2], "Fallback_Penultimate"))

    return targets


def compute_gradcam(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    img_size: int = 640,
    target_cls: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính toán Grad-CAM đa tầng chuẩn xác, đảm bảo Gradient > 0 và giữ nguyên tỷ lệ ảnh."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    orig_h, orig_w = image_bgr.shape[:2]
    lettered_bgr, ratio, pad = letterbox(image_bgr, new_shape=(img_size, img_size))
    lettered_rgb = cv2.cvtColor(lettered_bgr, cv2.COLOR_BGR2RGB)

    input_tensor = torch.from_numpy(lettered_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    targets = find_target_layers(model)
    activations: Dict[int, torch.Tensor] = {}
    gradients: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hooks(idx: int):
        def f_hook(mod: torch.nn.Module, inp: tuple, out: torch.Tensor) -> None:
            activations[idx] = out

        def b_hook(mod: torch.nn.Module, g_in: tuple, g_out: tuple) -> None:
            gradients[idx] = g_out[0]

        return f_hook, b_hook

    for idx, layer, _ in targets:
        fh, bh = make_hooks(idx)
        handles.append(layer.register_forward_hook(fh))
        handles.append(layer.register_full_backward_hook(bh))

    try:
        preds = model(input_tensor)
        raw_out = preds[0] if isinstance(preds, (tuple, list)) else preds

        cls_scores = raw_out[:, 4:, :]
        if target_cls is not None and 0 <= target_cls < cls_scores.shape[1]:
            target_score = cls_scores[:, target_cls, :].max()
        else:
            target_score = cls_scores.max()

        model.zero_grad()
        target_score.backward()

        cams: List[np.ndarray] = []
        for idx, _, _ in targets:
            g = gradients.get(idx)
            a = activations.get(idx)
            if g is not None and a is not None and g.abs().sum() > 0:
                weights = torch.mean(g, dim=(2, 3), keepdim=True)
                cam = F.relu(torch.sum(weights * a, dim=1, keepdim=True))
                cam_up = F.interpolate(
                    cam, size=(img_size, img_size), mode="bilinear", align_corners=False
                )
                cams.append(cam_up.squeeze().detach().cpu().numpy())

        if not cams:
            for idx, _, _ in targets:
                a = activations.get(idx)
                if a is not None:
                    cam = torch.mean(a, dim=1, keepdim=True)
                    cam_up = F.interpolate(
                        cam, size=(img_size, img_size), mode="bilinear", align_corners=False
                    )
                    cams.append(cam_up.squeeze().detach().cpu().numpy())
                    break

        if cams:
            aggregated = np.sum(cams, axis=0)
            c_min, c_max = float(aggregated.min()), float(aggregated.max())
            diff = c_max - c_min
            norm_cam = (aggregated - c_min) / diff if diff > 1e-25 else np.zeros_like(aggregated)
        else:
            norm_cam = np.zeros((img_size, img_size), dtype=np.float32)

        cam_orig = unletterbox_heatmap(norm_cam, (orig_h, orig_w), pad)
        cam_orig = np.clip(cam_orig, 0.0, 1.0)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_orig), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
        return overlay, heatmap, cam_orig
    finally:
        for h in handles:
            h.remove()
