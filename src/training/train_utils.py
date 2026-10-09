from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

AUG_PROFILES: Dict[str, Dict[str, Any]] = {
    "v2": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.15,
        "bgr": 0.0,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.1,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.5,
        "close_mosaic": 10,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "augmentations": [],
    },
    "v2_no_mosaic": {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.15,
        "bgr": 0.0,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.1,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "augmentations": [],
    },
    "legacy": {
        "close_mosaic": 5,
    },
}


def check_overwrite_safety(project: str, name: str, overwrite: bool) -> None:
    """Ngăn chặn ghi đè ngoài ý muốn khi checkpoint best.pt đã tồn tại."""
    best_checkpoint = Path(project) / name / "weights" / "best.pt"
    if best_checkpoint.exists() and not overwrite:
        raise SystemExit(
            f"❌ LỖI AN TOÀN: Run '{name}' đã có checkpoint hoàn tất tại:\n"
            f"   {best_checkpoint.resolve()}\n"
            f"👉 Thêm tham số '--overwrite' nếu bạn thực sự muốn chạy lại và ghi đè kết quả."
        )


def verify_args_yaml(save_dir: Path, aug_profile: str) -> bool:
    """Đối soát args.yaml thực tế được Ultralytics lưu với cấu hình profile kỳ vọng."""
    args_file = Path(save_dir) / "args.yaml"
    if not args_file.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy {args_file} để đối soát tham số.")
        return False

    with open(args_file, "r", encoding="utf-8") as f:
        actual_args = yaml.safe_load(f) or {}

    expected_cfg = AUG_PROFILES.get(aug_profile, {})
    mismatches: Dict[str, Tuple[Any, Any]] = {}

    for param, expected_val in expected_cfg.items():
        actual_val = actual_args.get(param)
        if param == "augmentations":
            if aug_profile != "legacy" and actual_val not in ([], None):
                mismatches["augmentations"] = (actual_val, [])
        elif actual_val != expected_val:
            mismatches[param] = (actual_val, expected_val)

    if mismatches:
        print(f"❌ CẢNH BÁO: args.yaml của YOLO không khớp hoàn toàn với profile '{aug_profile}':")
        for param, (act, exp) in mismatches.items():
            print(f"   - {param}: thực tế={act} | kỳ vọng={exp}")
        return False

    print(f"✅ Đối soát thành công: Toàn bộ tham số trong args.yaml khớp 100% profile '{aug_profile}'!")
    return True
