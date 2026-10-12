import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
import pandas as pd

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


def get_run_epoch_progress(project: str, name: str) -> Tuple[int, bool, bool]:
    """Kiểm tra tiến độ thực tế của một run: (số epoch hoàn thành, có last.pt, có best.pt)."""
    run_dir = Path(project) / name
    csv_file = run_dir / "results.csv"
    last_ckpt = run_dir / "weights" / "last.pt"
    best_ckpt = run_dir / "weights" / "best.pt"

    completed_epochs = 0
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            completed_epochs = len(df)
        except Exception:
            completed_epochs = 0

    return completed_epochs, last_ckpt.exists(), best_ckpt.exists()


def check_overwrite_safety(
    project: str,
    name: str,
    overwrite: bool = False,
    resume: bool = False
) -> None:
    """Kiểm tra an toàn ghi đè và hỗ trợ chế độ tiếp tục (resume) huấn luyện."""
    run_dir = Path(project) / name
    last_checkpoint = run_dir / "weights" / "last.pt"
    best_checkpoint = run_dir / "weights" / "best.pt"

    if resume:
        if not last_checkpoint.exists():
            raise SystemExit(
                f"❌ LỖI RESUME: Không tìm thấy checkpoint 'last.pt' tại:\n"
                f"   {last_checkpoint.resolve()}\n"
                f"👉 Kiểm tra lại thư mục chạy '{name}' hoặc bỏ cờ '--resume' để huấn luyện từ đầu."
            )
        print(f"🔄 CHẾ ĐỘ RESUME: Tìm thấy checkpoint tại {last_checkpoint.resolve()}")
        return

    if overwrite:
        if run_dir.exists():
            print(f"⚠️ CỜ OVERWRITE: Đang dọn dẹp weights/ và logs cũ của run '{name}'...")
            weights_dir = run_dir / "weights"
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
            for fname in ("results.csv", "args.yaml"):
                target_f = run_dir / fname
                if target_f.exists():
                    target_f.unlink()
        return

    if best_checkpoint.exists() and not overwrite:
        raise SystemExit(
            f"❌ LỖI AN TOÀN: Run '{name}' đã có checkpoint hoàn tất tại:\n"
            f"   {best_checkpoint.resolve()}\n"
            f"👉 Thêm tham số '--overwrite' nếu bạn thực sự muốn chạy lại từ đầu, "
            f"hoặc dùng '--resume' để tiếp tục."
        )

    if last_checkpoint.exists() and not overwrite:
        raise SystemExit(
            f"❌ LỖI AN TOÀN: Run '{name}' đã có checkpoint dở dang tại:\n"
            f"   {last_checkpoint.resolve()}\n"
            f"👉 Dùng cờ '--resume' để tiếp tục từ điểm ngắt, "
            f"hoặc '--overwrite' nếu muốn xóa và chạy lại từ đầu."
        )


def verify_args_yaml(
    save_dir: Path,
    aug_profile: str,
    expected_train_args: Optional[Dict[str, Any]] = None
) -> bool:
    """Đối soát args.yaml thực tế được Ultralytics lưu với profile và training args kỳ vọng."""
    args_file = Path(save_dir) / "args.yaml"
    if not args_file.exists():
        print(f"⚠️ Cảnh báo: Không tìm thấy {args_file} để đối soát tham số.")
        return False

    with open(args_file, "r", encoding="utf-8") as f:
        actual_args = yaml.safe_load(f) or {}

    expected_cfg = {**AUG_PROFILES.get(aug_profile, {}), **(expected_train_args or {})}
    mismatches: Dict[str, Tuple[Any, Any]] = {}

    for param, expected_val in expected_cfg.items():
        actual_val = actual_args.get(param)
        if param == "augmentations":
            if aug_profile != "legacy" and actual_val not in ([], None):
                mismatches["augmentations"] = (actual_val, [])
        elif actual_val != expected_val:
            mismatches[param] = (actual_val, expected_val)

    if mismatches:
        print(f"❌ CẢNH BÁO: args.yaml của YOLO không khớp hoàn toàn với cấu hình kỳ vọng:")
        for param, (act, exp) in mismatches.items():
            print(f"   - {param}: thực tế={act} | kỳ vọng={exp}")
        return False

    print(
        f"✅ Đối soát thành công: Toàn bộ tham số trong args.yaml khớp 100% cấu hình kỳ vọng (profile '{aug_profile}')!"
    )
    return True
