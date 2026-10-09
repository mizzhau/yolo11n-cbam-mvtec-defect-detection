import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.models.cbam import register_cbam_to_ultralytics
from src.training.train_utils import AUG_PROFILES, check_overwrite_safety, verify_args_yaml


def train_cbam(
    data_yaml: str,
    model_cfg: str = "configs/models/yolo11n_cbam_backbone.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    save_period: int = 10,
    project: str = "experiments/cbam",
    name: str = "cbam_backbone",
    aug_profile: str = "v2",
    overwrite: bool = False,
    drive_backup: str = "",
    device: str = "",
    seed: int = 42,
    fraction: float = 1.0
) -> None:
    register_cbam_to_ultralytics()
    from ultralytics import YOLO

    check_overwrite_safety(project=project, name=name, overwrite=overwrite)

    if aug_profile not in AUG_PROFILES:
        raise ValueError(
            f"Augmentation profile '{aug_profile}' không hợp lệ. "
            f"Chọn một trong: {list(AUG_PROFILES.keys())}"
        )

    print("=" * 70)
    print(" KHỞI CHẠY HUẤN LUYỆN: YOLO11n + CBAM (ABLATION EXPERIMENT)")
    print(f" - Model config:  {model_cfg}")
    print(f" - Dataset:       {data_yaml}")
    print(f" - Epochs:        {epochs}")
    print(f" - Batch size:    {batch_size}")
    print(f" - Image size:    {img_size}")
    print(f" - Fraction:      {fraction}")
    print(f" - Aug profile:   {aug_profile}")
    print(f" - Save period:   {save_period} (Lưu checkpoint mỗi {save_period} epoch)")
    print(f" - Project dir:   {project}/{name}")
    print("=" * 70)

    model = YOLO(model_cfg)

    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "fraction": fraction,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 2.0,
        "save": True,
        "save_period": save_period,
        "project": project,
        "name": name,
        "seed": seed,
        "plots": True,
        "exist_ok": True,
    }
    train_kwargs.update(AUG_PROFILES[aug_profile])

    if device:
        train_kwargs["device"] = device

    results = model.train(**train_kwargs)
    print("Hoàn tất huấn luyện YOLO11n + CBAM!")

    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / name))
    verify_args_yaml(save_dir, aug_profile)

    if drive_backup:
        backup_path = Path(drive_backup)
        backup_path.mkdir(parents=True, exist_ok=True)
        print(f"Đang sao lưu kết quả sang Google Drive: {backup_path}...")
        for item in save_dir.glob("*"):
            dest = backup_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        print("Đã sao lưu an toàn sang Google Drive!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO11n + CBAM.")
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data/exp_2/data_70_15_15.yaml",
        help="Đường dẫn dataset YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="configs/models/yolo11n_cbam_backbone.yaml",
        help="File cấu hình kiến trúc YOLO11n + CBAM"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save_period", type=int, default=10, help="Lưu checkpoint sau mỗi N epochs")
    parser.add_argument("--project", type=str, default="experiments/cbam")
    parser.add_argument("--name", type=str, default="cbam_backbone")
    parser.add_argument(
        "--aug_profile",
        type=str,
        choices=list(AUG_PROFILES.keys()),
        default="v2",
        help="Profile augmentation (v2: chuẩn đề tài, v2_no_mosaic: đối chứng, legacy: YOLO cũ)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép ghi đè nếu checkpoint best.pt đã tồn tại"
    )
    parser.add_argument("--drive_backup", type=str, default="", help="Đường dẫn sao lưu Google Drive")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=1.0, help="Tỷ lệ dataset (0.01 = 1%%)")
    args = parser.parse_args()

    train_cbam(
        data_yaml=args.data,
        model_cfg=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        save_period=args.save_period,
        project=args.project,
        name=args.name,
        aug_profile=args.aug_profile,
        overwrite=args.overwrite,
        drive_backup=args.drive_backup,
        device=args.device,
        seed=args.seed,
        fraction=args.fraction
    )


if __name__ == "__main__":
    main()
