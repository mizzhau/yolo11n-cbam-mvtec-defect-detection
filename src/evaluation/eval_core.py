import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def evaluate_test_core(
    weights_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "",
    out_json: str = ""
) -> Dict[str, Any]:
    """Đánh giá mô hình trên tập Test Lõi (test_core.txt)."""
    w_path = Path(weights_path)
    if not w_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {w_path}")

    if "cbam" in str(w_path).lower():
        from src.models.cbam import register_cbam_to_ultralytics
        register_cbam_to_ultralytics()

    from ultralytics import YOLO

    print("=" * 70)
    print(" ĐÁNH GIÁ TRÊN TẬP TEST LÕI (CORE TEST EVALUATION)")
    print(f" - Trọng số:   {weights_path}")
    print(f" - Data YAML:  {data_yaml}")
    print(f" - Image size: {img_size}")
    print(f" - Batch size: {batch_size}")
    print("=" * 70)

    model = YOLO(weights_path)

    val_kwargs: Dict[str, Any] = {
        "data": data_yaml,
        "split": "test",
        "imgsz": img_size,
        "batch": batch_size,
        "plots": False,
        "verbose": True
    }
    if device:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)

    p = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
    r = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
    f1 = 2.0 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    speed_info = getattr(metrics, "speed", {})
    infer_ms = float(speed_info.get("inference", 0.0))
    total_ms = sum(float(v) for v in speed_info.values())
    fps = round(1000.0 / total_ms, 2) if total_ms > 0 else 0.0

    total_params = sum(param.numel() for param in model.model.parameters())
    params_m = round(total_params / 1e6, 3)

    summary = {
        "weights": str(w_path),
        "data": data_yaml,
        "mAP@0.5": round(map50, 4),
        "mAP@0.5:0.95": round(map50_95, 4),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f1, 4),
        "Latency_ms": round(infer_ms, 2),
        "FPS": fps,
        "Params_M": params_m
    }

    print("\n[KẾT QUẢ TEST LÕI]")
    for k, v in summary.items():
        print(f" - {k}: {v}")

    if out_json:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"-> Đã lưu kết quả đánh giá Test Lõi vào: {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập Test Lõi (test_core.txt).")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file trọng số best.pt")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file data yaml trỏ tới test_core")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out", type=str, default="", help="Đường dẫn file json xuất kết quả")
    args = parser.parse_args()

    evaluate_test_core(
        weights_path=args.weights,
        data_yaml=args.data,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        out_json=args.out
    )


if __name__ == "__main__":
    main()
