import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from src.models.cbam import register_cbam_to_ultralytics


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def benchmark_model(
    weights_path: str,
    data_yaml: str,
    img_size: int = 640,
    batch_size: int = 1,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "",
    out_file: str = ""
) -> Dict[str, Any]:
    w_p = Path(weights_path)
    if not w_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {w_p}")

    if "cbam" in str(w_p).lower():
        register_cbam_to_ultralytics()

    from ultralytics import YOLO

    device_str = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_str)

    yolo = YOLO(weights_path)
    model = yolo.model.to(device_obj)
    model.eval()


    total_params = sum(p.numel() for p in model.parameters())
    params_m = round(total_params / 1e6, 3)


    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    if device_obj.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    mean_latency_ms = round((total_time / num_runs) * 1000.0, 2)
    fps = round((num_runs * batch_size) / total_time, 2)

    val_metrics = yolo.val(
        data=data_yaml,
        split="test",
        imgsz=img_size,
        batch=16,
        plots=False,
        verbose=False,
        device=device_str
    )

    p = float(val_metrics.results_dict.get("metrics/precision(B)", 0.0))
    r = float(val_metrics.results_dict.get("metrics/recall(B)", 0.0))
    map50 = float(val_metrics.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
    f1 = calculate_f1(p, r)

    summary = {
        "model_name": w_p.parent.parent.name,
        "weights": str(w_p),
        "device": device_str,
        "img_size": img_size,
        "parameters_M": params_m,
        "latency_ms": mean_latency_ms,
        "fps": fps,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "mAP@0.5": round(map50, 4),
        "mAP@0.5:0.95": round(map50_95, 4)
    }

    print("=" * 75)
    print(" KẾT QUẢ ĐO ĐẠC ĐỘ TRỄ & HIỆU NĂNG SUY LUẬN THỰC TẾ")
    print(f" - Mô hình:        {summary['model_name']}")
    print(f" - Thiết bị:       {summary['device']}")
    print(f" - Tham số:        {summary['parameters_M']} M")
    print(f" - Độ trễ:         {summary['latency_ms']} ms / ảnh")
    print(f" - Tốc độ (FPS):   {summary['fps']} khung hình/giây")
    print(f" - Precision:      {summary['precision']}")
    print(f" - Recall:         {summary['recall']}")
    print(f" - F1-Score:       {summary['f1_score']}")
    print(f" - mAP@0.5:        {summary['mAP@0.5']}")
    print(f" - mAP@0.5:0.95:   {summary['mAP@0.5:0.95']}")
    print("=" * 75)

    if out_file:
        out_p = Path(out_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() == ".json":
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        else:
            with open(out_p, "w", encoding="utf-8") as f:
                f.write(f"# Hiệu Năng Inference Thực Tế: {summary['model_name']}\n\n")
                f.write("| Chỉ Số Đánh Giá | Giá Trị Thực Tế |\n")
                f.write("|:---|:---:|\n")
                f.write(f"| **Độ trễ (Latency)** | **{summary['latency_ms']} ms** |\n")
                f.write(f"| **Tốc độ (Throughput)** | **{summary['fps']} FPS** |\n")
                f.write(f"| **Số tham số (Params)** | **{summary['parameters_M']} M** |\n")
                f.write(f"| **Precision** | {summary['precision']} |\n")
                f.write(f"| **Recall** | {summary['recall']} |\n")
                f.write(f"| **F1-Score** | **{summary['f1_score']}** |\n")
                f.write(f"| **mAP@0.5** | **{summary['mAP@0.5']}** |\n")
                f.write(f"| **mAP@0.5:0.95** | {summary['mAP@0.5:0.95']} |\n")
        print(f"✅ Đã lưu kết quả đo đạc vào: {out_p}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo đạc độ trễ suy luận, FPS, Params và F1-Score của mô hình YOLO.")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn file best.pt")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn file data yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out", type=str, default="", help="Đường dẫn file xuất kết quả (.json hoặc .md)")
    args = parser.parse_args()

    benchmark_model(
        weights_path=args.weights,
        data_yaml=args.data,
        img_size=args.imgsz,
        batch_size=args.batch,
        num_warmup=args.warmup,
        num_runs=args.runs,
        device=args.device,
        out_file=args.out
    )


if __name__ == "__main__":
    main()
