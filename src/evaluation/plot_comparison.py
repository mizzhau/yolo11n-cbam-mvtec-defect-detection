import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import pandas as pd


def load_ultralytics_results(results_csv: Path) -> pd.DataFrame:
    if not results_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy results.csv tại: {results_csv}")
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def plot_learning_curves(
    base_df: pd.DataFrame,
    cbam_df: pd.DataFrame,
    output_path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train Box Loss
    ax = axes[0, 0]
    if "train/box_loss" in base_df.columns and "train/box_loss" in cbam_df.columns:
        ax.plot(base_df["epoch"], base_df["train/box_loss"], label="Baseline (YOLO11n)", color="tab:blue", lw=2)
        ax.plot(cbam_df["epoch"], cbam_df["train/box_loss"], label="Main (YOLO11n+CBAM)", color="tab:orange", lw=2)
        ax.set_title("Train Box Loss", fontsize=12, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    # 2. Train Class Loss
    ax = axes[0, 1]
    if "train/cls_loss" in base_df.columns and "train/cls_loss" in cbam_df.columns:
        ax.plot(base_df["epoch"], base_df["train/cls_loss"], label="Baseline (YOLO11n)", color="tab:blue", lw=2)
        ax.plot(cbam_df["epoch"], cbam_df["train/cls_loss"], label="Main (YOLO11n+CBAM)", color="tab:orange", lw=2)
        ax.set_title("Train Class Loss", fontsize=12, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    # 3. mAP@0.5
    ax = axes[1, 0]
    col_map50 = "metrics/mAP50(B)"
    if col_map50 in base_df.columns and col_map50 in cbam_df.columns:
        ax.plot(base_df["epoch"], base_df[col_map50], label="Baseline (YOLO11n)", color="tab:blue", marker="o", lw=2)
        ax.plot(cbam_df["epoch"], cbam_df[col_map50], label="Main (YOLO11n+CBAM)", color="tab:orange", marker="s", lw=2)
        ax.set_title("Validation mAP@0.5", fontsize=12, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP@0.5")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    # 4. mAP@0.5:0.95
    ax = axes[1, 1]
    col_map5095 = "metrics/mAP50-95(B)"
    if col_map5095 in base_df.columns and col_map5095 in cbam_df.columns:
        ax.plot(base_df["epoch"], base_df[col_map5095], label="Baseline (YOLO11n)", color="tab:blue", marker="o", lw=2)
        ax.plot(cbam_df["epoch"], cbam_df[col_map5095], label="Main (YOLO11n+CBAM)", color="tab:orange", marker="s", lw=2)
        ax.set_title("Validation mAP@0.5:0.95", fontsize=12, weight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP@0.5:0.95")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ Learning Curves: {output_path}")


def plot_metrics_comparison_bar(
    base_df: pd.DataFrame,
    cbam_df: pd.DataFrame,
    output_path: Path
) -> pd.DataFrame:
    metrics = {
        "mAP@0.5": ("metrics/mAP50(B)", "max"),
        "mAP@0.5:0.95": ("metrics/mAP50-95(B)", "max"),
        "Precision": ("metrics/precision(B)", "max"),
        "Recall": ("metrics/recall(B)", "max"),
    }

    data = []
    for label, (col, agg) in metrics.items():
        base_val = getattr(base_df[col], agg)() if col in base_df.columns else 0.0
        cbam_val = getattr(cbam_df[col], agg)() if col in cbam_df.columns else 0.0
        diff = cbam_val - base_val
        diff_pct = (diff / base_val * 100) if base_val > 0 else 0.0
        data.append({
            "Metric": label,
            "Baseline (YOLO11n)": round(base_val, 4),
            "YOLO11n + CBAM": round(cbam_val, 4),
            "Diff (Abs)": round(diff, 4),
            "Diff (%)": f"{diff_pct:+.2f}%"
        })

    summary_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(summary_df))
    width = 0.35

    ax.bar([i - width / 2 for i in x], summary_df["Baseline (YOLO11n)"], width, label="Baseline (YOLO11n)", color="tab:blue")
    ax.bar([i + width / 2 for i in x], summary_df["YOLO11n + CBAM"], width, label="YOLO11n + CBAM", color="tab:orange")

    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["Metric"], fontsize=11, weight="bold")
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("So sánh Chỉ số Đánh giá Cao nhất", fontsize=13, weight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()

    for i in x:
        b_val = summary_df.loc[i, "Baseline (YOLO11n)"]
        c_val = summary_df.loc[i, "YOLO11n + CBAM"]
        ax.text(i - width / 2, b_val + 0.02, f"{b_val:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, c_val + 0.02, f"{c_val:.3f}", ha="center", fontsize=9, weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ so sánh chỉ số: {output_path}")

    return summary_df


def compare_runs(baseline_dir: Path, cbam_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_df = load_ultralytics_results(baseline_dir / "results.csv")
    cbam_df = load_ultralytics_results(cbam_dir / "results.csv")

    plot_learning_curves(base_df, cbam_df, output_dir / "learning_curves_comparison.png")
    summary_df = plot_metrics_comparison_bar(base_df, cbam_df, output_dir / "metrics_bar_comparison.png")

    summary_csv = output_dir / "summary_comparison.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_md = output_dir / "summary_comparison.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Bảng So Sánh Kết Quả Huấn Luyện\n\n")
        try:
            f.write(summary_df.to_markdown(index=False))
        except Exception:
            cols = list(summary_df.columns)
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in summary_df.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
        f.write("\n")

    print(f"Hoàn tất đánh giá so sánh! Kết quả lưu tại: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá so sánh kết quả huấn luyện Baseline vs CBAM.")
    parser.add_argument("--baseline_dir", type=str, required=True, help="Thư mục kết quả Baseline")
    parser.add_argument("--cbam_dir", type=str, required=True, help="Thư mục kết quả CBAM")
    parser.add_argument("--output_dir", type=str, default="experiments/ablation/testing_20epochs", help="Thư mục xuất biểu đồ")
    args = parser.parse_args()

    compare_runs(Path(args.baseline_dir), Path(args.cbam_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
