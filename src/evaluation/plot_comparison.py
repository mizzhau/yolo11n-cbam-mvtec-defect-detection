import argparse

import os

import sys

from pathlib import Path

from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):

    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy as np

import pandas as pd

plt.rcParams.update({

    "font.family": "serif",

    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],

    "mathtext.fontset": "stix",

    "axes.edgecolor": "#333333",

    "axes.linewidth": 1.0,

    "axes.grid": True,

    "grid.color": "#E5E7EB",

    "grid.linestyle": ":",

    "grid.linewidth": 0.8,

    "grid.alpha": 0.75,

    "figure.dpi": 300,

    "savefig.dpi": 300,

    "savefig.bbox": "tight",

    "savefig.pad_inches": 0.08,

})

                                                   

COLOR_BASELINE = "#1F4E79"                        

COLOR_CBAM = "#C0392B"                      

COLOR_ACCENT = "#D35400"                   

COLOR_POS = "#1E8449"                                         

COLOR_NEG = "#A93226"                                         

COLOR_MUTED = "#5D6D7E"                    

PALETTE_ABLATION = [

    "#2C3E50",                         

    "#2980B9",                              

    "#16A085",                          

    "#8E44AD",                               

    "#D35400",                                    

    "#C0392B",                                           

]

def load_ultralytics_results(results_csv: Path) -> pd.DataFrame:

    """Đọc file results.csv của Ultralytics và chuẩn hóa tên cột."""

    if not results_csv.exists():

        raise FileNotFoundError(f"Không tìm thấy results.csv tại: {results_csv}")

    df = pd.read_csv(results_csv)

    df.columns = [c.strip() for c in df.columns]

    return df

def calculate_f1(precision: float, recall: float) -> float:

    """Tính F1-score từ Precision và Recall."""

    if precision + recall <= 0:

        return 0.0

    return 2.0 * (precision * recall) / (precision + recall)

def despine(ax: plt.Axes) -> None:

    """Loại bỏ viền trên và viền phải theo nguyên lý Data-Ink của Edward Tufte."""

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#333333")

    ax.spines["bottom"].set_color("#333333")

def plot_learning_curves(

    base_df: pd.DataFrame,

    cbam_df: pd.DataFrame,

    output_path: Path,

    base_label: str = "Baseline (YOLO11n)",

    cbam_label: str = "YOLO11n + CBAM"

) -> None:

    """Vẽ biểu đồ quá trình học (Learning Curves) gồm 4 subplots chuẩn khoa học."""

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    epoch_base = base_df["epoch"]

    epoch_cbam = cbam_df["epoch"]

                                    

    ax = axes[0, 0]

    if "train/box_loss" in base_df.columns and "train/box_loss" in cbam_df.columns:

        ax.plot(epoch_base, base_df["train/box_loss"], label=base_label,

                color=COLOR_BASELINE, linestyle="-", lw=1.8, alpha=0.9)

        ax.plot(epoch_cbam, cbam_df["train/box_loss"], label=cbam_label,

                color=COLOR_CBAM, linestyle="--", lw=1.8, alpha=0.9)

        ax.set_title("(a) Training Box Loss", fontsize=12, fontweight="bold", pad=8)

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")

        ax.set_ylabel("Bounding Box Loss", fontsize=11, fontweight="bold")

        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=9.5)

        despine(ax)

                                      

    ax = axes[0, 1]

    if "train/cls_loss" in base_df.columns and "train/cls_loss" in cbam_df.columns:

        ax.plot(epoch_base, base_df["train/cls_loss"], label=base_label,

                color=COLOR_BASELINE, linestyle="-", lw=1.8, alpha=0.9)

        ax.plot(epoch_cbam, cbam_df["train/cls_loss"], label=cbam_label,

                color=COLOR_CBAM, linestyle="--", lw=1.8, alpha=0.9)

        ax.set_title("(b) Training Class Loss", fontsize=12, fontweight="bold", pad=8)

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")

        ax.set_ylabel("Classification Loss", fontsize=11, fontweight="bold")

        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=9.5)

        despine(ax)

                                        

    ax = axes[1, 0]

    col_map50 = "metrics/mAP50(B)"

    if col_map50 in base_df.columns and col_map50 in cbam_df.columns:

        ax.plot(epoch_base, base_df[col_map50], label=base_label,

                color=COLOR_BASELINE, linestyle="-", lw=1.8, marker="o",

                markevery=10, markersize=5, alpha=0.85)

        ax.plot(epoch_cbam, cbam_df[col_map50], label=cbam_label,

                color=COLOR_CBAM, linestyle="--", lw=1.8, marker="s",

                markevery=10, markersize=5, alpha=0.85)

                                               

        base_best_idx = base_df[col_map50].idxmax()

        cbam_best_idx = cbam_df[col_map50].idxmax()

        base_best_ep = int(base_df.loc[base_best_idx, "epoch"])

        base_best_val = float(base_df.loc[base_best_idx, col_map50])

        cbam_best_ep = int(cbam_df.loc[cbam_best_idx, "epoch"])

        cbam_best_val = float(cbam_df.loc[cbam_best_idx, col_map50])

        ax.scatter([base_best_ep], [base_best_val], color=COLOR_BASELINE,

                   s=120, marker="*", zorder=5, edgecolor="black", linewidth=0.7)

        ax.scatter([cbam_best_ep], [cbam_best_val], color=COLOR_CBAM,

                   s=120, marker="*", zorder=5, edgecolor="black", linewidth=0.7)

                                         

        ax.annotate(f"Best: {base_best_val:.4f}\n(Ep {base_best_ep})",

                    xy=(base_best_ep, base_best_val),

                    xytext=(-35, -28), textcoords="offset points",

                    fontsize=8.5, color=COLOR_BASELINE, fontweight="bold",

                    arrowprops=dict(arrowstyle="->", color=COLOR_BASELINE, lw=0.8))

        ax.annotate(f"Best: {cbam_best_val:.4f}\n(Ep {cbam_best_ep})",

                    xy=(cbam_best_ep, cbam_best_val),

                    xytext=(-35, 12), textcoords="offset points",

                    fontsize=8.5, color=COLOR_CBAM, fontweight="bold",

                    arrowprops=dict(arrowstyle="->", color=COLOR_CBAM, lw=0.8))

        ax.set_title(r"(c) Validation $\text{mAP}_{50}$", fontsize=12, fontweight="bold", pad=8)

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")

        ax.set_ylabel(r"$\text{mAP}_{50}$", fontsize=11, fontweight="bold")

        ax.set_ylim(-0.02, max(base_best_val, cbam_best_val) + 0.08)

        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=9.5, loc="lower right")

        despine(ax)

                                             

    ax = axes[1, 1]

    col_map5095 = "metrics/mAP50-95(B)"

    if col_map5095 in base_df.columns and col_map5095 in cbam_df.columns:

        ax.plot(epoch_base, base_df[col_map5095], label=base_label,

                color=COLOR_BASELINE, linestyle="-", lw=1.8, marker="o",

                markevery=10, markersize=5, alpha=0.85)

        ax.plot(epoch_cbam, cbam_df[col_map5095], label=cbam_label,

                color=COLOR_CBAM, linestyle="--", lw=1.8, marker="s",

                markevery=10, markersize=5, alpha=0.85)

        base_best_idx = base_df[col_map5095].idxmax()

        cbam_best_idx = cbam_df[col_map5095].idxmax()

        base_best_ep = int(base_df.loc[base_best_idx, "epoch"])

        base_best_val = float(base_df.loc[base_best_idx, col_map5095])

        cbam_best_ep = int(cbam_df.loc[cbam_best_idx, "epoch"])

        cbam_best_val = float(cbam_df.loc[cbam_best_idx, col_map5095])

        ax.scatter([base_best_ep], [base_best_val], color=COLOR_BASELINE,

                   s=120, marker="*", zorder=5, edgecolor="black", linewidth=0.7)

        ax.scatter([cbam_best_ep], [cbam_best_val], color=COLOR_CBAM,

                   s=120, marker="*", zorder=5, edgecolor="black", linewidth=0.7)

        ax.set_title(r"(d) Validation $\text{mAP}_{50-95}$", fontsize=12, fontweight="bold", pad=8)

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")

        ax.set_ylabel(r"$\text{mAP}_{50-95}$", fontsize=11, fontweight="bold")

        ax.set_ylim(-0.02, max(base_best_val, cbam_best_val) + 0.08)

        ax.legend(frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=9.5, loc="lower right")

        despine(ax)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)

    plt.close()

    print(f"-> Đã lưu biểu đồ Learning Curves chuẩn khoa học: {output_path}")

def plot_metrics_comparison_bar(

    summary_df: pd.DataFrame,

    output_path: Path,

    base_name: str = "Baseline (YOLO11n)",

    cbam_name: str = "YOLO11n + CBAM",

    chart_title: Optional[str] = None

) -> None:

    """Vẽ biểu đồ cột so sánh các chỉ số hiệu năng chuẩn publication."""

    fig, ax = plt.subplots(figsize=(10.5, 6))

    n_metrics = len(summary_df)

    x = np.arange(n_metrics)

    bar_width = 0.35

                                         

    label_map = {

        "mAP@0.5": r"$\mathbf{mAP}_{50}$",

        "mAP@0.5:0.95": r"$\mathbf{mAP}_{50-95}$",

        "Precision": r"$\mathbf{Precision}$",

        "Recall": r"$\mathbf{Recall}$",

        "F1": r"$\mathbf{F_1}\text{-}\mathbf{Score}$",

        "F1-Score": r"$\mathbf{F_1}\text{-}\mathbf{Score}$"

    }

    x_labels = [label_map.get(m, m) for m in summary_df["Metric"]]

    bars_base = ax.bar(

        x - bar_width / 2,

        summary_df["Baseline (YOLO11n)"],

        bar_width,

        label=base_name,

        color=COLOR_BASELINE,

        edgecolor="#163A5C",

        linewidth=1.0,

        zorder=3

    )

    bars_cbam = ax.bar(

        x + bar_width / 2,

        summary_df["YOLO11n + CBAM"],

        bar_width,

        label=cbam_name,

        color=COLOR_CBAM,

        edgecolor="#8B1E16",

        linewidth=1.0,

        zorder=3

    )

    ax.set_xticks(x)

    ax.set_xticklabels(x_labels, fontsize=11.5)

    ax.set_ylabel("Score / Performance Metric", fontsize=11.5, fontweight="bold")

    title_text = chart_title if chart_title is not None else f"Ablation Performance Comparison: {base_name} vs. {cbam_name}"

    ax.set_title(title_text, fontsize=13, fontweight="bold", pad=15)

    ax.set_ylim(0, 1.10)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

                                                     

    for i in range(n_metrics):

        b_val = float(summary_df.loc[i, "Baseline (YOLO11n)"])

        c_val = float(summary_df.loc[i, "YOLO11n + CBAM"])

        diff_val = c_val - b_val

        diff_pct = (diff_val / b_val * 100) if b_val > 0 else 0.0

                              

        ax.text(

            x[i] - bar_width / 2,

            b_val + 0.015,

            f"{b_val:.4f}",

            ha="center",

            va="bottom",

            fontsize=9.5,

            color="#2C3E50",

            fontweight="normal"

        )

                          

        ax.text(

            x[i] + bar_width / 2,

            c_val + 0.015,

            f"{c_val:.4f}",

            ha="center",

            va="bottom",

            fontsize=9.5,

            color="#922B21",

            fontweight="bold"

        )

                                 

        max_val = max(b_val, c_val)

        delta_color = COLOR_POS if diff_pct >= 0 else COLOR_NEG

        sign_str = "+" if diff_pct > 0 else ""

        badge_text = f"({sign_str}{diff_pct:.2f}%)"

        ax.text(

            x[i] + bar_width / 2,

            max_val + 0.055,

            badge_text,

            ha="center",

            va="bottom",

            fontsize=8.5,

            fontweight="bold",

            color=delta_color,

            bbox=dict(

                boxstyle="round,pad=0.22",

                facecolor="#F8F9FA",

                edgecolor=delta_color,

                linewidth=0.8,

                alpha=0.9

            )

        )

    despine(ax)

    ax.legend(

        loc="upper left",

        frameon=True,

        framealpha=0.95,

        edgecolor="#CCCCCC",

        fontsize=10.5

    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)

    plt.close()

    print(f"-> Đã lưu biểu đồ Metrics Bar chuẩn khoa học: {output_path}")

def build_or_update_summary_df(

    existing_csv: Optional[Path],

    base_df: pd.DataFrame,

    cbam_df: pd.DataFrame

) -> pd.DataFrame:

    """Tạo hoặc cập nhật summary DataFrame đầy đủ 5 metrics kể cả F1-Score."""

                                                                                

    if existing_csv and existing_csv.exists():

        raw_df = pd.read_csv(existing_csv)

        records = []

        p_base, r_base = 0.0, 0.0

        p_cbam, r_cbam = 0.0, 0.0

        for _, row in raw_df.iterrows():

            m_name = str(row["Metric"]).strip()

            b_val = float(row["Baseline (YOLO11n)"])

            c_val = float(row["YOLO11n + CBAM"])

            diff_abs = c_val - b_val

            diff_pct_str = f"{(diff_abs / b_val * 100):+.2f}%" if b_val > 0 else "+0.00%"

            if m_name == "Precision":

                p_base, p_cbam = b_val, c_val

            elif m_name == "Recall":

                r_base, r_cbam = b_val, c_val

            records.append({

                "Metric": m_name,

                "Baseline (YOLO11n)": round(b_val, 4),

                "YOLO11n + CBAM": round(c_val, 4),

                "Diff (Abs)": round(diff_abs, 4),

                "Diff (%)": diff_pct_str

            })

                                                        

        has_f1 = any(r["Metric"] in ["F1", "F1-Score"] for r in records)

        if not has_f1 and p_base > 0 and r_base > 0:

            f1_base = calculate_f1(p_base, r_base)

            f1_cbam = calculate_f1(p_cbam, r_cbam)

            f1_diff = f1_cbam - f1_base

            f1_pct_str = f"{(f1_diff / f1_base * 100):+.2f}%" if f1_base > 0 else "+0.00%"

            records.append({

                "Metric": "F1-Score",

                "Baseline (YOLO11n)": round(f1_base, 4),

                "YOLO11n + CBAM": round(f1_cbam, 4),

                "Diff (Abs)": round(f1_diff, 4),

                "Diff (%)": f1_pct_str

            })

        return pd.DataFrame(records)

                                                                            

    col_map50 = "metrics/mAP50(B)"

    base_idx = base_df[col_map50].idxmax() if col_map50 in base_df.columns else len(base_df) - 1

    cbam_idx = cbam_df[col_map50].idxmax() if col_map50 in cbam_df.columns else len(cbam_df) - 1

    base_row = base_df.iloc[base_idx]

    cbam_row = cbam_df.iloc[cbam_idx]

    b_p = float(base_row.get("metrics/precision(B)", 0.0))

    b_r = float(base_row.get("metrics/recall(B)", 0.0))

    c_p = float(cbam_row.get("metrics/precision(B)", 0.0))

    c_r = float(cbam_row.get("metrics/recall(B)", 0.0))

    metrics = [

        ("mAP@0.5", float(base_row.get("metrics/mAP50(B)", 0.0)), float(cbam_row.get("metrics/mAP50(B)", 0.0))),

        ("mAP@0.5:0.95", float(base_row.get("metrics/mAP50-95(B)", 0.0)), float(cbam_row.get("metrics/mAP50-95(B)", 0.0))),

        ("Precision", b_p, c_p),

        ("Recall", b_r, c_r),

        ("F1-Score", calculate_f1(b_p, b_r), calculate_f1(c_p, c_r)),

    ]

    records = []

    for label, b_val, c_val in metrics:

        diff_abs = c_val - b_val

        diff_pct = (diff_abs / b_val * 100) if b_val > 0 else 0.0

        records.append({

            "Metric": label,

            "Baseline (YOLO11n)": round(b_val, 4),

            "YOLO11n + CBAM": round(c_val, 4),

            "Diff (Abs)": round(diff_abs, 4),

            "Diff (%)": f"{diff_pct:+.2f}%"

        })

    return pd.DataFrame(records)

def save_summary_tables(summary_df: pd.DataFrame, output_dir: Path, title: str) -> None:

    """Lưu bảng kết quả so sánh sang định dạng CSV và Markdown."""

    csv_path = output_dir / "summary_comparison.csv"

    summary_df.to_csv(csv_path, index=False)

    md_path = output_dir / "summary_comparison.md"

    with open(md_path, "w", encoding="utf-8") as f:

        f.write(f"# Bảng So Sánh Kết Quả Huấn Luyện: {title}\n\n")

        cols = list(summary_df.columns)

        f.write("| " + " | ".join(cols) + " |\n")

        f.write("| " + " | ".join([":---" if i == 0 else "---:" for i in range(len(cols))]) + " |\n")

        for _, row in summary_df.iterrows():

            f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")

        f.write("\n")

def plot_overall_ablation_summary(

    eval_root: Path,

    overall_data: List[Dict[str, any]]

) -> None:

    """Vẽ biểu đồ tổng hợp toàn diện Ablation Study so sánh đồng thời tất cả các mô hình."""

    if not overall_data:

        return

    df_all = pd.DataFrame(overall_data)

                                                                     

    champion_model_name = ""

    if not df_all.empty:

        champ_row = df_all.sort_values(by=["mAP@0.5", "F1"], ascending=False).iloc[0]

        champion_model_name = champ_row["Model"]

                                                             

    metrics = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1"]

    n_models = len(df_all)

    x = np.arange(len(metrics))

    total_group_width = 0.82

    bar_width = total_group_width / n_models

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for idx, row in df_all.iterrows():

        model_name = row["Model"]

        color = PALETTE_ABLATION[idx % len(PALETTE_ABLATION)]

        offset = (idx - (n_models - 1) / 2) * bar_width

        vals = [row[m] for m in metrics]

                                                        

        is_champion = (model_name == champion_model_name)

        edge_c = "#000000" if is_champion else "none"

        lw = 1.2 if is_champion else 0.0

        bars = ax.bar(

            x + offset, vals, bar_width * 0.92,

            label=f"{model_name}" + (" (Optimal)" if is_champion else ""),

            color=color, edgecolor=edge_c, linewidth=lw, zorder=3

        )

        for bar, val in zip(bars, vals):

            if val > 0:

                ax.text(

                    bar.get_x() + bar.get_width() / 2,

                    val + 0.012,

                    f"{val:.3f}",

                    ha="center",

                    va="bottom",

                    fontsize=8.0,

                    rotation=90 if n_models > 4 else 0,

                    color="#222222",

                    fontweight="bold" if is_champion else "normal"

                )

    label_map = {

        "mAP@0.5": r"$\mathbf{mAP}_{50}$",

        "mAP@0.5:0.95": r"$\mathbf{mAP}_{50-95}$",

        "Precision": r"$\mathbf{Precision}$",

        "Recall": r"$\mathbf{Recall}$",

        "F1": r"$\mathbf{F_1}\text{-}\mathbf{Score}$"

    }

    ax.set_xticks(x)

    ax.set_xticklabels([label_map.get(m, m) for m in metrics], fontsize=12)

    ax.set_ylabel("Evaluation Score", fontsize=12, fontweight="bold")

    ax.set_title("Overall Ablation Study Performance Comparison (100 Epochs)",

                 fontsize=14, fontweight="bold", pad=15)

    ax.set_ylim(0, 1.08)

    despine(ax)

    ax.legend(

        loc="upper right",

        bbox_to_anchor=(1.0, 1.0),

        frameon=True,

        framealpha=0.95,

        edgecolor="#CCCCCC",

        fontsize=9.5

    )

    plt.tight_layout()

    bar_path = eval_root / "overall_ablation_metrics_comparison.png"

    plt.savefig(bar_path)

    plt.close()

    print(f"-> Đã lưu biểu đồ tổng quan Ablation Metrics Bar: {bar_path}")

                                             

    md_path = eval_root / "overall_ablation_summary.md"

    with open(md_path, "w", encoding="utf-8") as f:

        f.write("# Bảng Tổng Hợp Ablation Study Toàn Diện (100 Epochs)\n\n")

        f.write("| Model | CBAM Position | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | Best Epoch |\n")

        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")

        for _, r in df_all.iterrows():

            best_ep = r.get("Best_Epoch", "-")

            f.write(f"| **{r['Model']}** | {r.get('Position', '-')} | {r['mAP@0.5']:.4f} | {r['mAP@0.5:0.95']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | **{r['F1']:.4f}** | {best_ep} |\n")

                                                           

        if not df_all.empty:

            best_row = df_all.sort_values(by=["mAP@0.5", "F1"], ascending=False).iloc[0]

            f.write(

                f"\n> ★ Mô hình đạt hiệu năng tối ưu toàn diện: **{best_row['Model']}** "

                f"({best_row.get('Position', '-')}) với mAP@0.5 = {best_row['mAP@0.5']:.4f}, "

                f"F1 = {best_row['F1']:.4f} tại Epoch {best_row.get('Best_Epoch', '-')}.\n"

            )

    print(f"-> Đã lưu bảng tóm tắt tổng quan Ablation: {md_path}")

def process_all_comparisons(

    base_dir: Path,

    eval_dir: Path,

    baseline_csv_override: Optional[Path] = None

) -> None:

    """Tự động xử lý và tái tạo toàn bộ biểu đồ khoa học cho tất cả các folder thí nghiệm."""

    eval_dir.mkdir(parents=True, exist_ok=True)

                                                 

    baseline_csv = None

    if baseline_csv_override and baseline_csv_override.exists():

        baseline_csv = baseline_csv_override

    else:

        candidate_baselines = [

            base_dir / "baseline" / "results.csv",

            base_dir.parent / "baseline" / "results.csv",

            Path("experiments/baseline/baseline/results.csv"),

            Path("experiments/baseline/results.csv"),

        ]

        for c in candidate_baselines:

            if c.exists():

                baseline_csv = c

                break

    base_df = load_ultralytics_results(baseline_csv) if baseline_csv else None

                              

    comparisons = [

        {

            "subfolder": "baseline vs cbam_backbone",

            "model_dir": "cbam_backbone",

            "label": "YOLO11n + CBAM (Backbone)",

            "pos": "Backbone (P3/P4/P5)",

            "chart_title": "Baseline vs YOLOv11n _ CBAM (BackBone)"

        },

        {

            "subfolder": "baseline vs cbam_neck",

            "model_dir": "cbam_neck",

            "label": "YOLO11n + CBAM (Neck)",

            "pos": "Neck (P3/P4/P5)",

            "chart_title": "Baseline vs YOLOv11n _ CBAM (Neck)"

        },

        {

            "subfolder": "baseline vs cbam_head",

            "model_dir": "cbam_head",

            "label": "YOLO11n + CBAM (Head)",

            "pos": "Head (P3/P4/P5)",

            "chart_title": "Baseline vs YOLOv11n _ CBAM (Head)"

        },

        {

            "subfolder": "baseline vs cbam_neck_head",

            "model_dir": "cbam_neck_head",

            "label": "YOLO11n + CBAM (Neck+Head)",

            "pos": "Neck + Head (P3/P4/P5)",

            "chart_title": "Baseline vs YOLOv11n _ CBAM (Neck + Head)"

        },

        {

            "subfolder": "baseline vs cbam_backbone_head",

            "model_dir": "cbam_backbone_head",

            "label": "YOLO11n + CBAM (Backbone+Head)",

            "pos": "Backbone + Head",

            "chart_title": "Baseline vs YOLOv11n _ CBAM (BackBone + Head)"

        },

    ]

                                                   

    overall_list = []

    if base_df is not None:

                                                                             

        col_map50 = "metrics/mAP50(B)"

        base_idx = int(base_df[col_map50].idxmax()) if col_map50 in base_df.columns else len(base_df) - 1

        base_row = base_df.iloc[base_idx]

        base_best_ep = int(base_row.get("epoch", base_idx))

        b_p = float(base_row.get("metrics/precision(B)", 0.0))

        b_r = float(base_row.get("metrics/recall(B)", 0.0))

        b_map50 = float(base_row.get("metrics/mAP50(B)", 0.0))

        b_map50_95 = float(base_row.get("metrics/mAP50-95(B)", 0.0))

        b_f1 = calculate_f1(b_p, b_r)

        overall_list.append({

            "Model": "Baseline (YOLO11n)",

            "Position": "—",

            "mAP@0.5": b_map50,

            "mAP@0.5:0.95": b_map50_95,

            "Precision": b_p,

            "Recall": b_r,

            "F1": b_f1,

            "Best_Epoch": base_best_ep

        })

    else:

        print(f"[THÔNG BÁO] Chưa tìm thấy baseline results.csv trong {base_dir}. Chế độ Standalone: chỉ xuất kết quả thực tế của các mô hình đã huấn luyện, không tạo dữ liệu giả Baseline.")

    for item in comparisons:

        out_folder = eval_dir / item["subfolder"]

        out_folder.mkdir(parents=True, exist_ok=True)

                                                   

        candidates = [item["model_dir"], item["model_dir"].replace("cbam_", "")]

        cbam_csv = None

        for cand in candidates:

            cand_path = base_dir / cand / "results.csv"

            if cand_path.exists():

                cbam_csv = cand_path

                break

        if not cbam_csv:

            print(f"[CẢNH BÁO] Bỏ qua {item['model_dir']} do không tìm thấy results.csv trong {base_dir}")

            continue

        cbam_df = load_ultralytics_results(cbam_csv)

                                                                    

        cbam_col_map50 = "metrics/mAP50(B)"

        cbam_idx = int(cbam_df[cbam_col_map50].idxmax()) if cbam_col_map50 in cbam_df.columns else len(cbam_df) - 1

        cbam_row = cbam_df.iloc[cbam_idx]

        cbam_best_ep = int(cbam_row.get("epoch", cbam_idx))

        c_p = float(cbam_row.get("metrics/precision(B)", 0.0))

        c_r = float(cbam_row.get("metrics/recall(B)", 0.0))

        c_map50 = float(cbam_row.get("metrics/mAP50(B)", 0.0))

        c_map50_95 = float(cbam_row.get("metrics/mAP50-95(B)", 0.0))

        c_f1 = calculate_f1(c_p, c_r)

        if base_df is not None:

                                                                                             

            summary_df = build_or_update_summary_df(None, base_df, cbam_df)

            save_summary_tables(summary_df, out_folder, item["label"])

                               

            plot_metrics_comparison_bar(

                summary_df=summary_df,

                output_path=out_folder / "metrics_bar_comparison.png",

                base_name="Baseline (YOLO11n)",

                cbam_name=item["label"],

                chart_title=item.get("chart_title")

            )

                                           

            plot_learning_curves(

                base_df=base_df,

                cbam_df=cbam_df,

                output_path=out_folder / "learning_curves_comparison.png",

                base_label="Baseline (YOLO11n)",

                cbam_label=item["label"]

            )

        else:

                                                                          

            single_summary_df = pd.DataFrame([

                {"Metric": "mAP@0.5", "Giá trị": round(c_map50, 4)},

                {"Metric": "mAP@0.5:0.95", "Giá trị": round(c_map50_95, 4)},

                {"Metric": "Precision", "Giá trị": round(c_p, 4)},

                {"Metric": "Recall", "Giá trị": round(c_r, 4)},

                {"Metric": "F1-Score", "Giá trị": round(c_f1, 4)},

            ])

            single_summary_df.to_csv(out_folder / "summary_standalone.csv", index=False)

        overall_list.append({

            "Model": item["label"],

            "Position": item["pos"],

            "mAP@0.5": c_map50,

            "mAP@0.5:0.95": c_map50_95,

            "Precision": c_p,

            "Recall": c_r,

            "F1": c_f1,

            "Best_Epoch": cbam_best_ep

        })

                                                     
    if overall_list:
        plot_overall_ablation_summary(eval_dir, overall_list)

                                                                                    
    plot_overall_learning_curves_map50(base_dir, eval_dir, comparisons, baseline_csv=baseline_csv)

def plot_overall_learning_curves_map50(
    base_dir: Path,
    eval_dir: Path,
    comparisons: List[Dict[str, str]],
    baseline_csv: Optional[Path] = None
) -> None:
    """Vẽ đường cong mAP@0.5 của tất cả các mô hình trên cùng 1 biểu đồ duy nhất."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

                       
    b_df = None
    b_max_map50 = 0.0

    if baseline_csv and baseline_csv.exists():

        b_df = load_ultralytics_results(baseline_csv)

        b_max_map50 = float(b_df["metrics/mAP50(B)"].max()) if "metrics/mAP50(B)" in b_df.columns else 0.0

                                                                              

    model_records = []

    best_overall_map50 = b_max_map50

    champ_label = "Baseline (YOLO11n)" if b_df is not None else ""

    for idx, item in enumerate(comparisons):

        candidates = [item["model_dir"], item["model_dir"].replace("cbam_", "")]

        csv_p = None

        for cand in candidates:

            cand_p = base_dir / cand / "results.csv"

            if cand_p.exists():

                csv_p = cand_p

                break

        if csv_p:

            c_df = load_ultralytics_results(csv_p)

            c_max = float(c_df["metrics/mAP50(B)"].max()) if "metrics/mAP50(B)" in c_df.columns else 0.0

            model_records.append((idx, item, c_df, c_max))

            if c_max > best_overall_map50 or not champ_label:

                best_overall_map50 = c_max

                champ_label = item["label"]

                                          

    if b_df is not None:

        is_base_champ = (champ_label == "Baseline (YOLO11n)")

        ax.plot(

            b_df["epoch"], b_df["metrics/mAP50(B)"],

            label="Baseline (YOLO11n)" + (" (Optimal)" if is_base_champ else ""),

            color=PALETTE_ABLATION[0],

            lw=2.5 if is_base_champ else 1.8,

            linestyle="-" if is_base_champ else "--",

            zorder=5 if is_base_champ else 3

        )

                          

    for idx, item, c_df, c_max in model_records:

        color = PALETTE_ABLATION[(idx + 1) % len(PALETTE_ABLATION)]

        is_champ = (item["label"] == champ_label)

        ls = "-" if is_champ else "--"

        lw = 2.5 if is_champ else 1.6

        alpha = 1.0 if is_champ else 0.8

        zorder = 5 if is_champ else 3

        ax.plot(

            c_df["epoch"], c_df["metrics/mAP50(B)"],

            label=item["label"] + (" (Optimal)" if is_champ else ""),

            color=color, lw=lw, linestyle=ls, alpha=alpha, zorder=zorder

        )

    ax.set_title(r"Ablation Study Convergence Dynamics: Validation $\text{mAP}_{50}$ over 100 Epochs",

                 fontsize=13, fontweight="bold", pad=12)

    ax.set_xlabel("Training Epochs", fontsize=11.5, fontweight="bold")

    ax.set_ylabel(r"Validation $\text{mAP}_{50}$", fontsize=11.5, fontweight="bold")

    ax.set_ylim(-0.02, 0.72)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    despine(ax)

    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=10)

    plt.tight_layout()

    out_p = eval_dir / "overall_learning_curves_map50.png"

    plt.savefig(out_p)

    plt.close()

    print(f"-> Đã lưu biểu đồ Overall Learning Curves: {out_p}")

def main() -> None:

    parser = argparse.ArgumentParser(description="Vẽ lại toàn bộ biểu đồ so sánh chuẩn khoa học.")

    parser.add_argument("--base_dir", type=str, default="",

                        help="Thư mục chứa các kết quả thí nghiệm (gồm baseline, cbam_*)")

    parser.add_argument("--eval_dir", type=str, default="",

                        help="Thư mục xuất biểu đồ đánh giá tổng hợp")

    parser.add_argument("--baseline_csv", type=str, default="",

                        help="Đường dẫn trực tiếp tới file results.csv của baseline (tùy chọn)")

                                                            

    parser.add_argument("--baseline_dir", type=str, default="",

                        help="Thư mục kết quả baseline (so sánh đơn lẻ)")

    parser.add_argument("--cbam_dir", type=str, default="",

                        help="Thư mục kết quả cbam (so sánh đơn lẻ)")

    parser.add_argument("--output_dir", type=str, default="",

                        help="Thư mục xuất kết quả (so sánh đơn lẻ)")

    args = parser.parse_args()

                                                                    

    if args.baseline_dir and args.cbam_dir:

        out_p = Path(args.output_dir or (Path(args.cbam_dir) / "evaluation"))

        out_p.mkdir(parents=True, exist_ok=True)

        base_df = load_ultralytics_results(Path(args.baseline_dir) / "results.csv")

        cbam_df = load_ultralytics_results(Path(args.cbam_dir) / "results.csv")

        variant_name = Path(args.cbam_dir).name

        label = f"YOLO11n + CBAM ({variant_name})"

        summary_df = build_or_update_summary_df(None, base_df, cbam_df)

        save_summary_tables(summary_df, out_p, label)

        plot_metrics_comparison_bar(summary_df, out_p / "metrics_bar_comparison.png",

                                     "Baseline (YOLO11n)", label)

        plot_learning_curves(base_df, cbam_df, out_p / "learning_curves_comparison.png",

                             "Baseline (YOLO11n)", label)

        print(f"[HOÀN THÀNH] Đã vẽ so sánh đơn lẻ giữa Baseline và {variant_name} tại: {out_p}")

        return

                                           

    base_dir = Path(args.base_dir or "experiments/testing/03-backbone_neck_head_neckNhead-100")

    eval_dir = Path(args.eval_dir or str(base_dir / "full_evaluation"))

    baseline_override = Path(args.baseline_csv) if args.baseline_csv else None

    process_all_comparisons(base_dir, eval_dir, baseline_csv_override=baseline_override)

    print("\n[HOÀN THÀNH] Toàn bộ hệ thống biểu đồ đã được vẽ lại theo chuẩn khoa học cao cấp!")

if __name__ == "__main__":
    main()
