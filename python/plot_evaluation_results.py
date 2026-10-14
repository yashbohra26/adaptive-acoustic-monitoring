"""Report-ready visualization and analysis script for acoustic anomaly detection.

Generates high-resolution publication-grade plots and markdown/CSV summary tables:
1. Performance Comparison across policies (F1, Precision, Recall, False-Alarm Rate).
2. Computational Work vs. Desktop Runtime (Ops Proxy vs. Mean Python Runtime [ms]).
3. Recall and False-Alarm Rate by Anomaly Case and SNR.
4. Adaptive Escalation Rate by Case and SNR.
5. Per-Case Error Breakdown (False Negatives and False Positives).

All figures are saved to results/figures/ (PNG at 300 DPI and PDF).
All tables are saved to results/tables/ (CSV and GitHub-flavored Markdown).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_overall_performance(benchmark_data: dict, out_dir: Path):
    """Plot F1, Precision, Recall, and False-Alarm Rate across methods."""
    methods = ["fixed_low", "fixed_high", "candidate_v1.1", "historical_v1.0"]
    labels = ["Fixed-Low", "Fixed-High", "Candidate v1.1", "Historical v1.0"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    metrics = ["f1", "precision", "recall", "false_alarm_rate"]
    metric_titles = ["F1 Score", "Precision", "Recall", "False-Alarm Rate"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=False)

    for ax_idx, (m_key, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[ax_idx]
        vals = [benchmark_data["methods"][m]["metrics"][m_key] for m in methods]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)

        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.08 if m_key != "false_alarm_rate" else 0.25)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_overall_performance.png", dpi=300)
    fig.savefig(out_dir / "fig1_overall_performance.pdf")
    plt.close(fig)
    print("Generated Figure 1: Overall Performance Comparison.")


def plot_work_and_runtime(benchmark_data: dict, out_dir: Path):
    """Dual-panel plot: Operation-Count Proxy and Desktop Python Runtime."""
    methods = ["fixed_low", "fixed_high", "candidate_v1.1", "historical_v1.0"]
    labels = ["Fixed-Low", "Fixed-High", "Candidate v1.1", "Historical v1.0"]
    colors = ["#2b5c8f", "#d95f02", "#1b9e77", "#7570b3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: Ops proxy
    ops = [benchmark_data["methods"][m]["avg_op_count_proxy"] for m in methods]
    bars1 = ax1.bar(labels, ops, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax1.axhline(29250, color="#d95f02", linestyle=":", linewidth=1.5, label="Fixed-High Baseline (29,250)")
    ax1.set_ylabel("Operation-Count Proxy (ops/window)")
    ax1.set_title("Algorithmic Work Proxy", fontweight="bold")
    ax1.set_ylim(0, 42000)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.legend(loc="upper left")

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:,.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5)

    # Panel 2: Desktop Runtime in ms
    rts = [benchmark_data["methods"][m]["runtime_ms"]["mean"] for m in methods]
    errs = [benchmark_data["methods"][m]["runtime_ms"]["std"] for m in methods]
    bars2 = ax2.bar(labels, rts, yerr=errs, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("Desktop Mean Runtime (ms/window)")
    ax2.set_title("Benchmark Execution Time (Python 3.13, Host CPU)", fontweight="bold")
    ax2.set_ylim(0, 0.9)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=20, ha="right")

    for bar, val in zip(bars2, rts):
        ax2.annotate(f"{val:.3f} ms",
                     xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_work_and_runtime.png", dpi=300)
    fig.savefig(out_dir / "fig2_work_and_runtime.pdf")
    plt.close(fig)
    print("Generated Figure 2: Work & Runtime Comparison.")


def plot_case_and_snr_breakdown(df_windows: pd.DataFrame, out_dir: Path):
    """Plot Recall by Anomaly Case and SNR, plus False-Alarm Rate by SNR."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    cases = ["frequency_shift", "added_harmonic", "impact"]
    case_labels = ["Frequency Shift", "Added Harmonic", "Impact"]

    # Recall by case for Fixed-High vs Candidate v1.1
    x = np.arange(len(cases))
    w = 0.25

    rec_low = []
    rec_high = []
    rec_v11 = []

    for c in cases:
        sub = df_windows[df_windows["case"] == c]
        rec_low.append(np.mean(sub["fixed_low_pred"] == 1))
        rec_high.append(np.mean(sub["fixed_high_pred"] == 1))
        rec_v11.append(np.mean(sub["candidate_v1.1_pred"] == 1))

    ax1.bar(x - w, rec_low, width=w, label="Fixed-Low", color="#2b5c8f", alpha=0.85, edgecolor="black")
    ax1.bar(x, rec_high, width=w, label="Fixed-High", color="#d95f02", alpha=0.85, edgecolor="black")
    ax1.bar(x + w, rec_v11, width=w, label="Candidate v1.1", color="#1b9e77", alpha=0.85, edgecolor="black")

    ax1.set_xticks(x)
    ax1.set_xticklabels(case_labels)
    ax1.set_ylabel("Anomaly Recall")
    ax1.set_title("Recall by Anomaly Mechanism", fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="lower left")

    # Recall across SNR levels
    snrs = [30.0, 20.0, 10.0, 0.0]
    snr_labels = ["30 dB", "20 dB", "10 dB", "0 dB"]
    xs = np.arange(len(snrs))

    snr_low = []
    snr_high = []
    snr_v11 = []

    anom_df = df_windows[df_windows["anomaly_true"] == 1]
    for snr in snrs:
        sub = anom_df[anom_df["snr_db"] == snr]
        snr_low.append(np.mean(sub["fixed_low_pred"] == 1))
        snr_high.append(np.mean(sub["fixed_high_pred"] == 1))
        snr_v11.append(np.mean(sub["candidate_v1.1_pred"] == 1))

    ax2.plot(snr_labels, snr_low, "o-", color="#2b5c8f", linewidth=1.8, label="Fixed-Low")
    ax2.plot(snr_labels, snr_high, "s-", color="#d95f02", linewidth=1.8, label="Fixed-High")
    ax2.plot(snr_labels, snr_v11, "^-", color="#1b9e77", linewidth=2.2, label="Candidate v1.1")

    ax2.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax2.set_ylabel("Anomaly Recall")
    ax2.set_title("Detection Robustness across Noise Levels", fontweight="bold")
    ax2.set_ylim(0.2, 1.05)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_case_and_snr_recall.png", dpi=300)
    fig.savefig(out_dir / "fig3_case_and_snr_recall.pdf")
    plt.close(fig)
    print("Generated Figure 3: Recall by Case & SNR.")


def plot_adaptive_escalation(df_windows: pd.DataFrame, out_dir: Path):
    """Plot escalation rates by case and SNR for Candidate v1.1."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # By Case
    cases = ["normal", "frequency_shift", "added_harmonic", "impact"]
    case_labels = ["Normal", "Freq Shift", "Added Harmonic", "Impact"]
    esc_by_case = [np.mean(df_windows[df_windows["case"] == c]["candidate_v1.1_fft"]) * 100.0 for c in cases]
    colors = ["#4575b4", "#fee090", "#fdae61", "#d73027"]

    bars = ax1.bar(case_labels, esc_by_case, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("FFT Escalation Rate (%)")
    ax1.set_title("Stage 2 Escalation by Signal Type", fontweight="bold")
    ax1.set_ylim(0, 115)

    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5)

    # By SNR
    snrs = [30.0, 20.0, 10.0, 0.0]
    snr_labels = ["30 dB", "20 dB", "10 dB", "0 dB"]
    esc_by_snr = [np.mean(df_windows[df_windows["snr_db"] == s]["candidate_v1.1_fft"]) * 100.0 for s in snrs]

    ax2.plot(snr_labels, esc_by_snr, "o-", color="#1b9e77", linewidth=2.0, markersize=7)
    ax2.axhline(64.91, color="#d95f02", linestyle=":", linewidth=1.5, label="Break-Even Ceiling (64.91%)")
    ax2.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax2.set_ylabel("FFT Escalation Rate (%)")
    ax2.set_title("Escalation Rate vs. Noise Level", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")

    for x_pos, y_pos in zip(snr_labels, esc_by_snr):
        ax2.annotate(f"{y_pos:.1f}%",
                     xy=(x_pos, y_pos),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_adaptive_escalation.png", dpi=300)
    fig.savefig(out_dir / "fig4_adaptive_escalation.pdf")
    plt.close(fig)
    print("Generated Figure 4: Adaptive Escalation Breakdown.")


def plot_error_matrix(df_windows: pd.DataFrame, out_dir: Path):
    """Plot False Negatives and False Positives by case for Candidate v1.1."""
    cases = ["normal", "frequency_shift", "added_harmonic", "impact"]
    case_labels = ["Normal", "Freq Shift", "Added Harmonic", "Impact"]

    fn_counts = []
    fp_counts = []

    for c in cases:
        sub = df_windows[df_windows["case"] == c]
        fn = np.sum((sub["anomaly_true"] == 1) & (sub["candidate_v1.1_pred"] == 0))
        fp = np.sum((sub["anomaly_true"] == 0) & (sub["candidate_v1.1_pred"] == 1))
        fn_counts.append(fn)
        fp_counts.append(fp)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(cases))
    w = 0.35

    b1 = ax.bar(x - w/2, fn_counts, width=w, label="False Negatives (Misses)", color="#d73027", alpha=0.85, edgecolor="black")
    b2 = ax.bar(x + w/2, fp_counts, width=w, label="False Positives (False Alarms)", color="#fc8d59", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.set_ylabel("Window Count")
    ax.set_title("Candidate v1.1 Error Breakdown by Mechanism (320 Test Windows)", fontweight="bold")
    ax.set_ylim(0, 22)
    ax.legend(loc="upper right")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_error_breakdown.png", dpi=300)
    fig.savefig(out_dir / "fig5_error_breakdown.pdf")
    plt.close(fig)
    print("Generated Figure 5: Per-Case Error Matrix.")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a GitHub-flavored Markdown table without tabulate."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def generate_summary_tables(benchmark_data: dict, out_dir: Path):
    """Export publication tables in CSV and Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall comparison table
    overall_rows = []
    for m, key in [
        ("Fixed-Low (Standalone)", "fixed_low"),
        ("Fixed-High (Standalone)", "fixed_high"),
        ("Candidate v1.1 (Protocol v1.1.0)", "candidate_v1.1"),
        ("Historical Adaptive v1.0", "historical_v1.0"),
    ]:
        data = benchmark_data["methods"][key]
        m_met = data["metrics"]
        overall_rows.append({
            "Method": m,
            "Accuracy": f"{m_met['accuracy']:.4f}",
            "Precision": f"{m_met['precision']:.4f}",
            "Recall": f"{m_met['recall']:.4f}",
            "F1 Score": f"{m_met['f1']:.4f}",
            "False Alarm Rate": f"{m_met['false_alarm_rate']:.4f}",
            "Freq MAE (Hz)": f"{data['freq_mae_hz']:.2f}" if data['freq_mae_hz'] is not None else "N/A",
            "Avg Ops Proxy": f"{data['avg_op_count_proxy']:,.1f}",
            "Ops Savings (%)": f"{data['savings_vs_fixed_high_percent']:+.2f}%",
            "Mean Runtime (ms)": f"{data['runtime_ms']['mean']:.4f} +- {data['runtime_ms']['std']:.4f}",
            "Escalation Rate (%)": f"{data['escalation_rate']*100.0:.2f}%",
        })

    df_overall = pd.DataFrame(overall_rows)
    df_overall.to_csv(out_dir / "overall_performance_table.csv", index=False)
    with open(out_dir / "overall_performance_table.md", "w", encoding="utf-8") as f:
        f.write(format_markdown_table(df_overall))

    # 2. Case breakdown table for Candidate v1.1
    cand_cases = benchmark_data["methods"]["candidate_v1.1"]["by_case"]
    case_rows = []
    for c, c_name in [
        ("normal", "Clean / Noisy Normal"),
        ("frequency_shift", "Frequency Shift Anomaly"),
        ("added_harmonic", "Added Harmonic Anomaly"),
        ("impact", "High-Crest Impact Anomaly"),
    ]:
        c_data = cand_cases[c]
        c_met = c_data["metrics"]
        case_rows.append({
            "Signal Mechanism": c_name,
            "Windows": c_data["count"],
            "True Positives (TP)": c_met["tp"],
            "False Negatives (FN)": c_met["fn"],
            "True Negatives (TN)": c_met["tn"],
            "False Positives (FP)": c_met["fp"],
            "Recall / FAR": f"{c_met['recall']:.4f}" if c != "normal" else f"FAR={c_met['false_alarm_rate']:.4f}",
            "Avg Ops": f"{c_data['avg_ops']:,.1f}",
            "Escalation Rate": f"{c_data['escalation_rate']*100.0:.1f}%",
            "Mean Runtime (ms)": f"{c_data['avg_runtime_ms']:.4f}",
        })

    df_case = pd.DataFrame(case_rows)
    df_case.to_csv(out_dir / "case_breakdown_table.csv", index=False)
    with open(out_dir / "case_breakdown_table.md", "w", encoding="utf-8") as f:
        f.write(format_markdown_table(df_case))

    print(f"Generated summary tables in {out_dir}")


def main():
    fig_dir = Path("results/figures")
    tab_dir = Path("results/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()

    with open("results/benchmark_fair_evaluation.json", "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    df_windows = pd.read_csv("results/benchmark_fair_windows.csv")

    plot_overall_performance(benchmark_data, fig_dir)
    plot_work_and_runtime(benchmark_data, fig_dir)
    plot_case_and_snr_breakdown(df_windows, fig_dir)
    plot_adaptive_escalation(df_windows, fig_dir)
    plot_error_matrix(df_windows, fig_dir)
    generate_summary_tables(benchmark_data, tab_dir)


if __name__ == "__main__":
    main()
