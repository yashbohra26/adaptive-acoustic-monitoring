"""Visualization and reporting script for Candidate v1.2 independent holdout evaluation.

Generates separate, high-resolution figures and markdown/CSV tables specifically
documenting Candidate v1.2 performance, work, and uncertainty on the independent holdout:
1. fig1_v1_2_independent_performance.png / .pdf
2. fig2_v1_2_work_and_escalation.png / .pdf
3. fig3_v1_2_case_and_snr_breakdown.png / .pdf
4. results/tables/v1_2_overall_performance_table.md / .csv
5. results/tables/v1_2_case_breakdown_table.md / .csv
6. results/tables/v1_2_snr_breakdown_table.md / .csv
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


def plot_v1_2_performance_with_cis(prov_data: dict, out_dir: Path):
    """Plot F1, Precision, Recall, and False-Alarm Rate with 95% bootstrap CIs."""
    methods = ["fixed_low", "fixed_high", "candidate_v1.1", "candidate_v1.2"]
    labels = ["Fixed-Low", "Fixed-High", "Candidate v1.1\n(frozen)", "Candidate v1.2\n(improved)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    metrics = ["f1", "precision", "recall", "false_alarm_rate"]
    metric_titles = ["F1 Score", "Precision", "Recall", "False-Alarm Rate"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    ci_data = prov_data["bootstrap_uncertainty_95ci"]

    for ax_idx, (m_key, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[ax_idx]
        means = [ci_data[m][m_key]["mean"] for m in methods]
        yerr_low = [means[i] - ci_data[methods[i]][m_key]["ci_95"][0] for i in range(len(methods))]
        yerr_high = [ci_data[methods[i]][m_key]["ci_95"][1] - means[i] for i in range(len(methods))]
        yerr = [yerr_low, yerr_high]

        bars = ax.bar(labels, means, yerr=yerr, capsize=4, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 1.2, "ecolor": "black"})

        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.12 if m_key != "false_alarm_rate" else 0.25)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 10 if m_key == "false_alarm_rate" else 8),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_v1_2_independent_performance.png", dpi=300)
    fig.savefig(out_dir / "fig1_v1_2_independent_performance.pdf")
    plt.close(fig)
    print("Generated Figure 1: Candidate v1.2 Independent Performance with Bootstrap CIs.")


def plot_v1_2_work_and_escalation(prov_data: dict, df: pd.DataFrame, out_dir: Path):
    """Dual-panel: Algorithmic Work (Ops proxy with 95% CI) and Escalation Rates."""
    methods = ["fixed_low", "fixed_high", "candidate_v1.1", "candidate_v1.2"]
    labels = ["Fixed-Low", "Fixed-High", "Candidate v1.1\n(frozen)", "Candidate v1.2\n(improved)"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # Panel 1: Algorithmic Work Proxy with 95% bootstrap CI
    ci_data = prov_data["bootstrap_uncertainty_95ci"]
    ops_means = [ci_data[m]["avg_ops"]["mean"] for m in methods]
    yerr_low = [ops_means[i] - ci_data[methods[i]][avg_ops_key]["ci_95"][0] for i, avg_ops_key in enumerate(["avg_ops"] * len(methods))]
    yerr_high = [ci_data[methods[i]]["avg_ops"]["ci_95"][1] - ops_means[i] for i in range(len(methods))]
    yerr = [yerr_low, yerr_high]

    bars1 = ax1.bar(labels, ops_means, yerr=yerr, capsize=4, color=colors, alpha=0.85,
                    edgecolor="black", linewidth=0.8, error_kw={"elinewidth": 1.2, "ecolor": "black"})
    ax1.axhline(29250, color="#d95f02", linestyle=":", linewidth=1.5, label="Fixed-High Reference (29,250)")
    ax1.set_ylabel("Operation-Count Proxy (ops/window)")
    ax1.set_title("Algorithmic Work Proxy (with 95% CI)", fontweight="bold")
    ax1.set_ylim(0, 36000)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.legend(loc="upper left")

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:,.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5)

    # Panel 2: Escalation rate by signal case comparing Candidate v1.1 vs Candidate v1.2
    cases = ["normal", "frequency_shift", "added_harmonic", "impact"]
    case_names = ["Normal\n(Target: Low)", "Freq Shift\n(Target: High)", "Added Harm\n(Target: High)", "Impact\n(Target: High)"]

    esc_v1_1 = [df[df["case"] == c]["candidate_v1.1_fft"].mean() * 100 for c in cases]
    esc_v1_2 = [df[df["case"] == c]["candidate_v1.2_fft"].mean() * 100 for c in cases]

    x = np.arange(len(cases))
    width = 0.35

    rects1 = ax2.bar(x - width/2, esc_v1_1, width, label="Candidate v1.1", color="#7570b3", alpha=0.85, edgecolor="black", linewidth=0.8)
    rects2 = ax2.bar(x + width/2, esc_v1_2, width, label="Candidate v1.2", color="#1b9e77", alpha=0.85, edgecolor="black", linewidth=0.8)

    ax2.set_ylabel("Escalation Rate to Stage 2 FFT (%)")
    ax2.set_title("Escalation by Signal Type", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(case_names, fontsize=8.5)
    ax2.set_ylim(0, 115)
    ax2.legend(loc="upper left")

    for rect in rects1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}%",
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_v1_2_work_and_escalation.png", dpi=300)
    fig.savefig(out_dir / "fig2_v1_2_work_and_escalation.pdf")
    plt.close(fig)
    print("Generated Figure 2: Work and Escalation Comparison.")


def plot_v1_2_case_and_snr_breakdown(df: pd.DataFrame, out_dir: Path):
    """Plot per-case Recall and SNR breakdown comparing Fixed-High, v1.1, and v1.2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    anomaly_cases = ["frequency_shift", "added_harmonic", "impact"]
    case_labels = ["Frequency Shift\n(N=80)", "Added Harmonic\n(N=80)", "Impact\n(N=80)"]

    methods = [("fixed_high", "Fixed-High", "#d95f02"),
               ("candidate_v1.1", "Candidate v1.1", "#7570b3"),
               ("candidate_v1.2", "Candidate v1.2", "#1b9e77")]

    # Panel 1: Recall by Anomaly Case
    x = np.arange(len(anomaly_cases))
    width = 0.25

    for idx, (m_id, m_label, color) in enumerate(methods):
        recalls = []
        for c in anomaly_cases:
            sub = df[df["case"] == c]
            rec = sub[f"{m_id}_pred"].sum() / len(sub)
            recalls.append(rec)

        pos = x + (idx - 1) * width
        bars = ax1.bar(pos, recalls, width, label=m_label, color=color, alpha=0.85, edgecolor="black", linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax1.annotate(f"{h:.3f}",
                         xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    ax1.set_ylabel("Recall")
    ax1.set_title("Recall by Anomaly Type (Independent Holdout)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_labels)
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="lower right")

    # Panel 2: Added Harmonic Recall across SNR levels
    snr_levels = [30.0, 20.0, 10.0, 0.0]
    snr_labels = ["30 dB (Clean)", "20 dB", "10 dB", "0 dB (Noisy)"]
    x_snr = np.arange(len(snr_levels))

    sub_harm = df[df["case"] == "added_harmonic"]
    for idx, (m_id, m_label, color) in enumerate(methods):
        harm_recalls = []
        for snr in snr_levels:
            sub_snr = sub_harm[sub_harm["snr_db"] == snr]
            rec = sub_snr[f"{m_id}_pred"].sum() / len(sub_snr)
            harm_recalls.append(rec)

        pos = x_snr + (idx - 1) * width
        bars = ax2.bar(pos, harm_recalls, width, label=m_label, color=color, alpha=0.85, edgecolor="black", linewidth=0.8)
        for b in bars:
            h = b.get_height()
            ax2.annotate(f"{h:.2f}",
                         xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Recall on Added Harmonic")
    ax2.set_title("Added Harmonic Sensitivity vs. Noise Floor", fontweight="bold")
    ax2.set_xticks(x_snr)
    ax2.set_xticklabels(snr_labels)
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_v1_2_case_and_snr_breakdown.png", dpi=300)
    fig.savefig(out_dir / "fig3_v1_2_case_and_snr_breakdown.pdf")
    plt.close(fig)
    print("Generated Figure 3: Case and SNR Breakdown Comparison.")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a GitHub-flavored Markdown table without tabulate."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def generate_tables(prov_data: dict, df: pd.DataFrame, tables_dir: Path):
    """Generate Markdown and CSV tables for Candidate v1.2 evaluation."""
    ci_data = prov_data["bootstrap_uncertainty_95ci"]
    methods = ["fixed_low", "fixed_high", "candidate_v1.1", "candidate_v1.2"]
    labels = ["Fixed-Low", "Fixed-High", "Candidate v1.1 (frozen)", "Candidate v1.2 (improved)"]

    # Table 1: Overall Performance with 95% CIs
    t1_rows = []
    for m, lbl in zip(methods, labels):
        m_ci = ci_data[m]
        # Escalation rate
        if f"{m}_fft" in df.columns:
            esc_rate = df[f"{m}_fft"].mean() * 100
        else:
            esc_rate = 0.0 if m == "fixed_low" else 100.0
        t1_rows.append({
            "Method": lbl,
            "F1 Score [95% CI]": f"{m_ci['f1']['mean']:.4f} [{m_ci['f1']['ci_95'][0]:.4f}, {m_ci['f1']['ci_95'][1]:.4f}]",
            "Precision [95% CI]": f"{m_ci['precision']['mean']:.4f} [{m_ci['precision']['ci_95'][0]:.4f}, {m_ci['precision']['ci_95'][1]:.4f}]",
            "Recall [95% CI]": f"{m_ci['recall']['mean']:.4f} [{m_ci['recall']['ci_95'][0]:.4f}, {m_ci['recall']['ci_95'][1]:.4f}]",
            "FAR [95% CI]": f"{m_ci['false_alarm_rate']['mean']:.4f} [{m_ci['false_alarm_rate']['ci_95'][0]:.4f}, {m_ci['false_alarm_rate']['ci_95'][1]:.4f}]",
            "Avg Ops Proxy": f"{m_ci['avg_ops']['mean']:,.1f}",
            "Ops Savings (%)": f"{m_ci['savings']['mean']:+.2f}% [{m_ci['savings']['ci_95'][0]:+.2f}%, {m_ci['savings']['ci_95'][1]:+.2f}%]",
            "Stage 2 Escalation (%)": f"{esc_rate:.1f}%",
        })

    t1_df = pd.DataFrame(t1_rows)
    t1_df.to_csv(tables_dir / "v1_2_overall_performance_table.csv", index=False)
    with open(tables_dir / "v1_2_overall_performance_table.md", "w") as f:
        f.write("# Candidate v1.2 Independent Holdout Overall Performance\n\n")
        f.write("Evaluated on 320 independent held-out windows (20 groups, seed 20261014). Confidence intervals are 95% group-level bootstrap (1,000 iterations).\n\n")
        f.write(format_markdown_table(t1_df))
        f.write("\n")

    # Table 2: Case Breakdown
    cases = ["normal", "frequency_shift", "added_harmonic", "impact"]
    t2_rows = []
    for c in cases:
        sub = df[df["case"] == c]
        n_win = len(sub)
        is_anom = sub["anomaly_true"].iloc[0] == 1
        metric_name = "Recall" if is_anom else "Specificity (1-FAR)"

        row = {"Signal Case": c, "Anomaly": "Yes" if is_anom else "No", "Windows": n_win}
        for m, lbl in zip(methods, labels):
            if is_anom:
                val = sub[f"{m}_pred"].sum() / n_win
            else:
                val = (sub[f"{m}_pred"] == 0).sum() / n_win
            esc = sub[f"{m}_fft"].mean() * 100 if f"{m}_fft" in sub.columns else (0.0 if m == "fixed_low" else 100.0)
            row[f"{lbl} ({metric_name})"] = f"{val:.4f}"
            row[f"{lbl} (Esc %)"] = f"{esc:.1f}%"
        t2_rows.append(row)

    t2_df = pd.DataFrame(t2_rows)
    t2_df.to_csv(tables_dir / "v1_2_case_breakdown_table.csv", index=False)
    with open(tables_dir / "v1_2_case_breakdown_table.md", "w") as f:
        f.write("# Candidate v1.2 Independent Holdout Per-Case Breakdown\n\n")
        f.write(format_markdown_table(t2_df))
        f.write("\n")

    # Table 3: SNR Breakdown
    snrs = [30.0, 20.0, 10.0, 0.0]
    t3_rows = []
    for s in snrs:
        sub_s = df[df["snr_db"] == s]
        sub_anom = sub_s[sub_s["anomaly_true"] == 1]
        sub_norm = sub_s[sub_s["anomaly_true"] == 0]

        row = {"SNR (dB)": f"{s:.0f} dB", "Anomaly N": len(sub_anom), "Normal N": len(sub_norm)}
        for m, lbl in zip(["fixed_high", "candidate_v1.1", "candidate_v1.2"],
                          ["Fixed-High", "Candidate v1.1", "Candidate v1.2"]):
            rec = sub_anom[f"{m}_pred"].sum() / len(sub_anom)
            far = sub_norm[f"{m}_pred"].sum() / len(sub_norm)
            ops = sub_s[f"{m}_ops"].mean()
            row[f"{lbl} Recall"] = f"{rec:.4f}"
            row[f"{lbl} FAR"] = f"{far:.4f}"
            row[f"{lbl} Avg Ops"] = f"{ops:,.0f}"
        t3_rows.append(row)

    t3_df = pd.DataFrame(t3_rows)
    t3_df.to_csv(tables_dir / "v1_2_snr_breakdown_table.csv", index=False)
    with open(tables_dir / "v1_2_snr_breakdown_table.md", "w") as f:
        f.write("# Candidate v1.2 Independent Holdout SNR Breakdown\n\n")
        f.write(format_markdown_table(t3_df))
        f.write("\n")

    print("Generated all v1.2 Markdown and CSV tables in results/tables/.")


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    prov_path = results_dir / "candidate_v1.2_provenance_and_uncertainty.json"
    rep_csv_path = results_dir / "candidate_v1.2_reproduced_windows.csv"

    if not prov_path.exists() or not rep_csv_path.exists():
        raise FileNotFoundError("Missing provenance json or reproduced windows csv. Run verify_v1_2_provenance.py first.")

    with open(prov_path, "r") as f:
        prov_data = json.load(f)

    df = pd.read_csv(rep_csv_path)

    set_plot_style()
    plot_v1_2_performance_with_cis(prov_data, figures_dir)
    plot_v1_2_work_and_escalation(prov_data, df, figures_dir)
    plot_v1_2_case_and_snr_breakdown(df, figures_dir)
    generate_tables(prov_data, df, tables_dir)


if __name__ == "__main__":
    main()
