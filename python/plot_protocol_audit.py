"""Plot saved operation-proxy versus recall results without rerunning evaluation.

The two panels use separate held-out cohorts. Error bars are saved 95% group-
bootstrap percentile intervals. This script reads evaluation artifacts only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "fixed_low": "#2b5c8f",
    "fixed_high": "#d95f02",
    "adaptive_v1.0": "#7570b3",
    "candidate_v1.1": "#7b61a8",
    "candidate_v1.2": "#1b9e77",
}
LABELS = {
    "fixed_low": "Fixed-Low",
    "fixed_high": "Fixed-High",
    "adaptive_v1.0": "Historical Adaptive v1.0",
    "candidate_v1.1": "Candidate v1.1",
    "candidate_v1.2": "Candidate v1.2",
}


def interval_pair(container: dict, method: str, metric: str, nested: bool) -> list[float]:
    values = container[method][metric]
    return values["ci_95"] if nested else values


def add_cohort_panel(ax, title: str, evaluation: dict, bootstrap: dict,
                     methods: list[str], nested_ci: bool) -> None:
    high = evaluation["methods"]["fixed_high"]
    relative_recall_floor = high["metrics"]["recall"] - 0.05
    frozen_dev_recall_floor = 0.8791666666666667

    ax.axhline(
        relative_recall_floor, color="#444444", linestyle="--", linewidth=1.3,
        label="Fixed-High recall − 0.05 (same holdout)",
    )
    ax.axhline(
        frozen_dev_recall_floor, color="#b2182b", linestyle=":", linewidth=1.5,
        label="Phase 4 frozen development cutoff (0.8792)",
    )
    ax.axvline(
        29250, color="#d95f02", linestyle="-.", linewidth=1.1,
        label="Fixed-High cost reference",
    )

    for method in methods:
        result = evaluation["methods"][method]
        point_x = float(result["avg_op_count_proxy"])
        point_y = float(result["metrics"]["recall"])
        x_ci = interval_pair(bootstrap, method, "avg_ops", nested_ci)
        y_ci = interval_pair(bootstrap, method, "recall", nested_ci)
        xerr = [[max(0.0, point_x - x_ci[0])], [max(0.0, x_ci[1] - point_x)]]
        yerr = [[max(0.0, point_y - y_ci[0])], [max(0.0, y_ci[1] - point_y)]]
        ax.errorbar(
            [point_x], [point_y], xerr=xerr, yerr=yerr,
            fmt="o", markersize=7, capsize=3, elinewidth=1.0,
            color=COLORS[method], markeredgecolor="white", markeredgewidth=0.7,
            zorder=3,
        )

        if method == "fixed_high":
            offset = (8, 10)
        elif method == "candidate_v1.2":
            offset = (-118, -34)
        elif method == "candidate_v1.1" and nested_ci:
            offset = (8, -31)
        elif method == "candidate_v1.1":
            offset = (8, 9)
        elif method == "adaptive_v1.0":
            offset = (8, 8)
        else:
            offset = (8, -20)

        f1 = result["metrics"]["f1"]
        ax.annotate(
            f"{LABELS[method]}\nF1={f1:.3f}",
            (point_x, point_y), xytext=offset, textcoords="offset points",
            fontsize=8, color=COLORS[method],
        )

    ax.set_title(title)
    ax.set_xlabel("Operation-count proxy (ops/window)")
    ax.set_xlim(8000, 34000)
    ax.set_ylim(0.30, 1.03)
    ax.grid(True, alpha=0.25)


def make_plot(results_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    phase5 = json.loads((results_dir / "phase5_final_evaluation.json").read_text(encoding="utf-8"))
    independent = json.loads((results_dir / "candidate_v1.2_independent_evaluation.json").read_text(encoding="utf-8"))
    provenance = json.loads((results_dir / "candidate_v1.2_provenance_and_uncertainty.json").read_text(encoding="utf-8"))
    phase4 = json.loads((results_dir / "phase4_study" / "selector_study_results.json").read_text(encoding="utf-8"))

    assert phase5["test_windows"] == 320 and phase5["test_groups"] == 20
    assert independent["eval_windows"] == 320
    assert independent["protocol_version"] == "1.2.0"
    assert abs(
        phase4["quality_constraints"]["min_recall"] - 0.8791666666666667
    ) < 1e-12
    assert provenance["signal_generation"]["total_groups"] == 20

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "audit_cost_vs_recall.png"
    pdf_path = output_dir / "audit_cost_vs_recall.pdf"
    if png_path.exists() or pdf_path.exists():
        raise FileExistsError("Audit plot output already exists; refusing to overwrite it")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 180})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    add_cohort_panel(
        axes[0], "Phase 5 holdout · Protocol v1.1.0 · 20 groups", phase5,
        phase5["bootstrap_95ci"],
        ["fixed_low", "fixed_high", "adaptive_v1.0", "candidate_v1.1"], False,
    )
    add_cohort_panel(
        axes[1], "Independent holdout · Protocol v1.2.0 · seed 20261014", independent,
        provenance["bootstrap_uncertainty_95ci"],
        ["fixed_low", "fixed_high", "candidate_v1.1", "candidate_v1.2"], True,
    )
    axes[0].set_ylabel("Anomaly recall")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Detection recall versus estimated algorithmic work", y=1.09, fontsize=13)
    fig.text(
        0.5, -0.015,
        "Error bars: saved 95% group-bootstrap percentile intervals. Panels are separate held-out cohorts; do not pool them. "
        "The two recall guides expose the documented cutoff difference. Operation counts are proxies, not measured energy or hardware cost.",
        ha="center", va="top", fontsize=8, wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--out", type=Path, default=root / "results" / "figures" / "protocol_audit")
    args = parser.parse_args()
    for path in make_plot(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
