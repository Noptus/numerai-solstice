"""Render historical NCORR paths from a packaged report (no training or selection)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def plot(root, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np

    root, output = Path(root), Path(output)
    result = json.loads((root / "result.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    baseline = result["baseline"]
    benchmark = plan["experiment"]["benchmark"]
    paths = {}
    with (root / "holdout_comparison_scores.csv").open() as handle:
        for row in csv.DictReader(handle):
            paths.setdefault(row["model"], []).append(
                (datetime.fromisoformat(row["date"]), float(row["ncorr"]))
            )
    names = ["selected_ensemble", baseline, benchmark]
    colors = ["#047857", "#6b7280", "#2563eb"]
    labels = ["Frozen selection", "Predeclared control", "Public benchmark"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, layout="constrained")
    heading = (
        "Numerai v6 · exploratory historical follow-up"
        if plan["experiment"].get("historical_evaluation_status") == "reused_exploratory"
        else "Numerai v6 · reserved historical evaluation"
    )
    fig.suptitle(heading, fontsize=17, fontweight="bold", x=0.07, ha="left")
    for name, color, label in zip(names, colors, labels, strict=True):
        ordered = sorted(paths[name])
        dates, scores = zip(*ordered, strict=True)
        scores = np.array(scores)
        window = 13
        rolling = np.convolve(scores, np.ones(window) / window, mode="valid")
        axes[0].plot(dates[window - 1 :], rolling, color=color, label=label, linewidth=1.8)
        axes[1].plot(dates, scores.cumsum(), color=color, label=label, linewidth=1.8)
    axes[0].set_title("13-date rolling mean NCORR", loc="left", fontsize=11)
    axes[1].set_title("Cumulative date NCORR · not an investment return", loc="left", fontsize=11)
    for ax in axes:
        ax.axhline(0, color="#9ca3af", linewidth=0.7, linestyle="--")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_ylabel("Score")
    axes[0].legend(loc="best", frameon=False, ncols=3)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].set_xlabel(
        f"Snapshot round {plan['round']} · fixed selection; historical public data; no live performance claim"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.output)


if __name__ == "__main__":
    main()
