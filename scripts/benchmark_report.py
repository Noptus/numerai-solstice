"""Add a public-benchmark comparison to a completed frozen experiment.

This is descriptive analysis of an already consumed holdout. It never changes
model selection. Reusing the historical labels does not create new evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.data import atomic_json, file_hash
from research.runner import Runner
from research.scoring import summary


def compare(config, data, run):
    runner = Runner(config, data, run)
    if not (runner.output / "result.json").exists():
        raise ValueError("complete the frozen experiment before descriptive comparison")
    runner._selection()
    frame, _ = runner.frame("holdout")
    left = runner.experiment.holdout_start
    evaluation = frame.loc[frame.date >= left].reset_index(drop=True)
    rows = []
    with threadpool_limits(limits=runner.experiment.threads):
        for date, _, scorer in runner.contexts(evaluation):
            # prepare() has already Gaussian-ranked and neutralized this benchmark.
            ncorr = float(
                (scorer.target @ scorer.benchmark)
                / (np.linalg.norm(scorer.target) * np.linalg.norm(scorer.benchmark))
            )
            rows.append(
                {
                    "date": str(date.date()),
                    "model": runner.experiment.benchmark,
                    "ncorr": ncorr,
                    "benchmark_contribution": 0.0,
                }
            )
    scores = pd.read_csv(runner.output / "holdout_scores.csv")
    scores = pd.concat([scores, pd.DataFrame(rows)], ignore_index=True)
    paired = scores.pivot(index="date", columns="model", values="ncorr")
    if paired.isna().any().any():
        raise ValueError("benchmark and model date coverage differ")
    result = {
        "fingerprint": runner.fingerprint,
        "selection_sha256": runner._selection()["selection_sha256"],
        "result_sha256": file_hash(runner.output / "result.json"),
        "analysis": "descriptive comparison on the same consumed historical holdout; no reselection",
        "models": {
            name: summary(g.ncorr, block_size=runner.experiment.bootstrap_block_dates)
            for name, g in scores.groupby("model", sort=True)
        },
        "paired_gain_vs_public_benchmark": summary(
            paired.selected_ensemble - paired[runner.experiment.benchmark],
            block_size=runner.experiment.bootstrap_block_dates,
        ),
    }
    scores.to_csv(runner.output / "holdout_comparison_scores.csv", index=False)
    atomic_json(runner.output / "benchmark_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/wide.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/v6.0"))
    parser.add_argument("--run", type=Path, default=Path("runs/mac-wide-v6"))
    args = parser.parse_args()
    compare(args.config, args.data, args.run)


if __name__ == "__main__":
    main()
