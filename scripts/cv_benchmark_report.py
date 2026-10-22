"""Compare every predeclared CV candidate with the public benchmark, without selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.data import KEYS, atomic_json, file_hash, validate_keys
from research.runner import Runner, aligned_predictions


def compare(config, data, run, window=None):
    runner = Runner(config, data, run)
    rows = []
    names = runner.names("cv")
    windows = runner.windows("cv")
    if window is not None:
        windows = [w for w in windows if w[0] == window]
        if not windows:
            raise ValueError("unknown CV window")
    with threadpool_limits(limits=2):
        for label, left, right in windows:
            evaluation = pd.read_parquet(
                Path(data) / "validation.parquet",
                columns=[*KEYS, runner.experiment.primary_target],
                filters=[("date", ">=", left), ("date", "<=", right), ("data_type", "==", "validation")],
            ).dropna(subset=[runner.experiment.primary_target])
            validate_keys(evaluation)
            evaluation = evaluation.sort_values(["date", "numerai_ticker"]).reset_index(drop=True)
            matrix = []
            for name in names:
                base = runner._checkpoint("cv", label, name)
                if not runner._check_checkpoint(base):
                    raise ValueError(f"incomplete CV window: {label}/{name}")
                matrix.append(aligned_predictions(evaluation, base.with_suffix(".parquet")))
            matrix = np.column_stack(matrix)
            for date, indexes, scorer in runner.contexts(evaluation):
                scores = scorer.score(matrix[indexes])
                for i, name in enumerate(names):
                    rows.append(
                        {
                            "date": str(date.date()),
                            "window": label,
                            "model": name,
                            "ncorr": float(scores["ncorr"][i]),
                            "benchmark_contribution": float(scores["benchmark_contribution"][i]),
                        }
                    )
                value = float(
                    (scorer.target @ scorer.benchmark)
                    / (np.linalg.norm(scorer.target) * np.linalg.norm(scorer.benchmark))
                )
                rows.append(
                    {
                        "date": str(date.date()),
                        "window": label,
                        "model": runner.experiment.benchmark,
                        "ncorr": value,
                        "benchmark_contribution": 0.0,
                    }
                )
    scores = pd.DataFrame(rows)
    paired = scores.pivot(index="date", columns="model", values="ncorr")
    if paired.isna().any().any():
        raise ValueError("CV benchmark and candidate date coverage differ")
    result = {
        "fingerprint": runner.fingerprint,
        "report_source_sha256": file_hash(__file__),
        "analysis": "development CV comparison; no reselection and no independent significance claim",
        "windows": [w[0] for w in windows],
        "dates": len(paired),
        "models": {},
    }
    for name in [*names, runner.experiment.benchmark]:
        result["models"][name] = {
            "mean_ncorr": float(paired[name].mean()),
            "date_score_std": float(paired[name].std(ddof=0)),
            "mean_gain_vs_public_benchmark": float(
                (paired[name] - paired[runner.experiment.benchmark]).mean()
            ),
            "by_window": {
                w: float(g.ncorr.mean()) for w, g in scores.loc[scores.model == name].groupby("window")
            },
        }
    output = Path(run)
    if window:
        output = output / "diagnostics"
        output.mkdir(exist_ok=True)
    prefix = f"cv_benchmark_{window}" if window else "cv_benchmark"
    scores.to_csv(output / f"{prefix}_scores.csv", index=False)
    atomic_json(output / f"{prefix}_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rolling.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/v6.0"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--window", choices=["fold1", "fold2", "fold3"])
    args = parser.parse_args()
    print(json.dumps(compare(args.config, args.data, args.run, args.window), indent=2))


if __name__ == "__main__":
    main()
