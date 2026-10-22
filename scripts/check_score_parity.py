"""Compare fast scoring with the official package on one completed CV date."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numerai_tools.scoring import neutral_contribution, neutral_correlation
from threadpoolctl import threadpool_limits

from research.data import KEYS, atomic_json, load_date_auxiliary
from research.scoring import DateScorer


def check(data, predictions, date):
    root = Path(data)
    date = pd.Timestamp(date)
    filters = [("date", "==", str(date.date())), ("data_type", "==", "validation")]
    targets = pd.read_parquet(
        root / "validation.parquet", columns=[*KEYS, "target_jupiter_60"], filters=filters
    ).dropna()
    targets["date"] = pd.to_datetime(targets.date)
    pred = pd.read_parquet(predictions)
    pred = pred.loc[pd.to_datetime(pred.date) == date].copy()
    pred["date"] = pd.to_datetime(pred.date)
    neutral = load_date_auxiliary(root, "validation_neutralizer.parquet", [date])
    bench = load_date_auxiliary(
        root, "validation_benchmark_models.parquet", [date], [*KEYS, "lgbm_jupiter_60"]
    )
    joined = (
        targets.merge(pred, on=KEYS, validate="one_to_one")
        .merge(neutral, on=KEYS, validate="one_to_one")
        .merge(bench, on=KEYS, validate="one_to_one")
        .reset_index(drop=True)
    )
    if len(joined) != len(targets) or joined.empty:
        raise ValueError("parity requires the full target universe for this date")
    cols = [c for c in joined if c.startswith("neutralizer_column_")]
    with threadpool_limits(limits=2):
        fast = DateScorer.prepare(joined[cols], joined.target_jupiter_60, joined.lgbm_jupiter_60).score(
            joined.prediction
        )
        official = {
            "ncorr": neutral_correlation(joined[["prediction"]], joined[cols], joined.target_jupiter_60),
            "benchmark_contribution": neutral_contribution(
                joined[["prediction"]], joined.lgbm_jupiter_60, joined[cols], joined.target_jupiter_60
            ),
        }
    fast = {k: float(v[0]) for k, v in fast.items()}
    official = {k: float(np.asarray(v).ravel()[0]) for k, v in official.items()}
    for key in official:
        np.testing.assert_allclose(fast[key], official[key], atol=1e-8, rtol=1e-7)
    return {
        "date": str(date.date()),
        "rows": len(joined),
        "neutralizer_columns": len(cols),
        "fast": fast,
        "official_numerai_tools_0_7_2": official,
        "max_absolute_difference": max(abs(fast[k] - official[k]) for k in official),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/v6.0"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--date", default="2019-06-07")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.data, args.predictions, args.date)
    if args.output:
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
