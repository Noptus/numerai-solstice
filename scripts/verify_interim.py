"""Verify partial-study evidence without implying a completed selection or holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.data import file_hash
from research.runner import digest


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "evidence_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"published evidence changed: {name}")
    plan = json.loads((root / "plan.json").read_text())
    identity = {key: plan[key] for key in ("experiment", "source", "dependencies", "dataset", "round")}
    if digest(identity) != plan["fingerprint"]:
        raise ValueError("plan identity mismatch")
    status = json.loads((root / "interim_status.json").read_text())
    if status["fingerprint"] != plan["fingerprint"] or status["study_complete"]:
        raise ValueError("invalid partial-study status")
    if status["selection_completed"] or status["holdout_opened"]:
        raise ValueError("this archive must not imply completed later stages")
    fits = json.loads((root / "completed_refits.json").read_text())
    if len(fits) != status["completed_refits"]:
        raise ValueError("completed-refit count mismatch")
    gap = pd.offsets.BDay(plan["experiment"]["purge_business_days"])
    for fit in fits:
        if fit["fingerprint"] != plan["fingerprint"]:
            raise ValueError("fit identity mismatch")
        if pd.Timestamp(fit["last_date"]) + gap >= pd.Timestamp(fit["evaluation_first_date"]):
            raise ValueError("outer purge violated")
        if "inner_fit_last_date" in fit:
            if pd.Timestamp(fit["inner_fit_last_date"]) + gap >= pd.Timestamp(fit["inner_stop_first_date"]):
                raise ValueError("inner purge violated")
        if set(fit["artifacts"]) != {".pkl", ".parquet"}:
            raise ValueError("incomplete checkpoint provenance")
    scores = pd.read_csv(root / "cv_benchmark_fold1_scores.csv")
    if scores.duplicated(["window", "date", "model"]).any():
        raise ValueError("duplicate development scores")
    comparison = json.loads((root / "cv_benchmark_fold1_comparison.json").read_text())
    if comparison["fingerprint"] != plan["fingerprint"] or comparison["windows"] != ["fold1"]:
        raise ValueError("development comparison identity mismatch")
    models = {item["name"] for item in plan["experiment"]["models"]} | {plan["experiment"]["benchmark"]}
    if set(scores.model) != models or set(comparison["models"]) != models:
        raise ValueError("missing candidate or benchmark")
    dates = set(scores.date)
    if len(dates) != comparison["dates"] or len(dates) != 53 or set(scores.window) != {"fold1"}:
        raise ValueError("unexpected development window")
    benchmark = comparison["models"][plan["experiment"]["benchmark"]]["mean_ncorr"]
    for name, group in scores.groupby("model"):
        if set(group.date) != dates or not np.isfinite(group.ncorr).all():
            raise ValueError("unmatched or nonfinite development scores")
        expected = comparison["models"][name]
        np.testing.assert_allclose(group.ncorr.mean(), expected["mean_ncorr"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            group.ncorr.mean() - benchmark, expected["mean_gain_vs_public_benchmark"], atol=1e-12, rtol=0
        )
    return {
        "verified_files": len(manifest["files"]),
        "completed_refits": len(fits),
        "development_dates": len(dates),
        "meaning": "internally consistent partial evidence; no completed selection, benchmark win or live validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(verify(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
