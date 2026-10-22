"""Verify a portable reference report using hashes and saved per-date scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.data import file_hash
from research.runner import digest
from research.scoring import summary


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "evidence_manifest.json").read_text())
    for name, value in manifest["files"].items():
        path = root / name
        if not path.is_file() or file_hash(path) != value:
            raise ValueError(f"published evidence changed: {name}")
    plan = json.loads((root / "plan.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    stored = selection.pop("selection_sha256")
    if digest(selection) != stored or selection["fingerprint"] != plan["fingerprint"]:
        raise ValueError("selection identity mismatch")
    if selection["trial_log_sha256"] != file_hash(root / "calibration_trials.json"):
        raise ValueError("selection trial log mismatch")
    access = json.loads((root / "holdout_access.json").read_text())
    if access["selection_sha256"] != stored:
        raise ValueError("holdout selection mismatch")
    if pd.Timestamp(access["opened_at_utc"]) < pd.Timestamp(selection["frozen_at_utc"]):
        raise ValueError("holdout opened before selection")
    block = plan["experiment"]["bootstrap_block_dates"]
    for stage in ("cv", "calibration", "holdout"):
        scores = pd.read_csv(root / f"{stage}_scores.csv")
        if scores.duplicated(["date", "model"]).any():
            raise ValueError("duplicate date scores")
        expected = json.loads((root / f"{stage}_summary.json").read_text())
        for name, group in scores.groupby("model", sort=True):
            for metric in ("ncorr", "benchmark_contribution"):
                measured = summary(group[metric], block_size=block)
                for key in ("mean", "std", "mean_ci95", "positive_fraction", "worst_date", "max_drawdown"):
                    np.testing.assert_allclose(measured[key], expected[name][metric][key], atol=1e-12)
    for fit in json.loads((root / "model_fits.json").read_text()):
        if fit["fingerprint"] != plan["fingerprint"]:
            raise ValueError("fit provenance differs from run plan")
        gap = pd.offsets.BDay(plan["experiment"]["purge_business_days"])
        if pd.Timestamp(fit["last_date"]) + gap >= pd.Timestamp(fit["evaluation_first_date"]):
            raise ValueError("fit metadata violates the outer purge")
        if "inner_fit_last_date" in fit and pd.Timestamp(fit["inner_fit_last_date"]) + gap >= pd.Timestamp(
            fit["inner_stop_first_date"]
        ):
            raise ValueError("fit metadata violates the inner purge")
    result = json.loads((root / "result.json").read_text())
    status = plan["experiment"].get("historical_evaluation_status", "reserved")
    if result.get("evaluation_status", "reserved") != status:
        raise ValueError("historical evaluation status differs from frozen plan")
    if (
        status == "reused_exploratory"
        and result["prior_experiment_fingerprint"] != plan["experiment"]["prior_experiment_fingerprint"]
    ):
        raise ValueError("prior experiment link differs")
    paired = pd.read_csv(root / "holdout_scores.csv").pivot(index="date", columns="model", values="ncorr")
    gain = summary(paired.selected_ensemble - paired[result["baseline"]], block_size=block)
    np.testing.assert_allclose(
        gain["mean_ci95"], result["paired_ncorr_gain_vs_baseline"]["mean_ci95"], atol=1e-12
    )
    return {
        "verified_files": len(manifest["files"]),
        "fingerprint": plan["fingerprint"],
        "meaning": "saved evidence is internally consistent; not an independent replication or live validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
