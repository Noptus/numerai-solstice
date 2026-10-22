"""Freeze a trusted bundle, collect future weekly predictions, score matured dates.

Local clocks are not trusted timestamps: publish the plan and each commitment
before outcomes mature. Raw predictions stay local. No uploads or trading.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.data import KEYS, atomic_json, file_hash, load_date_auxiliary, validate_keys
from research.runner import digest
from research.scoring import DateScorer, summary
from scripts.predict_bundle import predict


def utc_now():
    return datetime.now(timezone.utc)


def read_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    expected = plan.pop("plan_sha256")
    if digest(plan) != expected:
        raise ValueError("prospective plan was modified")
    return {**plan, "plan_sha256": expected}


def freeze(bundle, root):
    bundle, root = Path(bundle), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise ValueError("prospective plan already exists; never replace a frozen plan")
    exported = json.loads((bundle / "export.json").read_text())
    for name, expected in exported["artifacts"].items():
        if file_hash(bundle / name) != expected:
            raise ValueError("bundle artifacts changed")
    now = utc_now()
    plan = {
        "schema": 1,
        "frozen_at_utc": now.isoformat(),
        "first_prediction_date": str((pd.Timestamp(now.date()) + pd.offsets.BDay(1)).date()),
        "bundle_sha256": file_hash(bundle / "export.json"),
        "experiment_fingerprint": exported["identity"]["fingerprint"],
        "runtime": exported["runtime"],
        "recipe": exported["recipe"],
        "target": "target_jupiter_60",
        "benchmark": "lgbm_jupiter_60",
        "minimum_weekly_dates": 52,
        "bootstrap_block_dates": 16,
        "model_policy": "frozen bundle, no refits or model reselection during this evaluation",
        "sampling_policy": "first captured complete live date per ISO week; never replace a date",
        "primary_metric": "paired mean weekly NCORR difference against the public benchmark",
        "timestamp_requirement": "publish this plan and every commitment before outcome maturity",
        "timestamp_trust": "local clock only until independently anchored in public history",
        "limitations": [
            "52 weekly dates is a reporting threshold, not a statistical power guarantee.",
            "Overlapping 60-day labels reduce effective sample size.",
            "Any unrecorded/missing weeks and model changes must be disclosed.",
            "This evaluates the frozen exported model, not the historical quarterly-refit algorithm.",
        ],
    }
    plan["plan_sha256"] = digest(plan)
    atomic_json(root / "plan.json", plan)
    return plan


def collect(bundle, data, root):
    bundle, data, root = Path(bundle), Path(data), Path(root)
    plan = read_plan(root)
    if file_hash(bundle / "export.json") != plan["bundle_sha256"]:
        raise ValueError("model differs from frozen bundle")
    frame = pd.read_parquet(data / "live.parquet", columns=KEYS + ["data_type"])
    validate_keys(frame)
    if frame.empty or frame.date.nunique() != 1 or not (frame.data_type == "live").all():
        raise ValueError("expected a complete live date")
    date, today = frame.date.iloc[0], pd.Timestamp(utc_now().date())
    if date < pd.Timestamp(plan["first_prediction_date"]):
        raise ValueError("prediction date precedes the prospective boundary")
    if date > today or (today - date).days > 1:
        raise ValueError("collect only a fresh live date, never a historical reconstruction")
    year, week, _ = date.isocalendar()
    records = root / "commitments"
    records.mkdir(exist_ok=True)
    path = records / f"{year}-W{week:02d}.json"
    if path.exists():
        raise ValueError("this week's observation is immutable; do not replace or cherry-pick it")
    csv = root / "predictions" / f"{date.date()}.csv"
    if csv.exists():
        raise ValueError("uncommitted prediction file exists; investigate before collecting")
    report = predict(bundle, data, csv)
    commitment = {
        **report,
        "plan_sha256": plan["plan_sha256"],
        "captured_at_utc": utc_now().isoformat(),
        "prediction_file": f"predictions/{csv.name}",
        "timestamp_trust": "publish this hash before maturity for an external timestamp anchor",
    }
    atomic_json(path, commitment)
    return commitment


def evaluate(root, data):
    root, data = Path(root), Path(data)
    plan = read_plan(root)
    records = sorted((root / "commitments").glob("*.json"))
    if not records:
        raise ValueError("no prospective observations recorded")
    rows, pending, dates_seen, weeks_seen = [], [], set(), set()
    inputs = {"plan_sha256": plan["plan_sha256"], "commitments": {}, "scoring_files": {}}
    for name in (
        "validation.parquet",
        "validation_neutralizer.parquet",
        "validation_benchmark_models.parquet",
    ):
        inputs["scoring_files"][name] = file_hash(data / name)
    with threadpool_limits(limits=6):
        for path in records:
            entry = json.loads(path.read_text())
            date = pd.Timestamp(entry["date"])
            week = tuple(date.isocalendar()[:2])
            if date in dates_seen or week in weeks_seen:
                raise ValueError("duplicate prospective date or week")
            dates_seen.add(date)
            weeks_seen.add(week)
            captured = pd.Timestamp(entry["captured_at_utc"])
            if (
                entry["plan_sha256"] != plan["plan_sha256"]
                or entry["bundle_sha256"] != plan["bundle_sha256"]
                or date < pd.Timestamp(plan["first_prediction_date"])
                or captured < pd.Timestamp(plan["frozen_at_utc"])
                or not 0 <= (captured.date() - date.date()).days <= 1
            ):
                raise ValueError("commitment violates the frozen prospective boundary")
            prediction_path = root / entry["prediction_file"]
            if not prediction_path.resolve().is_relative_to((root / "predictions").resolve()):
                raise ValueError("prediction file is outside the journal")
            if file_hash(prediction_path) != entry["predictions_sha256"]:
                raise ValueError("committed predictions were modified")
            pred = pd.read_csv(prediction_path)
            validate_keys(pred)
            if len(pred) != entry["rows"] or pred.date.nunique() != 1 or pred.date.iloc[0] != date:
                raise ValueError("committed prediction coverage differs")
            if not np.isfinite(pred.prediction).all():
                raise ValueError("nonfinite committed predictions")
            inputs["commitments"][path.name] = file_hash(path)
            target = pd.read_parquet(
                data / "validation.parquet",
                columns=[*KEYS, "data_type", plan["target"]],
                filters=[("date", "==", str(date.date()))],
            )
            validate_keys(target)
            target = target.loc[target.data_type == "validation"]
            # Wait for the complete committed universe; never score the convenient subset.
            if target.empty or target[plan["target"]].isna().any():
                pending.append(entry["date"])
                continue
            aligned = pred.merge(target, on=KEYS, how="outer", validate="one_to_one", indicator=True)
            if not (aligned._merge == "both").all():
                raise ValueError("prospective target universe differs from committed predictions")
            neutral = load_date_auxiliary(data, "validation_neutralizer.parquet", [date])
            bench = load_date_auxiliary(
                data, "validation_benchmark_models.parquet", [date], [*KEYS, plan["benchmark"]]
            )
            cols = [c for c in neutral if c.startswith("neutralizer_column_")]
            if not cols:
                raise ValueError("missing official neutralizers")
            aligned = aligned.merge(neutral[[*KEYS, *cols]], on=KEYS, how="left", validate="one_to_one")
            aligned = aligned.merge(bench, on=KEYS, how="left", validate="one_to_one")
            scorer = DateScorer.prepare(
                aligned[cols].to_numpy(),
                aligned[plan["target"]].to_numpy(),
                aligned[plan["benchmark"]].to_numpy(),
            )
            scores = scorer.score(aligned[["prediction", plan["benchmark"]]].to_numpy())
            rows.append(
                {
                    "date": entry["date"],
                    "model_ncorr": float(scores["ncorr"][0]),
                    "benchmark_ncorr": float(scores["ncorr"][1]),
                    "benchmark_contribution": float(scores["benchmark_contribution"][0]),
                }
            )
    rows.sort(key=lambda row: row["date"])
    delta = np.array([r["model_ncorr"] - r["benchmark_ncorr"] for r in rows])
    first, last = min(dates_seen), max(dates_seen)
    captured_weeks = {(d.isocalendar().year, d.isocalendar().week) for d in dates_seen}
    expected_weeks = {
        (d.isocalendar().year, d.isocalendar().week) for d in pd.date_range(first, last, freq="D")
    }
    return {
        "inputs": inputs,
        "scores": rows,
        "mature_dates": len(rows),
        "pending_dates": pending,
        "missing_iso_weeks_between_first_and_last": sorted(expected_weeks - captured_weeks),
        "paired_ncorr_gain_vs_public_benchmark": summary(delta, block_size=plan["bootstrap_block_dates"])
        if len(delta) >= 2
        else None,
        "status": "interim; insufficient dates"
        if len(rows) < plan["minimum_weekly_dates"]
        else "reporting threshold reached; inspect interval, missing weeks and timestamp anchors",
        "caveat": "Local journals alone do not prove timely prediction; verify public timestamp anchors. "
        "No automated claim of a benchmark win. Scores are not returns.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "collect", "score"])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command in {"freeze", "collect"} and args.bundle is None:
        parser.error("--bundle is required")
    if args.command in {"collect", "score"} and args.data is None:
        parser.error("--data is required")
    if args.command == "freeze":
        result = freeze(args.bundle, args.journal)
    elif args.command == "collect":
        result = collect(args.bundle, args.data, args.journal)
    else:
        result = evaluate(args.journal, args.data)
        if args.output:
            atomic_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
