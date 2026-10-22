"""Refit the frozen selection and export one pinned date. Never uploads predictions."""

from __future__ import annotations

import argparse
import fcntl
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.data import KEYS, LIVE_FILES, atomic_json, file_hash, validate_keys, verify_snapshot
from research.models import fit_model
from research.runner import Runner, digest, event
from research.validation import prior_training_mask, sample_training


def export(config, data, run, output):
    runner = Runner(config, data, run)
    if not (runner.output / "result.json").is_file():
        raise ValueError("complete the frozen historical evaluation before exporting")
    selected = runner._selection()
    manifest = verify_snapshot(data, required=LIVE_FILES)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Immutable export identity, including helper source, prevents stale resume.
    identity = {
        "fingerprint": runner.fingerprint,
        "selection_sha256": selected["selection_sha256"],
        "snapshot_round": manifest["round"],
        "export_source_sha256": file_hash(__file__),
    }
    plan = output / "export_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != identity:
        raise ValueError("export identity changed; choose a fresh output directory")
    atomic_json(plan, identity)
    report_path = output / "export.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["identity"] != identity:
            raise ValueError("export provenance differs")
        for name, value in report["artifacts"].items():
            if file_hash(output / name) != value:
                raise ValueError(f"export artifact modified: {name}")
        return report
    frame, sets = runner.frame("holdout")
    required = list(dict.fromkeys(c for cols in sets.values() for c in cols))
    live = pd.read_parquet(Path(data) / "live.parquet", columns=[*KEYS, "data_type", *required])
    validate_keys(live)
    if live.empty or not (live.data_type == "live").all() or live.date.nunique() != 1:
        raise ValueError("expected exactly one complete live date")
    live = live.sort_values(KEYS).reset_index(drop=True)
    date = live.date.iloc[0]
    eligible = frame.loc[prior_training_mask(frame.date, date, runner.experiment.purge_business_days)]
    specs = {spec.name: spec for spec in runner.experiment.models}
    names, predictions, metadata, artifacts = [], [], {}, {}
    for name in selected["weights"]:
        event("live_fit_start", model=name, date=str(date.date()))
        spec = specs[name]
        sampled = sample_training(eligible, runner.experiment.max_train_rows_per_date, spec.seed)
        fitted = fit_model(sampled, sets[spec.feature_set], spec, runner.experiment)
        predictions.append(fitted.predict(live))
        names.append(name)
        path = output / f"{name}.pkl"
        part = path.with_suffix(".pkl.tmp")
        with part.open("wb") as handle:
            pickle.dump(fitted, handle, protocol=5)
        part.replace(path)
        artifacts[path.name] = file_hash(path)
        metadata[name] = fitted.metadata
        event("live_fit_complete", model=name, seconds=fitted.metadata["fit_seconds"])
        del fitted, sampled
    with threadpool_limits(limits=runner.experiment.threads):
        prediction = runner.blend(
            live, np.column_stack(predictions), names, selected["weights"], selected["neutralization"]
        )
    if not np.isfinite(prediction).all() or not ((prediction > 0) & (prediction < 1)).all():
        raise ValueError("invalid cross-sectional percentile predictions")
    result = live[KEYS].copy()
    result["date"] = result.date.dt.strftime("%Y-%m-%d")
    result["prediction"] = prediction
    csv = output / "predictions.csv"
    part = csv.with_suffix(".csv.tmp")
    result.to_csv(part, index=False)
    part.replace(csv)
    artifacts[csv.name] = file_hash(csv)
    report = {
        "identity": identity,
        "date": str(date.date()),
        "rows": len(result),
        "models": metadata,
        "key_sha256": digest(result[KEYS].to_dict(orient="records")),
        "artifacts": artifacts,
        "status": "dated local predictions only; not uploaded, staked, or traded",
    }
    atomic_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mac.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/v6.0"))
    parser.add_argument("--run", type=Path, default=Path("runs/mac-v6"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another process owns this run directory") from exc
        report = export(args.config, args.data, args.run, args.output)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
