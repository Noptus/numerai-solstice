"""Pinned public snapshots and schema-checked, column-pruned v6 data loading."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

KEYS = ["numerai_ticker", "date"]
CORE_FILES = (
    "features.json",
    "train.parquet",
    "validation.parquet",
    "validation_benchmark_models.parquet",
    "validation_neutralizer.parquet",
)
LIVE_FILES = ("live.parquet", "live_neutralizer.parquet", "live_benchmark_models.parquet")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def download_snapshot(root, *, include_live=False, client=None):
    """Never submits; a snapshot's round is fixed on its first invocation."""
    from numerapi import NumerAPI

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    client = client or NumerAPI(show_progress_bars=False)
    path = root / "snapshot.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["dataset_version"] != "v6.0":
            raise ValueError("snapshot is not v6.0")
    else:
        manifest = {
            "dataset_version": "v6.0",
            "round": client.get_current_round(),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": {},
        }
        atomic_json(path, manifest)
    required = [*CORE_FILES, *(LIVE_FILES if include_live else ())]
    available = set(client.list_datasets(round_num=manifest["round"]))
    missing = [name for name in required if f"v6.0/{name}" not in available]
    if missing:
        raise ValueError(f"required files unavailable for pinned round: {missing}")
    for name in required:
        dest = root / name
        existing = manifest["files"].get(name)
        if existing and dest.is_file():
            if dest.stat().st_size == existing["bytes"] and file_hash(dest) == existing["sha256"]:
                continue
            raise ValueError(f"cached snapshot file changed: {name}")
        print(json.dumps({"event": "download", "file": name, "round": manifest["round"]}), flush=True)
        part = dest.with_suffix(dest.suffix + ".part")
        client.download_dataset(f"v6.0/{name}", str(part), round_num=manifest["round"])
        part.replace(dest)
        manifest["files"][name] = {"sha256": file_hash(dest), "bytes": dest.stat().st_size}
        atomic_json(path, manifest)
    return manifest


def verify_snapshot(root, required=CORE_FILES):
    root = Path(root)
    manifest = json.loads((root / "snapshot.json").read_text())
    if manifest["dataset_version"] != "v6.0":
        raise ValueError("expected v6.0 snapshot")
    for name in required:
        entry = manifest["files"].get(name)
        path = root / name
        if (
            not entry
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            raise ValueError(f"snapshot integrity failed: {name}")
    return manifest


def feature_sets(root, names):
    root = Path(root)
    metadata = json.loads((root / "features.json").read_text())
    schema = pq.read_schema(root / "train.parquet")
    result = {}
    for name in sorted(set(names)):
        if name not in metadata["feature_sets"]:
            raise ValueError(f"unknown feature set: {name}")
        cols = [
            c
            for c in metadata["feature_sets"][name]
            if c != "feature_country"
            and c in schema.names
            and (pa.types.is_integer(schema.field(c).type) or pa.types.is_floating(schema.field(c).type))
        ]
        missing = set(metadata["feature_sets"][name]) - set(schema.names)
        if missing or not cols:
            raise ValueError(f"invalid feature metadata for {name}: missing={sorted(missing)}")
        result[name] = cols
    return result


def validate_keys(frame):
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("null or duplicated ticker/date keys")
    if (frame["numerai_ticker"].astype(str).str.len() == 0).any():
        raise ValueError("empty ticker")
    return frame


def load_training_frame(root, experiment, max_date=None):
    """Stream only needed feature rows while preserving exact training samples.

    Every evaluation observation is retained. Other dates keep the union of
    each candidate seed's deterministic per-date sample. Resampling that union
    for a candidate yields exactly the same keys as sampling the full history.
    Date/data_type predicates apply before batches reach Python.
    """
    from research.validation import sample_training

    root = Path(root)
    sets = feature_sets(root, [m.feature_set for m in experiment.models] + ["medium"])
    features = list(dict.fromkeys(c for cols in sets.values() for c in cols))
    targets = list(dict.fromkeys([experiment.primary_target, *(m.target for m in experiment.models)]))
    columns = [*KEYS, "data_type", *features, *targets]
    sources, narrow = [], []
    for name, allowed in (("train.parquet", "train"), ("validation.parquet", "validation")):
        missing = set(columns) - set(pq.read_schema(root / name).names)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        filters = [("data_type", "==", allowed)]
        predicate = ds.field("data_type") == allowed
        if max_date is not None:
            filters.append(("date", "<=", str(max_date)))
            predicate = predicate & (ds.field("date") <= str(max_date))
        keys = pd.read_parquet(root / name, columns=[*KEYS, experiment.primary_target], filters=filters)
        narrow.append(keys.dropna(subset=[experiment.primary_target]))
        sources.append((name, predicate))
    keys = pd.concat(narrow, ignore_index=True)
    del narrow
    validate_keys(keys)
    if keys.empty:
        raise ValueError("no mature primary targets")
    primary = keys[experiment.primary_target].to_numpy()
    if not np.isfinite(primary).all() or ((primary < 0) | (primary > 1)).any():
        raise ValueError("invalid bounded primary target")
    evaluation = (keys.date >= pd.Timestamp(experiment.holdout_start)).to_numpy(copy=True)
    windows = [*experiment.folds, (experiment.calibration_start, experiment.calibration_end)]
    for left, right in windows:
        evaluation |= keys.date.between(left, right).to_numpy()
    selected = [keys.loc[evaluation, KEYS]]
    history = keys.loc[~evaluation, KEYS]
    for seed in sorted({model.seed for model in experiment.models}):
        selected.append(sample_training(history, experiment.max_train_rows_per_date, seed)[KEYS])
    wanted = pd.concat(selected, ignore_index=True).drop_duplicates(KEYS)
    index = pd.MultiIndex.from_frame(wanted)
    del keys, history, selected, wanted
    frames = []
    for name, predicate in sources:
        scanner = ds.dataset(root / name, format="parquet").scanner(
            columns=columns, filter=predicate, batch_size=65_536, use_threads=False
        )
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            frame["date"] = pd.to_datetime(frame.date)
            keep = index.get_indexer(pd.MultiIndex.from_frame(frame[KEYS])) >= 0
            if keep.any():
                frames.append(frame.loc[keep].copy())
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != len(index):
        raise ValueError("streamed feature rows differ from selected ticker/date keys")
    validate_keys(frame)
    for target in targets:
        values = frame[target].dropna().to_numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"invalid bounded target: {target}")
    return frame.sort_values(["date", "numerai_ticker"]).reset_index(drop=True), sets


def load_date_auxiliary(root, name, dates, columns=None):
    iso_dates = pd.DatetimeIndex(dates).strftime("%Y-%m-%d").tolist()
    frame = pd.read_parquet(Path(root) / name, columns=columns, filters=[("date", "in", iso_dates)])
    validate_keys(frame)
    return frame
