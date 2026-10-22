"""Label-free inference from a trusted local export bundle; no refit or upload."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import research
from research.data import KEYS, atomic_json, file_hash, validate_keys, verify_snapshot
from research.scoring import neutralize_features, percentile


def predict(bundle, data, output):
    started = time.monotonic()
    bundle, data, output = Path(bundle), Path(data), Path(output)
    if output.suffix.lower() != ".csv" or output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("write predictions to a CSV outside the immutable model bundle")
    spec = json.loads((bundle / "export.json").read_text())
    if spec["dataset_version"] != "v6.0":
        raise ValueError("expected a v6.0 export bundle")
    source = Path(research.__file__).parent
    for name, expected in spec["runtime"]["source"].items():
        if file_hash(source / name) != expected:
            raise ValueError(f"bundle research source differs: {name}")
    for name, expected in spec["runtime"]["dependencies"].items():
        if version(name) != expected:
            raise ValueError(f"bundle dependency differs: {name}; use the recorded environment")
    for name, expected in spec["artifacts"].items():
        if file_hash(bundle / name) != expected:
            raise ValueError(f"bundle artifact changed: {name}")
    snapshot = verify_snapshot(data, required=("live.parquet",))
    recipe = spec["recipe"]
    if not recipe["weights"] or min(recipe["weights"].values()) <= 0:
        raise ValueError("bundle requires positive nonempty weights")
    models, binaries = {}, {}
    for name in recipe["weights"]:
        # Pickle is executable: load only your own trusted, hash-checked artifacts.
        identity = spec["artifacts"][f"{name}.pkl"]
        if identity not in binaries:
            with (bundle / f"{name}.pkl").open("rb") as handle:
                binaries[identity] = pickle.load(handle)
        models[name] = binaries[identity]
    features = list(
        dict.fromkeys(
            [*recipe["neutralization_features"], *(c for model in models.values() for c in model.features)]
        )
    )
    frame = pd.read_parquet(data / "live.parquet", columns=[*KEYS, "data_type", *features])
    validate_keys(frame)
    if frame.empty or frame.date.nunique() != 1 or not (frame.data_type == "live").all():
        raise ValueError("expected exactly one complete live date")
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    date = frame.date.iloc[0]
    cutoff = date - pd.offsets.BDay(recipe["purge_business_days"])
    if any(pd.Timestamp(model.metadata["last_date"]) >= cutoff for model in models.values()):
        raise ValueError("live date is too early for the model's training history")
    evaluated, columns = {}, []
    for name, model in models.items():
        identity = spec["artifacts"][f"{name}.pkl"]
        if identity not in evaluated:
            evaluated[identity] = model.predict(frame)
        columns.append(evaluated[identity])
    matrix = np.column_stack(columns)
    weights = np.array(list(recipe["weights"].values()), dtype=float)
    ranks = percentile(matrix)
    total = sum(recipe["weights"].values())
    blend = sum(ranks[:, i] * weight / total for i, weight in enumerate(weights))
    threads = max(model.metadata["threads"] for model in models.values())
    with threadpool_limits(limits=threads):
        values = neutralize_features(
            blend, frame[recipe["neutralization_features"]].to_numpy(), recipe["neutralization"]
        )
    if not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
        raise ValueError("invalid percentile predictions")
    result = frame[KEYS].copy()
    result["date"] = result.date.dt.strftime("%Y-%m-%d")
    result["prediction"] = values
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    result.to_csv(temp, index=False)
    temp.replace(output)
    report = {
        "dataset_version": "v6.0",
        "round": snapshot["round"],
        "date": str(date.date()),
        "rows": len(result),
        "bundle_sha256": file_hash(bundle / "export.json"),
        "live_sha256": snapshot["files"]["live.parquet"]["sha256"],
        "predictions_sha256": file_hash(output),
        "seconds": round(time.monotonic() - started, 3),
        "unique_models_evaluated": len(evaluated),
        "status": "local inference only; no labels, refitting, upload, staking or trading",
    }
    atomic_json(output.with_suffix(".json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.bundle, args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
