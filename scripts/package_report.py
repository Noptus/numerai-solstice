"""Package public-safe research evidence; never copies datasets or model binaries."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from research.data import atomic_json, file_hash

REPORT_FILES = (
    "plan.json",
    "cv_scores.csv",
    "cv_summary.json",
    "calibration_scores.csv",
    "calibration_summary.json",
    "calibration_trials.json",
    "selection.json",
    "holdout_access.json",
    "holdout_scores.csv",
    "holdout_summary.json",
    "result.json",
    "benchmark_comparison.json",
    "holdout_comparison_scores.csv",
)


def package(run, output, log=None):
    run, output = Path(run), Path(output)
    if run.resolve() == output.resolve():
        raise ValueError("package evidence into a separate directory")
    missing = [name for name in REPORT_FILES if not (run / name).is_file()]
    if missing:
        raise ValueError(f"incomplete research evidence: {missing}")
    output.mkdir(parents=True, exist_ok=True)
    for name in REPORT_FILES:
        shutil.copyfile(run / name, output / name)
    fits = []
    for path in sorted(run.glob("*/*/*.json")):
        meta = json.loads(path.read_text())
        if "evaluation_rows" in meta:
            for i, fit in enumerate(meta.get("refits", [meta])):
                fits.append({"checkpoint": str(path.relative_to(run)), "refit_index": i + 1, **fit})
    atomic_json(output / "model_fits.json", fits)
    live = run / "live-export/export.json"
    if live.exists():
        shutil.copyfile(live, output / "live_export.json")
    if log:
        events = []
        for line in Path(log).read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict) and "event" in entry:
                entry.pop("output", None)
                entry.pop("path", None)
                events.append(entry)
        atomic_json(output / "execution.json", events)
    manifest = {
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "date scores, run provenance and fit metadata; no raw observations or model binaries",
        "files": {
            path.name: file_hash(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "evidence_manifest.json"
        },
    }
    atomic_json(output / "evidence_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    package(args.run, args.output, args.log)


if __name__ == "__main__":
    main()
