"""Restore the exact public round and bytes recorded by a packaged experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.data import atomic_json, download_snapshot, file_hash
from scripts.verify_report import verify


def restore(reference, data, client=None):
    reference, data = Path(reference), Path(data)
    verify(reference)
    plan = json.loads((reference / "plan.json").read_text())
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = data / "snapshot.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["round"] != plan["round"] or existing["dataset_version"] != plan["experiment"]["version"]:
            raise ValueError("destination contains another snapshot; use a fresh directory")
    else:
        atomic_json(
            manifest_path,
            {
                "round": plan["round"],
                "dataset_version": plan["experiment"]["version"],
                "files": plan["dataset"],
                "restored_from": plan["fingerprint"],
            },
        )
    download_snapshot(data, include_live=True, client=client)
    for name, expected in plan["dataset"].items():
        path = data / name
        if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            raise ValueError(f"provider bytes no longer match this reference: {name}")
    return {
        "round": plan["round"],
        "verified_files": len(plan["dataset"]),
        "fingerprint": plan["fingerprint"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.reference, args.data), indent=2))


if __name__ == "__main__":
    main()
