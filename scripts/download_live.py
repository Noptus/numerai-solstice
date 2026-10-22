"""Pin and download just one public v6 live file for inexpensive inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.data import atomic_json, file_hash, verify_snapshot


def download(data, round_num=None, client=None):
    from numerapi import NumerAPI

    client = client or NumerAPI(show_progress_bars=False)
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    path = data / "snapshot.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["dataset_version"] != "v6.0" or (
            round_num is not None and manifest["round"] != round_num
        ):
            raise ValueError("destination contains another snapshot; use a fresh directory")
    else:
        pinned = round_num if round_num is not None else client.get_current_round()
        if pinned < 1:
            raise ValueError("round must be positive")
        manifest = {"dataset_version": "v6.0", "round": pinned, "files": {}}
        atomic_json(path, manifest)
    dest = data / "live.parquet"
    if dest.exists():
        return verify_snapshot(data, required=("live.parquet",))
    available = client.list_datasets(round_num=manifest["round"])
    if "v6.0/live.parquet" not in available:
        raise ValueError("live v6 file unavailable for the pinned round")
    temp = dest.with_suffix(".parquet.part")
    client.download_dataset("v6.0/live.parquet", str(temp), round_num=manifest["round"])
    temp.replace(dest)
    manifest["files"][dest.name] = {"bytes": dest.stat().st_size, "sha256": file_hash(dest)}
    atomic_json(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    manifest = download(args.data, args.round)
    print(json.dumps({"round": manifest["round"], "files": manifest["files"]}, indent=2))


if __name__ == "__main__":
    main()
