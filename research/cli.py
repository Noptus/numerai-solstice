"""Public-data research commands. No credentials, trading or upload commands."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from research.config import load_experiment
from research.data import download_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["plan", "download", "run", "train", "score", "select", "finalize"]
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mac.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/v6.0"))
    parser.add_argument("--output", type=Path, default=Path("runs/mac-v6"))
    parser.add_argument("--stage", choices=["cv", "calibration", "holdout"], default="cv")
    parser.add_argument("--include-live", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        config = load_experiment(args.config)
        print(json.dumps({"digest": config.digest, **config.as_dict()}, indent=2))
        return
    if args.command == "download":
        download_snapshot(args.data, include_live=args.include_live)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another process owns this run directory") from exc
        from research.runner import Runner

        runner = Runner(args.config, args.data, args.output)
        if args.command in {"train", "score"}:
            getattr(runner, args.command)(args.stage)
        else:
            getattr(runner, args.command)()


if __name__ == "__main__":
    main()
