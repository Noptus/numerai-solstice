"""Checkpointed, finite chronological research; holdout opened only after selection."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.config import load_experiment
from research.data import (
    KEYS,
    atomic_json,
    feature_sets,
    file_hash,
    load_date_auxiliary,
    load_training_frame,
    validate_keys,
    verify_snapshot,
)
from research.models import fit_model
from research.scoring import DateScorer, neutralize_features, percentile, summary
from research.validation import prior_training_mask, refit_windows, sample_training


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def event(event_type, **values):
    print(
        json.dumps({"event": event_type, "utc": datetime.now(timezone.utc).isoformat(), **values}), flush=True
    )


def aligned_predictions(frame, path):
    pred = pd.read_parquet(path)
    validate_keys(pred)
    if len(pred) != len(frame):
        raise ValueError(f"prediction coverage changed: {path}")
    joined = frame[KEYS].merge(pred, on=KEYS, how="left", validate="one_to_one", sort=False)
    values = joined.prediction.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"missing/nonfinite predictions: {path}")
    return values


class Runner:
    def __init__(self, config, data, output):
        self.config_path, self.data, self.output = Path(config), Path(data), Path(output)
        self.experiment = load_experiment(self.config_path)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = verify_snapshot(self.data)
        source = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        dependencies = {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyarrow", "numerai-tools")
        }
        # Retrieval time does not change identity; the pinned bytes do.
        data_identity = {
            name: {k: item[k] for k in ("bytes", "sha256")} for name, item in self.manifest["files"].items()
        }
        identity = {
            "experiment": self.experiment.as_dict(),
            "source": source,
            "dependencies": dependencies,
            "dataset": data_identity,
            "round": self.manifest["round"],
        }
        self.fingerprint = digest(identity)
        plan_path = self.output / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text())["fingerprint"] != self.fingerprint:
                raise ValueError(
                    "run identity changed; use a fresh output directory, never overwrite evidence"
                )
        else:
            atomic_json(
                plan_path,
                {
                    **identity,
                    "fingerprint": self.fingerprint,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "holdout_status": self.experiment.historical_evaluation_status,
                },
            )
        event("run_ready", fingerprint=self.fingerprint, output=str(self.output))

    def _selection(self):
        path = self.output / "selection.json"
        selection = json.loads(path.read_text())
        if selection["fingerprint"] != self.fingerprint:
            raise ValueError("selection belongs to a different experiment")
        stored = selection.pop("selection_sha256")
        if file_hash(self.output / "calibration_trials.json") != selection["trial_log_sha256"]:
            raise ValueError("calibration trial log was modified")
        if digest(selection) != stored:
            raise ValueError("locked selection was modified")
        return {**selection, "selection_sha256": stored}

    def _cv_names(self):
        return [spec.name for spec in self.experiment.models]

    def shortlist(self):
        result = json.loads((self.output / "cv_summary.json").read_text())
        eligible = [name for name, metrics in result.items() if metrics["ncorr"]["eligible"]]
        risk = self.experiment.selection_risk_penalty
        eligible.sort(
            key=lambda name: result[name]["ncorr"]["mean"] - risk * result[name]["ncorr"]["std"], reverse=True
        )
        if not eligible:
            raise ValueError("no eligible CV candidates")
        return eligible[: self.experiment.max_ensemble_models]

    def names(self, stage):
        if stage == "cv":
            return self._cv_names()
        if stage == "calibration":
            return list(dict.fromkeys([*self.shortlist(), self._cv_names()[0]]))
        if stage == "holdout":
            return list(dict.fromkeys([*self._selection()["weights"], self._cv_names()[0]]))
        raise ValueError("unknown research stage")

    def windows(self, stage):
        c = self.experiment
        if stage == "cv":
            return [(f"fold{i + 1}", left, right) for i, (left, right) in enumerate(c.folds)]
        if stage == "calibration":
            return [("calibration", c.calibration_start, c.calibration_end)]
        if stage == "holdout":
            return [("holdout", c.holdout_start, "2100-01-01")]
        raise ValueError("unknown research stage")

    def frame(self, stage):
        if stage == "holdout":
            self.open_holdout()
        # Do not even load holdout labels before the selection is frozen.
        max_date = self.windows(stage)[-1][2]
        frame, sets = load_training_frame(self.data, self.experiment, max_date=max_date)
        return frame, sets

    def _checkpoint(self, stage, window, name):
        return self.output / stage / window / name

    def _check_checkpoint(self, base):
        meta_path = base.with_suffix(".json")
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text())
        if meta["fingerprint"] != self.fingerprint:
            raise ValueError("model checkpoint identity changed")
        for suffix in (".parquet", ".pkl"):
            path = base.with_suffix(suffix)
            if not path.is_file() or file_hash(path) != meta["artifacts"][suffix]:
                raise ValueError(f"model checkpoint is missing or modified: {path}")
        for checkpoint in meta.get("refit_checkpoints", []):
            if not self._check_checkpoint(self.output / checkpoint):
                raise ValueError("missing refit checkpoint")
        return True

    def train(self, stage):
        if stage == "holdout":
            self.open_holdout()
        names = self.names(stage)
        frame, sets = self.frame(stage)
        event(
            "data_loaded",
            stage=stage,
            rows=len(frame),
            dates=int(frame.date.nunique()),
            memory_gib=round(frame.memory_usage(deep=True).sum() / 2**30, 3),
        )
        for window, left, right in self.windows(stage):
            evaluation = frame.loc[frame.date.between(left, right)]
            if evaluation.empty:
                raise ValueError(f"no mature observations in {window}")
            for spec in self.experiment.models:
                if spec.name not in names:
                    continue
                base = self._checkpoint(stage, window, spec.name)
                base.parent.mkdir(parents=True, exist_ok=True)
                if self._check_checkpoint(base):
                    event("checkpoint_reused", stage=stage, window=window, model=spec.name)
                    continue
                chunks = refit_windows(evaluation.date, spec.refit_every_dates)
                parts, refits, checkpoints = [], [], []
                for i, (chunk_left, chunk_right) in enumerate(chunks):
                    chunk = evaluation.loc[evaluation.date.between(chunk_left, chunk_right)]
                    refit_base = base.parent / base.name / f"refit{i + 1:03d}"
                    refit_base.parent.mkdir(parents=True, exist_ok=True)
                    if self._check_checkpoint(refit_base):
                        refit_meta = json.loads(refit_base.with_suffix(".json").read_text())
                        parts.append(pd.read_parquet(refit_base.with_suffix(".parquet")))
                        refits.append(refit_meta)
                        checkpoints.append(str(refit_base.relative_to(self.output)))
                        event("refit_reused", stage=stage, window=window, model=spec.name, refit=i + 1)
                        continue
                    event("fit_start", stage=stage, window=window, model=spec.name, refit=i + 1)
                    started = time.monotonic()
                    available = frame.loc[
                        prior_training_mask(frame.date, chunk_left, self.experiment.purge_business_days)
                    ]
                    sampled = sample_training(available, self.experiment.max_train_rows_per_date, spec.seed)
                    del available
                    fitted = fit_model(sampled, sets[spec.feature_set], spec, self.experiment)
                    prediction = chunk[KEYS].copy()
                    prediction["prediction"] = fitted.predict(chunk)
                    part = refit_base.with_suffix(".parquet.tmp")
                    prediction.to_parquet(part, index=False)
                    part.replace(refit_base.with_suffix(".parquet"))
                    part = refit_base.with_suffix(".pkl.tmp")
                    with part.open("wb") as handle:
                        pickle.dump(fitted, handle, protocol=5)
                    part.replace(refit_base.with_suffix(".pkl"))
                    refit_meta = {
                        "fingerprint": self.fingerprint,
                        "spec": asdict(spec),
                        **fitted.metadata,
                        "evaluation_first_date": str(chunk.date.min().date()),
                        "evaluation_last_date": str(chunk.date.max().date()),
                        "evaluation_rows": len(chunk),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "artifacts": {s: file_hash(refit_base.with_suffix(s)) for s in (".parquet", ".pkl")},
                    }
                    atomic_json(refit_base.with_suffix(".json"), refit_meta)
                    event(
                        "fit_complete",
                        stage=stage,
                        window=window,
                        model=spec.name,
                        refit=i + 1,
                        **fitted.metadata,
                    )
                    parts.append(prediction)
                    refits.append(refit_meta)
                    checkpoints.append(str(refit_base.relative_to(self.output)))
                    del fitted, sampled
                prediction = pd.concat(parts, ignore_index=True)
                validate_keys(prediction)
                if len(prediction) != len(evaluation):
                    raise ValueError("refits did not cover the complete evaluation window")
                part = base.with_suffix(".parquet.tmp")
                prediction.to_parquet(part, index=False)
                part.replace(base.with_suffix(".parquet"))
                # The historical bundle caches only the last refit; predictions and
                # all per-refit checkpoints retain the full walk-forward evidence.
                import shutil

                shutil.copyfile(refit_base.with_suffix(".pkl"), base.with_suffix(".pkl"))
                meta = {
                    **refits[0],
                    "evaluation_last_date": str(evaluation.date.max().date()),
                    "evaluation_rows": len(evaluation),
                    "fit_seconds": sum(item["fit_seconds"] for item in refits),
                    "refit_count": len(refits),
                    "refits": refits,
                    "refit_checkpoints": checkpoints,
                    "historical_pickle_scope": "last refit only; not a deployment bundle",
                    "artifacts": {s: file_hash(base.with_suffix(s)) for s in (".parquet", ".pkl")},
                }
                atomic_json(base.with_suffix(".json"), meta)
                del parts, prediction
        return names

    def contexts(self, evaluation):
        dates = evaluation.date.drop_duplicates().sort_values()
        neutral = load_date_auxiliary(self.data, "validation_neutralizer.parquet", dates)
        bench = load_date_auxiliary(
            self.data, "validation_benchmark_models.parquet", dates, [*KEYS, self.experiment.benchmark]
        )
        cols = [c for c in neutral if c.startswith("neutralizer_column_")]
        if not cols:
            raise ValueError("no supplied neutralizer columns")
        aux = neutral[[*KEYS, *cols]].merge(bench, on=KEYS, how="inner", validate="one_to_one")
        del neutral, bench
        for date, group in evaluation.groupby("date", sort=True):
            right = aux.loc[aux.date == date]
            joined = group[KEYS].merge(right, on=KEYS, how="left", validate="one_to_one", sort=False)
            scorer = DateScorer.prepare(
                joined[cols].to_numpy(),
                group[self.experiment.primary_target].to_numpy(),
                joined[self.experiment.benchmark].to_numpy(),
            )
            yield date, group.index.to_numpy(), scorer

    def score(self, stage):
        frame, _ = self.frame(stage)
        names = self.names(stage)
        rows = []
        with threadpool_limits(limits=self.experiment.threads):
            for window, left, right in self.windows(stage):
                evaluation = frame.loc[frame.date.between(left, right)].reset_index(drop=True)
                matrix = []
                for name in names:
                    base = self._checkpoint(stage, window, name)
                    if not self._check_checkpoint(base):
                        raise ValueError(f"missing checkpoint: {base}")
                    matrix.append(aligned_predictions(evaluation, base.with_suffix(".parquet")))
                matrix = np.column_stack(matrix)
                selection = self._selection() if stage == "holdout" else None
                if selection:
                    final = self.blend(
                        evaluation, matrix, names, selection["weights"], selection["neutralization"]
                    )
                    matrix = np.column_stack([matrix, final])
                    score_names = [*names, "selected_ensemble"]
                else:
                    score_names = names
                for date, indexes, scorer in self.contexts(evaluation):
                    scores = scorer.score(matrix[indexes])
                    for i, name in enumerate(score_names):
                        rows.append(
                            {
                                "date": str(date.date()),
                                "window": window,
                                "model": name,
                                "ncorr": float(scores["ncorr"][i]),
                                "benchmark_contribution": float(scores["benchmark_contribution"][i]),
                            }
                        )
                event("window_scored", stage=stage, window=window)
        scores = pd.DataFrame(rows)
        scores.to_csv(self.output / f"{stage}_scores.csv", index=False)
        result = {
            name: {
                metric: summary(g[metric], block_size=self.experiment.bootstrap_block_dates)
                for metric in ("ncorr", "benchmark_contribution")
            }
            for name, g in scores.groupby("model", sort=True)
        }
        atomic_json(self.output / f"{stage}_summary.json", result)
        return scores, result

    def blend(self, frame, matrix, names, weights, proportion):
        if set(weights) - set(names) or not weights or any(w < 0 for w in weights.values()):
            raise ValueError("invalid blend weights")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("empty ensemble")
        result = np.empty(len(frame))
        sets = feature_sets(self.data, ["medium"])
        for _, group in frame.groupby("date", sort=True):
            indexes = group.index.to_numpy()
            ranks = percentile(matrix[indexes])
            blend = sum(ranks[:, names.index(name)] * weight / total for name, weight in weights.items())
            result[indexes] = neutralize_features(blend, group[sets["medium"]].to_numpy(), proportion)
        return result

    def select(self):
        existing = self.output / "selection.json"
        if existing.exists():
            return self._selection()
        frame, _ = self.frame("calibration")
        c = self.experiment
        evaluation = frame.loc[frame.date.between(c.calibration_start, c.calibration_end)].reset_index(
            drop=True
        )
        del frame
        names = self.names("calibration")
        candidates = self.shortlist()
        for name in names:
            if not self._check_checkpoint(self._checkpoint("calibration", "calibration", name)):
                raise ValueError(f"missing calibration checkpoint: {name}")
        matrix = np.column_stack(
            [
                aligned_predictions(
                    evaluation, self._checkpoint("calibration", "calibration", name).with_suffix(".parquet")
                )
                for name in names
            ]
        )
        trials = []
        with threadpool_limits(limits=c.threads):
            contexts = list(self.contexts(evaluation))

            def objective(weights, proportion=0.0):
                blend = self.blend(evaluation, matrix, names, weights, proportion)
                scores = np.array([scorer.score(blend[idx])["ncorr"][0] for _, idx, scorer in contexts])
                if not np.isfinite(scores).all():
                    return -1e100
                value = float(scores.mean() - c.selection_risk_penalty * scores.std())
                trials.append(
                    {
                        "trial": len(trials) + 1,
                        "weights": dict(weights),
                        "neutralization": proportion,
                        "objective": value,
                        "mean_ncorr": float(scores.mean()),
                        "std_ncorr": float(scores.std()),
                    }
                )
                return value

            counts = {}
            best_weights, best_value = None, -1e100
            for _ in range(c.ensemble_steps):
                options = []
                for name in candidates:
                    proposed = dict(counts)
                    proposed[name] = proposed.get(name, 0) + 1
                    options.append((objective(proposed), name, proposed))
                value, _, counts = max(options, key=lambda x: x[0])
                if value > best_value:
                    best_value, best_weights = value, dict(counts)
            equal = {name: 1 for name in candidates}
            equal_value = objective(equal)
            if equal_value > best_value:
                best_weights, best_value = equal, equal_value
            if best_weights is None:
                raise ValueError("no valid calibration ensemble")
            options = [(objective(best_weights, p), p) for p in c.neutralization_proportions]
            _, proportion = max(options, key=lambda x: x[0])
        atomic_json(self.output / "calibration_trials.json", trials)
        total = sum(best_weights.values())
        selected = {
            "fingerprint": self.fingerprint,
            "weights": {k: v / total for k, v in best_weights.items()},
            "neutralization": proportion,
            "calibration_trials": len(trials),
            "shortlist": candidates,
            "objective": "mean date NCORR minus 0.25 times standard deviation",
            "trial_log_sha256": file_hash(self.output / "calibration_trials.json"),
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        selected["selection_sha256"] = digest(selected)
        atomic_json(existing, selected)
        event("selection_frozen", weights=selected["weights"], neutralization=proportion)
        return selected

    def open_holdout(self):
        selection = self._selection()
        path = self.output / "holdout_access.json"
        if path.exists():
            if json.loads(path.read_text())["selection_sha256"] != selection["selection_sha256"]:
                raise ValueError("holdout already opened with another selection")
        else:
            atomic_json(
                path,
                {
                    "selection_sha256": selection["selection_sha256"],
                    "opened_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluation_status": self.experiment.historical_evaluation_status,
                    "warning": "historical holdout now consumed; future changes need fresh evidence",
                },
            )

    def finalize(self):
        path = self.output / "result.json"
        if path.exists():
            event("completed_result_reused", path=str(path))
            return json.loads(path.read_text())
        self.open_holdout()
        self.train("holdout")
        scores, result = self.score("holdout")
        paired = scores.pivot(index="date", columns="model", values="ncorr")
        baseline = self._cv_names()[0]
        delta = paired.selected_ensemble - paired[baseline]
        result = {
            "fingerprint": self.fingerprint,
            "selection": self._selection(),
            "holdout": result,
            "paired_ncorr_gain_vs_baseline": summary(delta, block_size=self.experiment.bootstrap_block_dates),
            "baseline": baseline,
            "historical_holdout_consumed": True,
            "evaluation_status": self.experiment.historical_evaluation_status,
            "prior_experiment_fingerprint": self.experiment.prior_experiment_fingerprint,
            "caveats": [
                "Historical data, not a prospective or live result.",
                "NCORR and benchmark contribution are scores, not returns or profit.",
                "All trials in this run are logged; no claim of global optimality.",
                "Block bootstrap is approximate with overlapping 60-day labels.",
                "No trading, staking, or submission was performed.",
            ],
        }
        if self.experiment.historical_evaluation_status == "reused_exploratory":
            result["caveats"].insert(
                0,
                "This historical period was inspected in a prior experiment; this follow-up is exploratory.",
            )
        atomic_json(path, result)
        event("research_complete", path=str(path))
        return result

    def run(self):
        if (self.output / "result.json").exists():
            return self.finalize()
        self.train("cv")
        self.score("cv")
        self.train("calibration")
        self.score("calibration")
        self.select()
        return self.finalize()
