"""Finite experiment plans, validated before labels or scores are inspected."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str = "lightgbm"
    target: str = "target_jupiter_60"
    feature_set: str = "medium"
    seed: int = 42
    equal_date_weight: bool = True
    half_life_weeks: float | None = None
    era_boost_steps: int = 0
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment:
    name: str = "mac-v6"
    version: str = "v6.0"
    primary_target: str = "target_jupiter_60"
    benchmark: str = "lgbm_jupiter_60"
    purge_business_days: int = 90
    max_train_rows_per_date: int = 1500
    threads: int = 6
    early_stopping_dates: int = 26
    early_stopping_rounds: int = 75
    calibration_start: str = "2024-01-05"
    calibration_end: str = "2024-12-27"
    holdout_start: str = "2025-06-06"
    folds: tuple = (("2019-06-07", "2020-06-05"), ("2021-06-04", "2022-06-03"), ("2022-08-05", "2023-07-28"))
    max_ensemble_models: int = 5
    ensemble_steps: int = 12
    selection_risk_penalty: float = 0.25
    neutralization_proportions: tuple = (0.0, 0.25, 0.5, 0.75)
    bootstrap_block_dates: int = 16
    models: tuple[ModelSpec, ...] = ()

    def validate(self):
        if self.version != "v6.0":
            raise ValueError("the modern pipeline explicitly supports v6.0")
        horizon = re.search(r"_(\d+)$", self.primary_target)
        targets = [self.primary_target, *(m.target for m in self.models)]
        horizons = [int(re.search(r"_(\d+)$", t).group(1)) for t in targets if re.search(r"_(\d+)$", t)]
        if horizon is None or len(horizons) != len(targets) or self.purge_business_days < max(horizons) + 2:
            raise ValueError("purge must cover every explicit target horizon plus the two-day lag")
        if (
            min(
                self.threads,
                self.max_train_rows_per_date,
                self.early_stopping_dates,
                self.max_ensemble_models,
                self.ensemble_steps,
                self.bootstrap_block_dates,
            )
            < 1
        ):
            raise ValueError("resource and search limits must be positive")
        if not self.models or len({m.name for m in self.models}) != len(self.models):
            raise ValueError("provide distinct named model candidates")
        if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", m.name) for m in self.models):
            raise ValueError("candidate names must be safe file names")
        if any(m.kind not in {"lightgbm", "ridge"} for m in self.models):
            raise ValueError("unsupported model family")
        if any(m.half_life_weeks is not None and m.half_life_weeks <= 0 for m in self.models):
            raise ValueError("recency half life must be positive")
        if any(not 0 <= float(p) <= 1 for p in self.neutralization_proportions):
            raise ValueError("neutralization strengths must be in [0, 1]")
        import pandas as pd

        start, end, hold = map(
            pd.Timestamp, (self.calibration_start, self.calibration_end, self.holdout_start)
        )
        if not start <= end or end + pd.offsets.BDay(self.purge_business_days) >= hold:
            raise ValueError("calibration outcomes must mature before holdout begins")
        previous = None
        for left, right in self.folds:
            left, right = pd.Timestamp(left), pd.Timestamp(right)
            if left > right or right + pd.offsets.BDay(self.purge_business_days) >= start:
                raise ValueError("CV outcomes must mature before calibration begins")
            if previous is not None and left <= previous:
                raise ValueError("CV evaluation windows must be ordered and disjoint")
            previous = right
        return self

    def as_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def load_experiment(path: Path) -> Experiment:
    raw = tomllib.loads(path.read_text())
    models = tuple(ModelSpec(**m) for m in raw.pop("models"))
    for key in ("folds", "neutralization_proportions"):
        if key in raw:
            raw[key] = tuple(tuple(x) if isinstance(x, list) else x for x in raw[key])
    return Experiment(models=models, **raw).validate()
