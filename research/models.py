"""CPU models with training-only preprocessing and purged inner stopping."""

from __future__ import annotations

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.validation import inner_split, training_weights


@dataclass
class FittedModel:
    estimator: object
    features: list[str]
    metadata: dict

    def predict(self, frame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"prediction features missing: {sorted(missing)}")
        x = frame[self.features].to_numpy(dtype=np.float32, copy=True)
        x[~np.isfinite(x)] = np.nan
        with threadpool_limits(limits=self.metadata["threads"]):
            result = self.estimator.predict(x)
        if not np.isfinite(result).all():
            raise ValueError("model emitted nonfinite predictions")
        return np.asarray(result, dtype=float)


def fit_model(frame, features, spec, experiment):
    """The caller supplies only chronologically eligible sampled training rows."""
    started = time.monotonic()
    frame = frame.dropna(subset=[spec.target])
    if frame.empty:
        raise ValueError(f"no mature labels for {spec.name}")
    x = frame[features].to_numpy(dtype=np.float32, copy=True)
    x[~np.isfinite(x)] = np.nan
    y = frame[spec.target].to_numpy(dtype=np.float32)
    weights = training_weights(frame.date, spec.equal_date_weight, spec.half_life_weeks)
    meta = {
        "name": spec.name,
        "kind": spec.kind,
        "target": spec.target,
        "rows": len(frame),
        "dates": int(frame.date.nunique()),
        "first_date": str(frame.date.min().date()),
        "last_date": str(frame.date.max().date()),
        "features": len(features),
        "threads": experiment.threads,
        "seed": spec.seed,
    }
    with threadpool_limits(limits=experiment.threads):
        if spec.kind == "ridge":
            estimator = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True, copy=False),
                StandardScaler(copy=False),
                Ridge(**spec.params),
            )
            estimator.fit(x, y, ridge__sample_weight=weights)
        elif spec.kind == "lightgbm":
            params = dict(
                objective="regression",
                verbosity=-1,
                n_jobs=experiment.threads,
                random_state=spec.seed,
                deterministic=True,
                force_col_wise=True,
                max_bin=31,
            )
            params.update(spec.params)
            # Thread and reproducibility limits cannot be overridden by model parameters.
            params.update(
                n_jobs=experiment.threads,
                random_state=spec.seed,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            max_trees = int(params.pop("n_estimators", 1200))
            trees = max_trees
            if spec.early_stopping:
                fit, stop = inner_split(
                    frame.date, experiment.early_stopping_dates, experiment.purge_business_days
                )
                stopper = lgb.LGBMRegressor(**params, n_estimators=max_trees)
                stopper.fit(
                    x[fit],
                    y[fit],
                    sample_weight=weights[fit],
                    eval_set=[(x[stop], y[stop])],
                    eval_sample_weight=[weights[stop]],
                    callbacks=[lgb.early_stopping(experiment.early_stopping_rounds, verbose=False)],
                )
                trees = max(1, int(stopper.best_iteration_ or max_trees))
                meta.update(
                    inner_fit_last_date=str(frame.loc[fit, "date"].max().date()),
                    inner_stop_first_date=str(frame.loc[stop, "date"].min().date()),
                )
                del stopper
            meta.update(
                selected_trees=trees,
                max_trees=max_trees,
                tree_selection="purged_inner_rmse" if spec.early_stopping else "predeclared_fixed",
            )
            estimator = lgb.LGBMRegressor(**params, n_estimators=trees)
            estimator.fit(x, y, sample_weight=weights)
            if spec.era_boost_steps:
                # Explicit in-sample reweighting ablation; only outer scores establish value.
                additions = max(1, trees // 4)
                for _ in range(spec.era_boost_steps):
                    pred = estimator.predict(x)
                    scores = pd.DataFrame({"date": frame.date.to_numpy(), "p": pred, "y": y})
                    correlations = scores.groupby("date").apply(
                        lambda g: g.p.corr(g.y, method="spearman"), include_groups=False
                    )
                    hard = set(correlations.nsmallest(max(1, len(correlations) // 2)).index)
                    boosted = weights * np.where(frame.date.isin(hard), 2.0, 1.0)
                    booster = estimator.booster_
                    estimator = lgb.LGBMRegressor(**params, n_estimators=additions)
                    estimator.fit(x, y, sample_weight=boosted, init_model=booster)
                meta["era_boost_steps"] = spec.era_boost_steps
                meta["total_trees"] = estimator.booster_.num_trees()
        else:
            raise ValueError(f"unsupported model family: {spec.kind}")
    meta["fit_seconds"] = round(time.monotonic() - started, 3)
    return FittedModel(estimator, list(features), meta)
