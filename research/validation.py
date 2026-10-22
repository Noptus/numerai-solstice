"""Date-aware purging and deterministic, label-independent training sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prior_training_mask(dates, evaluation_start, purge_business_days):
    dates = pd.to_datetime(dates)
    cutoff = pd.Timestamp(evaluation_start) - pd.offsets.BDay(purge_business_days)
    return np.asarray(dates < cutoff)


def inner_split(dates, stop_dates, purge_business_days):
    dates = pd.Series(pd.to_datetime(dates))
    unique = dates.drop_duplicates().sort_values()
    if len(unique) <= stop_dates:
        raise ValueError("not enough dates for an inner stopping block")
    start = unique.iloc[-stop_dates]
    fit = prior_training_mask(dates, start, purge_business_days)
    stop = np.asarray(dates >= start)
    if fit.sum() == 0:
        raise ValueError("purge leaves no inner training observations")
    return fit, stop


def sample_training(frame, rows_per_date, seed):
    """Select by keys, never targets, and retain observations across all dates."""
    keys = frame[["numerai_ticker", "date"]].astype(str)
    hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    # Mix all bits; XOR with the small seed alone barely changes the selected set.
    salt = np.uint64((int(seed) * 0x9E3779B97F4A7C15) % (1 << 64))
    order = np.argsort(hashed ^ salt, kind="stable")
    # Rank dates on a narrow Series; never materialize a sorted copy of all features.
    ordered_dates = pd.Series(frame["date"].to_numpy()[order])
    ranks = ordered_dates.groupby(ordered_dates, sort=False).cumcount().to_numpy()
    selected = order[ranks < rows_per_date]
    return frame.iloc[selected].sort_values(["date", "numerai_ticker"])


def training_weights(dates, equal_date=True, half_life_weeks=None):
    dates = pd.Series(pd.to_datetime(dates))
    weights = np.ones(len(dates), dtype=float)
    if equal_date:
        counts = dates.map(dates.value_counts()).to_numpy(dtype=float)
        weights /= counts
    if half_life_weeks is not None:
        age = (dates.max() - dates).dt.days.to_numpy() / 7
        weights *= np.exp2(-age / half_life_weeks)
    return weights / weights.mean()
