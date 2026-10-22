"""Official v6 metrics plus a checked reusable neutralizer factorization.

The fast path uses the same 1e-6 relative singular-value cutoff as
numerai-tools 0.7.2. Tests compare NCORR and fixed-benchmark contribution
against that package, including dependent and constant neutralizer columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd
from scipy.special import ndtri
from scipy.stats import rankdata


def percentile(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("nonempty finite predictions required")
    return (rankdata(values, axis=0, method="average") - 0.5) / len(values)


def gaussian(values):
    return ndtri(percentile(values))


@dataclass
class DateScorer:
    basis: np.ndarray
    target: np.ndarray
    benchmark: np.ndarray

    @classmethod
    def prepare(cls, neutralizers, targets, benchmark):
        x = np.asarray(neutralizers, dtype=float)
        y, m = np.asarray(targets, dtype=float), np.asarray(benchmark, dtype=float)
        if x.ndim != 2 or y.ndim != 1 or m.ndim != 1 or not len(y) or len(x) != len(y) or len(m) != len(y):
            raise ValueError("misaligned scorer arrays")
        if not all(np.isfinite(a).all() for a in (x, y, m)):
            raise ValueError("missing or nonfinite scoring inputs")
        if ((y < 0) | (y > 1)).any():
            raise ValueError("v6 targets must be bounded [0,1]")
        if np.ptp(y) == 0 or np.ptp(m) == 0:
            raise ValueError("constant target or benchmark")
        x = np.column_stack([x, np.ones(len(x))])
        u, singular, _ = svd(x, full_matrices=False, check_finite=False)
        basis = u[:, singular > singular[0] * 1e-6]
        del u
        centered = y - y.mean()
        model = gaussian(m)[:, 0]
        model -= basis @ (basis.T @ model)
        if model @ model < 1e-20:
            raise ValueError("benchmark has no residual variance after neutralization")
        return cls(basis, centered, model)

    def score(self, predictions):
        p = gaussian(predictions)
        if len(p) != len(self.target):
            raise ValueError("prediction universe differs from scorer universe")
        p -= self.basis @ (self.basis.T @ p)
        denom = np.linalg.norm(p, axis=0) * np.linalg.norm(self.target)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (self.target @ p) / denom
        orthogonal = p - np.outer(self.benchmark, (self.benchmark @ p) / (self.benchmark @ self.benchmark))
        # Public v6 targets are bounded [0,1]; official contribution maps to [-2,2].
        contribution = (4 * self.target @ orthogonal) / len(self.target)
        invalid = np.linalg.norm(p, axis=0) < 1e-10
        corr[invalid] = np.nan
        contribution[invalid] = np.nan
        return {"ncorr": corr, "benchmark_contribution": contribution}


def summary(values, *, block_size=16, seed=42, bootstrap_samples=2000):
    """Block-bootstrap uncertainty accounts for serial dependence, imperfectly."""
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    result = {"dates": len(x), "valid_dates": int(valid.sum()), "eligible": bool(valid.all())}
    if not valid.all() or len(x) < 2:
        return result
    mean, std = float(x.mean()), float(x.std())
    cumulative = np.r_[0, x.cumsum()]
    block = min(block_size, len(x))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), size=(bootstrap_samples, int(np.ceil(len(x) / block))))
    indexes = (starts[:, :, None] + np.arange(block)) % len(x)
    draws = x[indexes.reshape(bootstrap_samples, -1)[:, : len(x)]].mean(axis=1)
    result.update(
        mean=mean,
        std=std,
        mean_std=mean / std if std > 1e-12 else None,
        positive_fraction=float((x > 0).mean()),
        worst_date=float(x.min()),
        max_drawdown=float(np.max(np.maximum.accumulate(cumulative) - cumulative)),
        mean_ci95=[float(v) for v in np.quantile(draws, [0.025, 0.975])],
        bootstrap_block_dates=block,
        uncertainty_note="circular block bootstrap of historical date scores; not future performance",
    )
    return result


def neutralize_features(predictions, features, proportion, top_k=50):
    """Contemporaneous cross-sectional postprocessing, never using targets."""
    if proportion == 0:
        return percentile(predictions)[:, 0]
    if not 0 <= proportion <= 1:
        raise ValueError("neutralization proportion outside [0,1]")
    p = gaussian(predictions)[:, 0]
    x = np.asarray(features, dtype=float).copy()
    if len(x) != len(p) or x.ndim != 2:
        raise ValueError("misaligned feature neutralization inputs")
    finite = np.isfinite(x)
    means = np.divide(
        np.where(finite, x, 0).sum(axis=0),
        finite.sum(axis=0),
        out=np.zeros(x.shape[1]),
        where=finite.sum(axis=0) > 0,
    )
    x = np.where(np.isfinite(x), x, means)
    x -= x.mean(axis=0)
    norms = np.linalg.norm(x, axis=0)
    keep = norms > 1e-12
    x, norms = x[:, keep], norms[keep]
    if not x.shape[1]:
        return percentile(p)[:, 0]
    exposure = np.abs((x.T @ p) / norms)
    selected = np.argsort(exposure, kind="stable")[-min(top_k, len(exposure)) :]
    x = np.column_stack([x[:, selected], np.ones(len(x))])
    coef = np.linalg.lstsq(x, p, rcond=1e-6)[0]
    return percentile(p - proportion * (x @ coef))[:, 0]
