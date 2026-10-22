import numpy as np
import pandas as pd
import pytest
from numerai_tools.scoring import neutral_contribution, neutral_correlation

from research.scoring import DateScorer, neutralize_features, summary


@pytest.mark.parametrize("dependent", [False, True])
def test_fast_scores_match_official_v6(dependent):
    rng = np.random.default_rng(65)
    n = 350
    x = rng.normal(size=(n, 12))
    if dependent:
        x = np.column_stack([x, x[:, 0], x[:, 1] * 2, np.ones(n)])
    y = rng.choice([0, 0.25, 0.5, 0.75, 1], size=n)
    m = rng.normal(size=n)
    p = np.column_stack([rng.normal(size=n), rng.integers(0, 5, size=n)])
    got = DateScorer.prepare(x, y, m).score(p)
    frame, targets, benchmark, neutralizers = (pd.DataFrame(p), pd.Series(y), pd.Series(m), pd.DataFrame(x))
    expected_corr = neutral_correlation(frame, neutralizers, targets)
    expected_contribution = neutral_contribution(frame, benchmark, neutralizers, targets)
    np.testing.assert_allclose(got["ncorr"], np.asarray(expected_corr).ravel(), atol=1e-10)
    np.testing.assert_allclose(
        got["benchmark_contribution"], np.asarray(expected_contribution).ravel(), atol=1e-10
    )


def test_constant_or_missing_predictions_are_ineligible():
    rng = np.random.default_rng(42)
    scorer = DateScorer.prepare(rng.normal(size=(100, 3)), rng.uniform(size=100), rng.normal(size=100))
    assert np.isnan(scorer.score(np.ones(100))["ncorr"]).all()
    with pytest.raises(ValueError):
        scorer.score(np.full(100, np.nan))
    assert not summary([0.1, np.nan])["eligible"]


def test_neutralization_is_label_free_and_handles_empty_features():
    p = np.linspace(0, 1, 100)
    x = np.column_stack([p, np.ones(100), np.full(100, np.nan)])
    reduced = neutralize_features(p, x, 0.5)
    assert np.isfinite(reduced).all()
    assert len(np.unique(reduced)) > 1
    np.testing.assert_allclose(neutralize_features(p, x, 0), (np.arange(100) + 0.5) / 100)
