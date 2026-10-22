"""Check temporal isolation and failure behavior with small synthetic arrays."""

from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import solstice


class ResearchMetricTests(unittest.TestCase):
    def test_future_era_values_cannot_change_earlier_ensemble(self):
        eras = np.repeat([1, 2], 60)
        a = np.arange(120, dtype=float)
        b = np.sin(a)
        first = solstice.weighted_rank_ensemble({"a": a, "b": b}, {"a": .6, "b": .4}, eras)
        a[60:] = np.arange(60) * -1000
        second = solstice.weighted_rank_ensemble({"a": a, "b": b}, {"a": .6, "b": .4}, eras)
        np.testing.assert_array_equal(first[:60], second[:60])

    def test_reserved_evaluation_neutralizes_one_era_at_a_time(self):
        eras = np.repeat([1, 2, 3], 60)
        x = np.arange(180, dtype=float)
        frame = pd.DataFrame({"era": eras, "feature": x, solstice.config.PRIMARY_TARGET: np.sin(x)})
        def neutralize(preds, features, benchmarks, *args):
            self.assertEqual(len(features), 60)
            self.assertEqual(len(preds), 60)
            return preds
        with patch.object(solstice, "two_stage_neutralize", side_effect=neutralize) as method:
            metrics = solstice.evaluate_pipeline({"a": x}, frame, ["feature"], [], ["feature"], {"a": 1}, [2, 3])
        self.assertEqual(method.call_count, 8)
        self.assertEqual(metrics["n_eras"], 2)

    def test_empty_or_nonfinite_metrics_fail_closed(self):
        for preds, targets, eras in [([], [], []), ([np.nan], [1], [1]), ([1] * 60, list(range(60)), [1] * 60)]:
            with self.assertRaises(ValueError):
                solstice.per_era_metrics(preds, targets, eras)

    def test_zero_correlation_variance_does_not_report_a_huge_ratio(self):
        x = np.tile(np.arange(60), 2)
        with self.assertRaisesRegex(ValueError, "undefined"):
            solstice.per_era_metrics(x, x, np.repeat([1, 2], 60))

    def test_unavailable_dataset_fails_before_any_download(self):
        client = Mock()
        client.list_datasets.return_value = ["v6.0/features.json"]
        with TemporaryDirectory() as directory, patch.object(solstice.config, "DATA_DIR", Path(directory)), \
             patch.dict("sys.modules", {"numerapi": types.SimpleNamespace(NumerAPI=lambda: client)}):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                solstice.download_data(training=True)
        client.download_dataset.assert_not_called()

    def test_cached_evaluation_does_not_request_live_or_training_data(self):
        with TemporaryDirectory() as directory, patch.object(solstice.config, "DATA_DIR", Path(directory)):
            for name in ["features.json", "validation.parquet", "validation_benchmark_models.parquet"]:
                (Path(directory) / name).write_text("fixture")
            # Any accidental network client construction would fail.
            with patch.dict("sys.modules", {"numerapi": None}):
                solstice.download_data(evaluation=True)
