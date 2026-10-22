"""Run the real training orchestration with fake models and synthetic data."""

from contextlib import ExitStack
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import solstice


class PipelineIsolationTests(unittest.TestCase):
    def test_training_routes_labels_to_separate_roles_and_saves_split(self):
        frame = pd.DataFrame({"era": np.arange(1, 301), "feature": np.arange(300),
                              solstice.config.PRIMARY_TARGET: np.arange(300) / 300})
        model = Mock()
        model.state_dict.return_value = {}
        transformer = Mock(return_value=(model, [0], [1]))
        calibration = Mock(return_value={"component": 1.0})
        evaluation = Mock()
        with TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.object(solstice.config, "MODEL_DIR", Path(directory)))
            stack.enter_context(patch("sys.argv", ["solstice.py", "--train"]))
            stack.enter_context(patch.dict("sys.modules", {"torch": types.SimpleNamespace(save=Mock())}))
            for name in ["download_data", "log"]:
                stack.enter_context(patch.object(solstice, name))
            stack.enter_context(patch.object(solstice, "load_features", return_value=["feature"]))
            stack.enter_context(patch.object(solstice, "load_data", return_value=(frame, [])))
            for name in ["train_benchmark_era_boosted", "train_multi_target", "train_residual", "train_catboost"]:
                stack.enter_context(patch.object(solstice, name, return_value=[]))
            stack.enter_context(patch.object(solstice, "train_ft_transformer", transformer))
            stack.enter_context(patch.object(solstice, "predict_all_components", return_value={"component": np.arange(300)}))
            stack.enter_context(patch.object(solstice, "calibrate_weights", calibration))
            stack.enter_context(patch.object(solstice, "evaluate_pipeline", evaluation))
            solstice.main()
            stopping_eras = set(transformer.call_args.args[1]["era"])
            calibration_eras = set(calibration.call_args.args[2])
            evaluation_eras = set(evaluation.call_args.args[-1])
            self.assertLess(max(stopping_eras), min(calibration_eras))
            self.assertLess(max(calibration_eras), min(evaluation_eras))
            self.assertEqual(len(calibration_eras), 95)
            self.assertEqual(len(evaluation_eras), 95)
            with (Path(directory) / "solstice_v4.pkl").open("rb") as handle:
                checkpoint = pickle.load(handle)
            self.assertEqual(set(checkpoint["validation_split"]["evaluation"]), evaluation_eras)
            self.assertEqual(checkpoint["data_version"], solstice.config.DATA_VERSION)

    def test_evaluation_refuses_legacy_checkpoint(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            (Path(directory) / "solstice_v4.pkl").write_bytes(pickle.dumps({}))
            stack.enter_context(patch.object(solstice.config, "MODEL_DIR", Path(directory)))
            stack.enter_context(patch("sys.argv", ["solstice.py", "--evaluate"]))
            stack.enter_context(patch.object(solstice, "download_data"))
            stack.enter_context(patch.object(solstice, "load_features", return_value=[]))
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                solstice.main()
