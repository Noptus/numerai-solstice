"""Research isolation checks using artificial era labels only."""

import unittest

from validation import chronological_split


class ValidationTests(unittest.TestCase):
    def test_chronology_disjointness_and_embargo(self):
        split = chronological_split(list(range(1, 301)) * 2)
        groups = [set(split.early_stopping), set(split.calibration), set(split.evaluation), set(split.embargo)]
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                self.assertFalse(group & other)
        self.assertEqual(len(split.calibration), 95)
        self.assertEqual(len(split.evaluation), 95)
        self.assertEqual(min(split.calibration) - max(split.early_stopping), 5)
        self.assertEqual(min(split.evaluation) - max(split.calibration), 5)
        self.assertEqual(set.union(*groups), set(range(1, 301)))

    def test_numeric_order_of_unpadded_eras(self):
        split = chronological_split([f"era{i}" for i in range(30, 0, -1)],
                                    evaluation_eras=5, calibration_eras=5,
                                    embargo_eras=2, min_early_stopping_eras=5)
        self.assertEqual(split.evaluation, tuple(f"era{i}" for i in range(26, 31)))

    def test_not_enough_history_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "need at least"):
            chronological_split(range(100))

    def test_ambiguous_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            chronological_split(["1", "01"], evaluation_eras=1,
                                calibration_eras=1, embargo_eras=0,
                                min_early_stopping_eras=1)

    def test_newer_evaluation_eras_do_not_enter_calibration(self):
        split = chronological_split(range(400))
        self.assertLess(max(split.calibration), min(split.evaluation))
        self.assertLess(max(split.early_stopping), min(split.calibration))

    def test_invalid_configuration(self):
        for kwargs in [dict(evaluation_eras=0), dict(embargo_eras=-1), dict(calibration_eras=True)]:
            with self.assertRaises(ValueError):
                chronological_split(range(400), **kwargs)


if __name__ == "__main__":
    unittest.main()
