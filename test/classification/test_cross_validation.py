import unittest

import numpy as np

from src.classification.evaluation.cross_validation import (
    InsufficientDataError,
    iter_participant_folds,
    resolve_n_splits,
    sample_mask_for_participants,
)


class TestCrossValidation(unittest.TestCase):
    def test_resolve_caps_at_min_class_count(self):
        labels = np.array([0, 0, 1, 1, 1, 1])
        self.assertEqual(resolve_n_splits(labels, desired=5), 2)

    def test_resolve_requires_both_classes(self):
        with self.assertRaises(InsufficientDataError):
            resolve_n_splits(np.array([1, 1, 1]), desired=2)

    def test_no_participant_leakage_across_folds(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        for train_idx, test_idx in iter_participant_folds(
            labels, n_splits=2, n_repeats=3, random_state=0
        ):
            self.assertEqual(set(train_idx).intersection(set(test_idx)), set())
            self.assertEqual(len(train_idx) + len(test_idx), len(labels))

    def test_sample_mask_for_participants(self):
        groups = np.array(["a", "a", "b", "c"], dtype=object)
        mask = sample_mask_for_participants(groups, ["a", "c"])
        np.testing.assert_array_equal(mask, [True, True, False, True])


if __name__ == "__main__":
    unittest.main()
