import unittest

import numpy as np
import pandas as pd

from eda.eda_nf_unsw_nb15_v3 import (
    build_split_drift_table,
    choose_bucket,
    jensen_shannon_divergence,
)


class TemporalEdaHelpersTest(unittest.TestCase):
    def test_choose_bucket_uses_session_span(self) -> None:
        self.assertEqual(choose_bucket(12 * 60 * 60), "1min")
        self.assertEqual(choose_bucket(2 * 24 * 60 * 60), "5min")

    def test_jensen_shannon_bounds_and_symmetry(self) -> None:
        left = np.array([9, 1, 0])
        right = np.array([5, 3, 2])
        divergence = jensen_shannon_divergence(left, right)
        self.assertGreater(divergence, 0)
        self.assertLess(divergence, 1)
        self.assertAlmostEqual(
            divergence, jensen_shannon_divergence(right, left)
        )
        self.assertAlmostEqual(jensen_shannon_divergence(left, left), 0)
        self.assertAlmostEqual(
            jensen_shannon_divergence([1, 0], [0, 1]), 1
        )

    def test_split_drift_separates_prevalence_from_attack_mix(self) -> None:
        counts = pd.DataFrame(
            [
                (split, attack, count)
                for split, values in {
                    "train": {"Benign": 90, "Exploit": 9, "Worm": 1},
                    "validation": {"Benign": 80, "Exploit": 18, "Worm": 2},
                    "test": {"Benign": 70, "Exploit": 27, "Worm": 3},
                }.items()
                for attack, count in values.items()
            ],
            columns=["split", "Attack", "flow_count"],
        )

        drift = build_split_drift_table(counts)
        train_test = drift.loc[
            (drift["left_split"] == "train")
            & (drift["right_split"] == "test")
        ].set_index("scope")

        self.assertGreater(
            train_test.loc["all_traffic", "jensen_shannon_divergence"], 0
        )
        self.assertAlmostEqual(
            train_test.loc["attacks_only", "jensen_shannon_divergence"], 0
        )
        self.assertEqual(
            train_test.loc["all_traffic", "largest_shift_class"], "Benign"
        )


if __name__ == "__main__":
    unittest.main()
