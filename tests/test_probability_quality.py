import unittest

import numpy as np

from main import probability_quality_metrics, select_weighting_by_validation


class ProbabilityQualityTests(unittest.TestCase):
    def test_baseline_uses_training_prevalence_not_test_prevalence(self):
        result = probability_quality_metrics(
            train_labels=np.array([1, 1, 1, 0, 0]),
            test_labels=np.array([0, 0, 1, 1]),
            probabilities=np.array([0.2, 0.2, 0.8, 0.8]),
        )
        self.assertEqual(result["training_prevalence"], 0.6)
        self.assertAlmostEqual(result["train_prevalence_constant_baseline_brier_score"], 0.26)
        self.assertAlmostEqual(result["model_brier_score"], 0.04)
        self.assertGreater(result["brier_skill_score"], 0)
        self.assertTrue(np.isfinite(result["model_log_loss"]))
        self.assertTrue(np.isfinite(result["train_prevalence_constant_baseline_log_loss"]))

    def test_rejects_non_finite_probabilities(self):
        with self.assertRaises(ValueError):
            probability_quality_metrics([0, 1], [0, 1], [0.2, float("nan")])

    def test_weighting_is_selected_from_validation_brier_scores(self):
        self.assertEqual(
            select_weighting_by_validation({"balanced": 0.207, "unweighted": 0.144}),
            "unweighted",
        )
        with self.assertRaises(ValueError):
            select_weighting_by_validation({"balanced": float("nan")})


if __name__ == "__main__":
    unittest.main()

