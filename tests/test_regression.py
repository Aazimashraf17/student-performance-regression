"""Check data separation and the mathematics independently of scikit-learn."""

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from regression import FEATURES, fit_models, load_data, score_models
from predict import predict_grade


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.train = cls.data[cls.data.split == "train"]
        cls.test = cls.data[cls.data.split == "test"]
        cls.models = fit_models(cls.train)

    def test_split_is_disjoint_and_covers_source(self):
        train_ids, test_ids = set(self.train.row_id), set(self.test.row_id)
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(train_ids | test_ids, set(range(1, 396)))
        self.assertEqual((len(train_ids), len(test_ids)), (316, 79))
        self.assertEqual(int((self.data.G3 == 0).sum()), 38)
        self.assertTrue(all("G3" not in features for features in FEATURES.values()))

    def test_ols_agrees_with_independent_linear_algebra(self):
        for name in ["simple_G2", "multiple_G1_G2"]:
            train_x = np.column_stack([np.ones(len(self.train)), self.train[FEATURES[name]]])
            test_x = np.column_stack([np.ones(len(self.test)), self.test[FEATURES[name]]])
            coefficients, *_ = np.linalg.lstsq(train_x, self.train.G3, rcond=None)
            expected = test_x @ coefficients
            actual = self.models[name].predict(self.test[FEATURES[name]])
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_baseline_uses_only_training_target(self):
        prediction = self.models["mean_baseline"].predict(self.test[["G2"]])
        np.testing.assert_allclose(prediction, self.train.G3.mean())
        metrics, _ = score_models(self.models, self.test)
        baseline = metrics[metrics.model == "mean_baseline"].iloc[0]
        self.assertAlmostEqual(baseline.mae, float(np.abs(self.test.G3 - self.train.G3.mean()).mean()))

    def test_saved_equation_matches_trained_model(self):
        for g1, g2 in [(12, 14), (0, 0), (20, 20)]:
            expected = self.models["multiple_G1_G2"].intercept_ + np.dot(
                self.models["multiple_G1_G2"].coef_, [g1, g2])
            self.assertAlmostEqual(predict_grade(g1, g2), expected, places=10)

    def test_invalid_grades_are_rejected(self):
        for value in [-1, 21, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                predict_grade(value, 12)
            with self.assertRaises(ValueError):
                predict_grade(12, value)


if __name__ == "__main__":
    unittest.main()
