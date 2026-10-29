"""Checks source parsing, past-only features, the split and independently scored errors."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from main import FEATURES, audit_source, evaluate, load_source, make_features, next_observation_forecasts


class ForecastChecks(unittest.TestCase):
    def test_source_units_and_calendar_limits(self):
        raw, data = load_source()
        self.assertEqual(data.iloc[0]["total_orders"], 539577)
        self.assertEqual(data.iloc[5]["total_orders"], 207364)
        audit = audit_source(raw, data)
        self.assertTrue(audit["target_equals_type_A_plus_B_plus_C"])
        self.assertTrue(audit["target_equals_nonurgent_plus_urgent_plus_fiscal"])
        self.assertEqual(audit["weekday_cycle_breaks_at_observations"], [33, 40])

    def test_checksum_rejects_modified_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.csv"
            path.write_text("modified input", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_source(path)

    def test_lags_and_average_are_shifted(self):
        data = pd.DataFrame({"observation": range(1, 7), "total_orders": [10, 20, 30, 40, 50, 60]})
        frame = make_features(data)
        self.assertEqual(frame.iloc[0][FEATURES].tolist(), [30, 20, 10])
        self.assertEqual(frame.iloc[0]["trailing_mean_3"], 20)
        self.assertEqual(frame.iloc[0]["total_orders"], 40)

    def test_current_and_future_targets_cannot_change_current_features(self):
        _, data = load_source()
        original = make_features(data)
        changed = data.copy()
        changed.loc[changed["observation"] >= 51, "total_orders"] += 999999
        altered = make_features(changed)
        np.testing.assert_array_equal(original.loc[original["observation"] == 51, FEATURES],
                                      altered.loc[altered["observation"] == 51, FEATURES])
        first_model, first_predictions, _ = evaluate(original)
        changed_model, changed_predictions, _ = evaluate(altered)
        np.testing.assert_allclose(first_model.coef_, changed_model.coef_)
        self.assertAlmostEqual(first_predictions.iloc[0]["linear_regression"],
                               changed_predictions.iloc[0]["linear_regression"])

    def test_holdout_and_metrics_match_independent_calculation(self):
        _, data = load_source()
        frame = make_features(data)
        _, predictions, scores = evaluate(frame)
        self.assertEqual(len(frame[frame["observation"] <= 50]), 47)
        self.assertEqual(predictions["observation"].tolist(), list(range(51, 61)))
        self.assertEqual(set(scores["method"]), {"last_observation", "trailing_mean_3", "linear_regression"})
        for method in scores["method"]:
            error = predictions["actual"].to_numpy() - predictions[method].to_numpy()
            row = scores[scores["method"] == method].iloc[0]
            self.assertAlmostEqual(row["mae_orders"], np.abs(error).mean())
            self.assertAlmostEqual(row["rmse_orders"], np.sqrt(np.mean(error ** 2)))

    def test_next_observation_is_unscored_and_uses_latest_values(self):
        _, data = load_source()
        forecasts = next_observation_forecasts(data, make_features(data)).set_index("method")
        self.assertTrue((forecasts["observation"] == 61).all())
        self.assertEqual(forecasts.loc["last_observation", "predicted_orders"], data.iloc[-1]["total_orders"])
        self.assertEqual(forecasts.loc["trailing_mean_3", "predicted_orders"], data["total_orders"].tail(3).mean())
        self.assertTrue((forecasts["predicted_orders"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
