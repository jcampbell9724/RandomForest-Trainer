"""Small synthetic checks for training, saved models, and leakage boundaries."""

import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

import app


def make_pipeline(features, problem_type="regression", custom_specs=None):
    params = {
        "n_estimators": 24, "criterion": "squared_error", "max_depth": 5,
        "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt",
        "bootstrap": True, "max_samples": None, "class_weight": None,
        "random_state": 42,
    }
    if problem_type == "classification":
        params["criterion"] = "gini"
    steps = []
    if custom_specs:
        steps.append(("custom_features", app.CustomFeatureTransformer(custom_specs)))
        features, _, errors = app.apply_custom_column_specs(features, custom_specs)
        if errors:
            raise ValueError(errors)
    steps.extend([
        ("datetime_features", app.DatetimeFeatureTransformer()),
        ("preprocessor", app.build_preprocessor(app.normalize_feature_dtypes(features))),
        ("model", app.build_model(problem_type, params).set_params(n_jobs=1)),
    ])
    return Pipeline(steps)


class TrainingChecks(unittest.TestCase):
    def test_predict_midnight_after_training_on_timestamps(self):
        training = pd.DataFrame({
            "created_at": pd.date_range("2026-01-01 09:30", periods=24, freq="D"),
            "quantity": np.arange(24),
        })
        model = make_pipeline(training).fit(training, np.arange(24) * 1.5)
        held_out = pd.DataFrame({
            "created_at": pd.to_datetime(["2026-01-25", None]),
            "quantity": [24, 25],
        })
        predictions = model.predict(held_out)
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(np.isfinite(predictions).all())

    def test_text_targets_are_detected_for_pandas_string_dtypes(self):
        for dtype in ("object", "string"):
            target = pd.Series(["basic", "premium"] * 20, dtype=dtype)
            self.assertEqual(app.infer_problem_type(target), "classification")

    def test_group_average_uses_training_values_for_held_out_groups(self):
        specs = [{"kind": "group_average", "name": "group_mean",
                  "value_column": "value", "group_column": "group"}]
        training = pd.DataFrame({"value": [10, 30, 8], "group": ["a", "a", "b"]})
        transformer = app.CustomFeatureTransformer(specs).fit(training)
        held_out = pd.DataFrame({"value": [999, 500], "group": ["a", "new"]})
        derived = transformer.transform(held_out)
        self.assertEqual(derived.loc[0, "group_mean"], 20)
        self.assertTrue(pd.isna(derived.loc[1, "group_mean"]))
        leaky_specs = [{"kind": "pair", "name": "copy_of_target",
                        "left_column": "target", "right_column": "group"},
                       {"kind": "pair", "name": "indirect_copy",
                        "left_column": "copy_of_target", "right_column": "value"}]
        self.assertEqual(app.get_target_leaky_custom_features(leaky_specs, "target"),
                         ["copy_of_target", "indirect_copy"])

    def test_mixed_features_training_cv_and_saved_model_round_trip(self):
        rng = np.random.default_rng(42)
        features = pd.DataFrame({
            "quantity": rng.integers(1, 20, 96),
            "region": np.tile(["east", "west", "north"], 32),
        })
        features.loc[0, "quantity"] = np.nan
        numeric_target = features["quantity"].fillna(10) * 2 + rng.normal(0, 0.2, 96)
        class_target = pd.Series(np.where(numeric_target > 20, "high", "low"))
        for problem_type, target in (("regression", numeric_target),
                                     ("classification", class_target)):
            with self.subTest(problem_type=problem_type):
                X_train, X_test, y_train, _ = train_test_split(
                    features, target, random_state=42, test_size=0.25)
                model = make_pipeline(features, problem_type).fit(X_train, y_train)
                predictions = model.predict(X_test)
                duplicate = make_pipeline(features, problem_type).fit(X_train, y_train)
                np.testing.assert_array_equal(predictions, duplicate.predict(X_test))
                scores = cross_val_score(model, X_train, y_train, cv=3,
                                         scoring=app.get_scoring(problem_type))
                self.assertTrue(np.isfinite(scores).all())
                artifact = {"pipeline": model, "problem_type": problem_type,
                            "feature_columns": list(features.columns)}
                restored = app.unpack_model_artifact(pickle.loads(pickle.dumps(artifact)))
                np.testing.assert_array_equal(predictions, restored["pipeline"].predict(X_test))
                new_category = pd.DataFrame({"quantity": [8], "region": ["unseen"]})
                self.assertEqual(restored["pipeline"].predict(new_category).shape, (1,))
                importance = app.get_feature_importance_table(model)
                self.assertAlmostEqual(float(importance["importance"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
