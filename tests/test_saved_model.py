import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from pathlib import Path

from latedeliveries.pipeline import (
    RANDOM_STATE,
    build_pipeline,
    split_orders,
)

"""Check that saving preserves the whole prediction pipeline."""


def test_saved_pipeline_preserves_predictions(
    orders: pd.DataFrame,
    tmp_path: Path,
) -> None:
    x_train, x_test, y_train, _ = split_orders(orders)
    pipeline = build_pipeline(
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )
    pipeline.fit(x_train, y_train)

    # Include an order that needs imputation and an unseen category.
    new_orders = x_test.copy()
    new_orders.loc[new_orders.index[0], "distance_km"] = np.nan
    new_orders.loc[new_orders.index[0], "weather"] = "hail"

    expected_probabilities = pipeline.predict_proba(new_orders)
    expected_predictions = pipeline.predict(new_orders)

    model_path = tmp_path / "late_orders.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)

    np.testing.assert_array_equal(
        restored.predict_proba(new_orders),
        expected_probabilities,
    )
    np.testing.assert_array_equal(
        restored.predict(new_orders),
        expected_predictions,
    )
