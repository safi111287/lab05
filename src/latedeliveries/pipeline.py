"""Feature selection and a repeatable train/test split."""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 7

NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "courier_months",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
    "vehicle",
]


def split_orders(
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Keep 25% of orders aside for the final evaluation."""
    features = orders[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = orders["late"]

    return train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )
