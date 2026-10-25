"""Tests for feature selection and the train/test split."""

import pandas as pd
import pytest

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    split_orders,
)


@pytest.mark.parametrize(
    "column",
    ["order_id", "prep_minutes", "customer_rating", "late"],
)
def test_features_exclude_unavailable_or_unsuitable_columns(column: str) -> None:
    assert column not in NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_split_uses_only_selected_features(orders: pd.DataFrame) -> None:
    x_train, x_test, _, _ = split_orders(orders)
    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert list(x_train.columns) == expected
    assert list(x_test.columns) == expected


def test_split_is_repeatable(orders: pd.DataFrame) -> None:
    first = split_orders(orders)
    second = split_orders(orders)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])


def test_split_keeps_classes_stratified(orders: pd.DataFrame) -> None:
    _, _, y_train, y_test = split_orders(orders)
    overall_rate = orders["late"].mean()

    assert abs(y_train.mean() - overall_rate) <= 1 / len(y_train)
    assert abs(y_test.mean() - overall_rate) <= 1 / len(y_test)


def test_split_has_correct_sizes_and_no_overlap(orders: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = split_orders(orders)

    assert len(x_train) == len(y_train) == 300
    assert len(x_test) == len(y_test) == 100
    assert set(x_train.index).isdisjoint(x_test.index)
    assert set(x_train.index) | set(x_test.index) == set(orders.index)
    assert x_train.index.equals(y_train.index)
    assert x_test.index.equals(y_test.index)
