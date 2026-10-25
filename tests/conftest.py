"""Shared data for the tests."""

import pandas as pd
import pytest

from latedeliveries.data import make_orders


@pytest.fixture
def orders() -> pd.DataFrame:
    """Return the same sample of orders for each test."""
    return make_orders(n=400)
