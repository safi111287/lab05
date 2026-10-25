import pytest

from latedeliveries.evaluation import choose_threshold, order_cost

"""Hand-calculated tests for costs and threshold selection."""


@pytest.mark.parametrize(
    ("flagged", "expected_cost"),
    [
        ([False, False, False, False], 30),
        ([True, True, True, True], 16),
        ([True, False, True, False], 8),
        ([True, True, False, False], 23),
    ],
)
def test_order_cost(flagged: list[bool], expected_cost: int) -> None:
    # Orders 1 and 3 are late.
    # The last case costs two vouchers plus one missed late order.
    actual = [1, 0, 1, 0]

    assert order_cost(actual, flagged) == expected_cost


def test_threshold_selects_cheapest_decisions() -> None:
    actual = [0, 1, 0, 1]
    probabilities = [0.10, 0.70, 0.20, 0.90]

    threshold, results = choose_threshold(actual, probabilities)

    # Thresholds above 0.20 through 0.70 flag only the two late orders.
    # They cost $8; the tie rule chooses 0.70.
    assert threshold == pytest.approx(0.70)
    assert order_cost(actual, [p >= threshold for p in probabilities]) == 8
    assert results["cost"].min() == 8


def test_threshold_can_flag_nobody() -> None:
    actual = [0, 0, 0]
    probabilities = [0.10, 0.20, 0.30]

    threshold, _ = choose_threshold(actual, probabilities)
    flagged = [p >= threshold for p in probabilities]

    assert not any(flagged)
    assert order_cost(actual, flagged) == 0
    assert threshold == pytest.approx(0.95)


def test_threshold_can_flag_everybody() -> None:
    actual = [1, 1, 1]
    probabilities = [0.10, 0.20, 0.30]

    threshold, _ = choose_threshold(actual, probabilities)
    flagged = [p >= threshold for p in probabilities]

    # Three vouchers cost $12; missing even one late order costs $15.
    assert all(flagged)
    assert order_cost(actual, flagged) == 12
    assert threshold == pytest.approx(0.10)


def test_threshold_checks_all_required_values() -> None:
    _, results = choose_threshold([0, 1], [0.10, 0.80])

    assert results["threshold"].tolist() == pytest.approx(
        [step / 100 for step in range(5, 100, 5)]
    )


def test_cost_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        order_cost([0, 1], [True])


@pytest.mark.parametrize("probabilities", [[-0.1], [1.1], [float("nan")]])
def test_threshold_rejects_invalid_probabilities(
    probabilities: list[float],
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1|finite"):
        choose_threshold([1], probabilities)
