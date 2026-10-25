import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from .pipeline import RANDOM_STATE, build_pipeline

"""Compare models using cross-validation on training data."""

VOUCHER_COST = 4
MISSED_LATE_COST = 15
THRESHOLDS = tuple(step / 100 for step in range(5, 100, 5))


def compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Report mean and standard deviation across five training folds."""
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            build_pipeline(model),
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            error_score="raise",
        )

        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())

        rows.append(row)

    return pd.DataFrame(rows)


def tune_logistic(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Tune logistic regression using training folds only."""
    pipeline = build_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    )

    parameter_grid = {
        "model__C": [0.01, 0.1, 1.0],
        "model__fit_intercept": [True, False],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    return search


def order_cost(y_true: ArrayLike, flagged: ArrayLike) -> int:
    """Count vouchers and missed late orders."""
    actual = np.asarray(y_true)
    decisions = np.asarray(flagged)

    if actual.ndim != 1 or decisions.ndim != 1:
        raise ValueError("Labels and decisions must be one-dimensional.")
    if actual.shape != decisions.shape:
        raise ValueError("Labels and decisions must have the same length.")
    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Actual labels must contain only 0 and 1.")
    if not np.isin(decisions, [0, 1]).all():
        raise ValueError("Decisions must contain only boolean or 0/1 values.")

    decisions = decisions.astype(bool)
    vouchers = int(decisions.sum())
    missed_late = int(((actual == 1) & ~decisions).sum())

    return VOUCHER_COST * vouchers + MISSED_LATE_COST * missed_late


def choose_threshold(
    y_true: ArrayLike,
    probabilities: ArrayLike,
) -> tuple[float, pd.DataFrame]:
    """Choose the lowest-cost threshold; ties use the higher threshold."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("Probabilities must be a non-empty one-dimensional array.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")

    rows = []
    for threshold in THRESHOLDS:
        flagged = probabilities >= threshold
        rows.append(
            {
                "threshold": threshold,
                "cost": order_cost(y_true, flagged),
                "flagged_count": int(flagged.sum()),
            }
        )

    results = pd.DataFrame(rows)
    best = results.sort_values(
        ["cost", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    return float(best["threshold"]), results
