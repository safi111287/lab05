import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from .pipeline import RANDOM_STATE, build_pipeline

"""Compare models using cross-validation on training data."""


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
