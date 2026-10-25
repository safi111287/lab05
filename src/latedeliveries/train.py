import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import make_orders
from .evaluation import (
    compare_models,
    order_cost,
    select_training_threshold,
    tune_logistic,
)
from .pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    split_orders,
)

"""Train the model and save its results."""


def write_json(path: Path, content: dict[str, object]) -> None:
    """Write a JSON report with a useful error message."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise OSError(f"Could not write JSON report to '{path}'.") from error


def main() -> int:
    """Compare, tune, choose a threshold, evaluate once, and save."""
    reports_folder = Path("reports")
    models_folder = Path("models")
    reports_folder.mkdir(parents=True, exist_ok=True)
    models_folder.mkdir(parents=True, exist_ok=True)

    orders = make_orders()
    x_train, x_test, y_train, y_test = split_orders(orders)

    comparison = compare_models(x_train, y_train)
    comparison.to_csv(
        reports_folder / "cv_results.csv",
        index=False,
        encoding="utf-8",
    )

    search = tune_logistic(x_train, y_train)
    threshold, threshold_costs = select_training_threshold(
        search.best_estimator_,
        x_train,
        y_train,
    )
    threshold_costs.to_csv(
        reports_folder / "threshold_costs.csv",
        index=False,
        encoding="utf-8",
    )

    # All choices are fixed before evaluating the held-out test set.
    final_pipeline = clone(search.best_estimator_)
    final_pipeline.fit(x_train, y_train)
    probabilities = final_pipeline.predict_proba(x_test)[:, 1]
    flagged = probabilities >= threshold

    metrics = {
        "threshold": threshold,
        "test_size": len(y_test),
        "confusion_matrix": confusion_matrix(y_test, flagged, labels=[0, 1]).tolist(),
        "precision": float(precision_score(y_test, flagged, zero_division=0)),
        "recall": float(recall_score(y_test, flagged, zero_division=0)),
        "f1": float(f1_score(y_test, flagged, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "costs": {
            "flag_nobody": order_cost(y_test, np.zeros(len(y_test), dtype=bool)),
            "flag_everybody": order_cost(y_test, np.ones(len(y_test), dtype=bool)),
            "model": order_cost(y_test, flagged),
        },
    }

    write_json(reports_folder / "metrics.json", metrics)

    model_info = {
        "scikit_learn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_settings": search.best_params_,
        "tuning": {
            "scoring": "roc_auc",
            "best_cv_mean": float(search.best_score_),
            "best_cv_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
        },
        "test_results": metrics,
    }

    model_path = models_folder / "late_orders.joblib"
    try:
        joblib.dump(final_pipeline, model_path)
    except OSError as error:
        raise OSError(f"Could not save model to '{model_path}'.") from error

    write_json(models_folder / "model_info.json", model_info)

    print(f"Chosen threshold: {threshold:.2f}")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
