"""
Model 1 - Logistic Regression: training.

Run from the project root:
    python model_1_logistic_regression/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from common.evaluation_utils import tune_and_fit  # noqa: E402
from preprocessing.preprocessing_utils import (  # noqa: E402
    RANDOM_STATE, build_preprocessor, choose_class_weight, get_train_test_data,
)

MODEL_KEY = "logistic_regression"

# Hyperparameters searched with 5-fold CV on the training data only.
# C = inverse regularisation strength (smaller C = simpler model).
PARAM_GRID = {
    "classifier__C": [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
}


def build_pipeline(class_weight) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),  # LR needs scaled inputs
        ("classifier", LogisticRegression(max_iter=5000, solver="lbfgs",
                                          class_weight=class_weight,
                                          random_state=RANDOM_STATE)),
    ])


def main():
    print("=" * 70)
    print("MODEL 1: LOGISTIC REGRESSION - TRAINING")
    print("=" * 70)
    X_train, _, y_train, _ = get_train_test_data()
    class_weight, msg = choose_class_weight(y_train)
    print(f"  Training rows      : {len(y_train)}")
    print(f"  Class balance      : {msg}")
    return tune_and_fit(MODEL_KEY, build_pipeline(class_weight), PARAM_GRID, X_train, y_train)


if __name__ == "__main__":
    main()
