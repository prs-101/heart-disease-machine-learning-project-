"""
Model 2 - Decision Tree: training.

Run from the project root:
    python model_2_decision_tree/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from common.evaluation_utils import tune_and_fit  # noqa: E402
from preprocessing.preprocessing_utils import (  # noqa: E402
    RANDOM_STATE, build_preprocessor, choose_class_weight, get_train_test_data,
)

MODEL_KEY = "decision_tree"

# Limiting depth / leaf size stops the tree from memorising the 242 training rows.
PARAM_GRID = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [2, 3, 4, 5, 6, None],
    "classifier__min_samples_leaf": [1, 5, 10, 20],
}


def build_pipeline(class_weight) -> Pipeline:
    return Pipeline([
        # Trees split on thresholds, so feature scaling is unnecessary.
        ("preprocessor", build_preprocessor(scale_numeric=False)),
        ("classifier", DecisionTreeClassifier(class_weight=class_weight,
                                              random_state=RANDOM_STATE)),
    ])


def main():
    print("=" * 70)
    print("MODEL 2: DECISION TREE - TRAINING")
    print("=" * 70)
    X_train, _, y_train, _ = get_train_test_data()
    class_weight, msg = choose_class_weight(y_train)
    print(f"  Training rows      : {len(y_train)}")
    print(f"  Class balance      : {msg}")
    return tune_and_fit(MODEL_KEY, build_pipeline(class_weight), PARAM_GRID, X_train, y_train)


if __name__ == "__main__":
    main()
