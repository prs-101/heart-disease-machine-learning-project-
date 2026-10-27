"""
Model 3 - Random Forest: training.

Run from the project root:
    python model_3_random_forest/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from common.evaluation_utils import tune_and_fit  # noqa: E402
from preprocessing.preprocessing_utils import (  # noqa: E402
    RANDOM_STATE, build_preprocessor, choose_class_weight, get_train_test_data,
)

MODEL_KEY = "random_forest"

PARAM_GRID = {
    "classifier__n_estimators": [200, 500],
    "classifier__max_depth": [None, 4, 8],
    "classifier__min_samples_leaf": [1, 3, 5],
    "classifier__max_features": ["sqrt", 0.5],
}


def build_pipeline(class_weight) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=False)),  # trees need no scaling
        # n_jobs=1 here because GridSearchCV already runs folds in parallel
        ("classifier", RandomForestClassifier(class_weight=class_weight,
                                              random_state=RANDOM_STATE, n_jobs=1)),
    ])


def main():
    print("=" * 70)
    print("MODEL 3: RANDOM FOREST - TRAINING")
    print("=" * 70)
    X_train, _, y_train, _ = get_train_test_data()
    class_weight, msg = choose_class_weight(y_train)
    print(f"  Training rows      : {len(y_train)}")
    print(f"  Class balance      : {msg}")
    print("  (Random Forest grid search can take 1-2 minutes ...)")
    return tune_and_fit(MODEL_KEY, build_pipeline(class_weight), PARAM_GRID, X_train, y_train)


if __name__ == "__main__":
    main()
