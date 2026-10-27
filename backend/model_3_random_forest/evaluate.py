"""
Model 3 - Random Forest: evaluation on the held-out test set + feature importance.

Run from the project root (after train.py):
    python model_3_random_forest/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common.evaluation_utils import evaluate_model, load_model  # noqa: E402
from preprocessing.preprocessing_utils import (  # noqa: E402
    get_feature_names, get_train_test_data, model_paths,
)

MODEL_KEY = "random_forest"


def plot_feature_importance(pipeline, out_dir: Path, top_n: int = 15) -> None:
    """Mean decrease in impurity, using the one-hot (transformed) feature names."""
    forest = pipeline.named_steps["classifier"]
    imp = pd.DataFrame({"feature": get_feature_names(pipeline),
                        "importance": forest.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)

    top = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#55A868")
    for y, v in enumerate(top["importance"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=9)
    ax.set_title(f"Random Forest - top {top_n} feature importances\n(mean decrease in Gini impurity)")
    ax.set_xlabel("Importance (all features sum to 1)")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("MODEL 3: RANDOM FOREST - EVALUATION")
    print("=" * 70)
    pipeline = load_model(MODEL_KEY)
    _, X_test, _, y_test = get_train_test_data()
    metrics = evaluate_model(MODEL_KEY, pipeline, X_test, y_test)
    plot_feature_importance(pipeline, model_paths(MODEL_KEY)["results_dir"])
    return metrics


if __name__ == "__main__":
    main()
