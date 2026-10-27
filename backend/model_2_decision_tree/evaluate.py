"""
Model 2 - Decision Tree: evaluation on the held-out test set + tree plot.

Run from the project root (after train.py):
    python model_2_decision_tree/evaluate.py
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
from sklearn.tree import export_text, plot_tree  # noqa: E402

from common.evaluation_utils import CLASS_NAMES, evaluate_model, load_model  # noqa: E402
from preprocessing.preprocessing_utils import (  # noqa: E402
    get_feature_names, get_train_test_data, model_paths,
)

MODEL_KEY = "decision_tree"
MAX_PLOT_DEPTH = 4  # deeper levels become unreadable on one image


def plot_decision_tree(pipeline, out_dir: Path) -> None:
    tree = pipeline.named_steps["classifier"]
    names = get_feature_names(pipeline)
    depth = tree.get_depth()
    shown = min(depth, MAX_PLOT_DEPTH)
    leaves = tree.get_n_leaves()

    width = max(16, min(40, 2.2 * min(leaves, 2 ** shown)))
    fig, ax = plt.subplots(figsize=(width, 3 + 2.6 * shown))
    plot_tree(tree, feature_names=names, class_names=CLASS_NAMES, filled=True,
              rounded=True, impurity=False, proportion=False, precision=2,
              max_depth=MAX_PLOT_DEPTH, fontsize=10, ax=ax)
    note = f" (first {MAX_PLOT_DEPTH} levels shown)" if depth > MAX_PLOT_DEPTH else ""
    ax.set_title(f"Decision Tree - depth {depth}, {leaves} leaves{note}\n"
                 "One-hot features: '<= 0.5' means the category is ABSENT (left branch)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "decision_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Full tree as text rules (always complete, regardless of depth)
    (out_dir / "decision_tree_rules.txt").write_text(
        export_text(tree, feature_names=names, decimals=2, show_weights=True),
        encoding="utf-8")


def main():
    print("=" * 70)
    print("MODEL 2: DECISION TREE - EVALUATION")
    print("=" * 70)
    pipeline = load_model(MODEL_KEY)
    _, X_test, _, y_test = get_train_test_data()
    metrics = evaluate_model(MODEL_KEY, pipeline, X_test, y_test)
    plot_decision_tree(pipeline, model_paths(MODEL_KEY)["results_dir"])
    return metrics


if __name__ == "__main__":
    main()
