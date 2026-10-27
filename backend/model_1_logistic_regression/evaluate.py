"""
Model 1 - Logistic Regression: evaluation on the held-out test set.

Run from the project root (after train.py):
    python model_1_logistic_regression/evaluate.py
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

MODEL_KEY = "logistic_regression"


def plot_coefficients(pipeline, out_dir: Path, top_n: int = 15) -> None:
    """Positive coefficient -> raises predicted disease probability."""
    names = get_feature_names(pipeline)
    coefs = pipeline.named_steps["classifier"].coef_[0]
    df = pd.DataFrame({"feature": names, "coefficient": coefs})
    df["abs"] = df["coefficient"].abs()
    df.sort_values("abs", ascending=False).drop(columns="abs").to_csv(
        out_dir / "coefficients.csv", index=False)

    top = df.nlargest(top_n, "abs").sort_values("coefficient")
    colors = ["#C44E52" if c > 0 else "#4C72B0" for c in top["coefficient"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Logistic Regression - top {top_n} coefficients\n"
                 "(red = increases disease risk, blue = decreases)")
    ax.set_xlabel("Coefficient (log-odds per unit of scaled / one-hot feature)")
    fig.tight_layout()
    fig.savefig(out_dir / "coefficients.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("MODEL 1: LOGISTIC REGRESSION - EVALUATION")
    print("=" * 70)
    pipeline = load_model(MODEL_KEY)
    _, X_test, _, y_test = get_train_test_data()
    metrics = evaluate_model(MODEL_KEY, pipeline, X_test, y_test)
    plot_coefficients(pipeline, model_paths(MODEL_KEY)["results_dir"])
    return metrics


if __name__ == "__main__":
    main()
