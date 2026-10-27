"""
Compare all three trained models on the SAME held-out test set.

Run from the project root (after the three train.py scripts):
    python comparison/compare_models.py

Outputs in comparison/results/:
    01_accuracy_comparison.png ... 05_roc_auc_comparison.png
    06_combined_roc_curves.png, 07_all_metrics_comparison.png,
    08_confusion_matrices.png, model_comparison.csv, model_comparison.md,
    comparison_report.txt
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402

from common.evaluation_utils import CLASS_NAMES, compute_metrics, load_model  # noqa: E402
from preprocessing.preprocessing_utils import (  # noqa: E402
    COMPARISON_DIR, EDUCATIONAL_WARNING, MODEL_NAMES, ensure_dirs, get_train_test_data,
)

MAIN_METRICS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
COLORS = {"Logistic Regression": "#4C72B0", "Decision Tree": "#DD8452",
          "Random Forest": "#55A868"}

MEDICAL_NOTE = """
WHY ACCURACY ALONE IS NOT ENOUGH IN MEDICAL CLASSIFICATION
-----------------------------------------------------------
* Accuracy counts every mistake equally. In screening, the two mistakes are not equal:
  - False Negative (FN): a sick patient is told "no disease" -> treatment is delayed.
    This is usually the most dangerous error.
  - False Positive (FP): a healthy patient is flagged -> extra tests, anxiety, cost,
    but the error is normally caught by follow-up tests.
* Recall (sensitivity) = TP / (TP + FN) measures how many real disease cases the model
  catches. A screening model should have high recall, i.e. few false negatives.
* Precision = TP / (TP + FP) measures how many flagged patients are really sick.
* ROC-AUC measures how well the model RANKS sick patients above healthy ones across
  ALL decision thresholds (0.5 = random, 1.0 = perfect). A model with high AUC can be
  moved to a lower threshold to raise recall, at the cost of more false positives.
* With only ~61 test patients, one patient changes accuracy by ~1.6 percentage points,
  so small differences between models are NOT statistically meaningful. The CV scores
  saved in each saved_model/best_params.json give a second, more stable estimate.
"""


def bar_chart(df: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(df["Algorithm"], df[metric],
                  color=[COLORS[a] for a in df["Algorithm"]], width=0.55)
    for b, v in zip(bars, df[metric]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    best = df[metric].max()
    ax.axhline(best, color="grey", ls=":", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} comparison (same held-out test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_chart(df: pd.DataFrame, path: Path) -> None:
    long = df.melt(id_vars="Algorithm", value_vars=MAIN_METRICS,
                   var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="Metric", y="Score", hue="Algorithm", palette=COLORS, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.12)
    ax.set_title("All metrics - three algorithms on the same test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def combined_roc(curves: dict, df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name, (fpr, tpr) in curves.items():
        auc = df.loc[df["Algorithm"] == name, "ROC-AUC"].item()
        ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.500)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves - all models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confusion_grid(preds: dict, y_test, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        annot = np.array([[f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"],
                          [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"]])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                    annot_kws={"fontsize": 12})
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion matrices (FN = missed disease cases)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leaders(df: pd.DataFrame, column: str, higher_is_better: bool = True) -> list[str]:
    target = df[column].max() if higher_is_better else df[column].min()
    return df.loc[np.isclose(df[column], target), "Algorithm"].tolist()


def main() -> pd.DataFrame:
    ensure_dirs(COMPARISON_DIR)
    _, X_test, _, y_test = get_train_test_data()

    rows, curves, preds = [], {}, {}
    for key, name in MODEL_NAMES.items():
        pipeline = load_model(key)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"Algorithm": name, **compute_metrics(y_test, y_pred, y_proba)})
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        curves[name], preds[name] = (fpr, tpr), y_pred

    df = pd.DataFrame(rows)
    table = df[["Algorithm"] + MAIN_METRICS + ["Specificity", "TN", "FP", "FN", "TP"]].round(4)
    table.to_csv(COMPARISON_DIR / "model_comparison.csv", index=False)

    md = ["| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |",
          "|-----------|----------|-----------|--------|----------|---------|"]
    for _, r in table.iterrows():
        md.append(f"| {r['Algorithm']} | " + " | ".join(f"{r[m]:.4f}" for m in MAIN_METRICS) + " |")
    (COMPARISON_DIR / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for i, metric in enumerate(MAIN_METRICS, start=1):
        fname = f"0{i}_{metric.lower().replace('-', '_')}_comparison.png"
        bar_chart(df, metric, COMPARISON_DIR / fname)
    combined_roc(curves, df, COMPARISON_DIR / "06_combined_roc_curves.png")
    grouped_chart(df, COMPARISON_DIR / "07_all_metrics_comparison.png")
    confusion_grid(preds, y_test, COMPARISON_DIR / "08_confusion_matrices.png")

    # ------------------------------------------------------------ verdict
    acc_winners = leaders(df, "Accuracy")
    best_acc = df["Accuracy"].max()
    if len(acc_winners) == 1:
        verdict = f"WINNER (highest accuracy): {acc_winners[0]} with accuracy {best_acc:.4f}"
    else:
        verdict = (f"TIE on accuracy ({best_acc:.4f}) between: {', '.join(acc_winners)}. "
                   "Use recall / ROC-AUC below to choose between them.")

    lines = [
        "FINAL MODEL COMPARISON - held-out test set "
        f"({len(y_test)} patients: {int(y_test.sum())} with disease, "
        f"{int(len(y_test) - y_test.sum())} without)",
        "=" * 78,
        table[["Algorithm"] + MAIN_METRICS].to_string(index=False),
        "",
        "Errors per model:",
        table[["Algorithm", "FN", "FP", "Specificity"]].to_string(index=False),
        "",
        verdict,
        f"Highest recall (fewest missed disease cases): {', '.join(leaders(df, 'Recall'))} "
        f"({df['Recall'].max():.4f})",
        f"Highest ROC-AUC (best ranking ability)     : {', '.join(leaders(df, 'ROC-AUC'))} "
        f"({df['ROC-AUC'].max():.4f})",
        f"Fewest false negatives                     : {', '.join(leaders(df, 'FN', False))} "
        f"({int(df['FN'].min())})",
        MEDICAL_NOTE,
        EDUCATIONAL_WARNING,
    ]
    report = "\n".join(lines)
    (COMPARISON_DIR / "comparison_report.txt").write_text(report, encoding="utf-8")

    print("=" * 78)
    print(report)
    print(f"\nCharts, CSV and report saved in {COMPARISON_DIR.relative_to(PROJECT_ROOT)}")
    return table


if __name__ == "__main__":
    main()
