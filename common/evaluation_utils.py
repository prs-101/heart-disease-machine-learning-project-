"""
Shared training and evaluation helpers used by all three models and the
comparison script, so every model is tuned, scored and plotted identically.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNG files, never open windows
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from preprocessing.preprocessing_utils import (  # noqa: E402
    EDUCATIONAL_WARNING, MODEL_NAMES, PROJECT_ROOT, get_cv, model_paths,
)

CLASS_NAMES = ["No disease (0)", "Disease (1)"]
CV_SCORING = "roc_auc"  # threshold-independent; robust for model selection


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def tune_and_fit(model_key: str, pipeline: Pipeline, param_grid: dict,
                 X_train, y_train) -> Pipeline:
    """
    Grid search with stratified 5-fold cross-validation on the TRAINING data
    only. The test set is never seen here. The best pipeline (preprocessing +
    classifier) is refitted on the whole training set and saved with joblib.
    """
    paths = model_paths(model_key)
    search = GridSearchCV(pipeline, param_grid, scoring=CV_SCORING, cv=get_cv(),
                          n_jobs=-1, refit=True, return_train_score=False)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    joblib.dump(best, paths["model_file"])

    cv = pd.DataFrame(search.cv_results_)
    keep = [c for c in cv.columns if c.startswith("param_")] + \
           ["mean_test_score", "std_test_score", "rank_test_score"]
    cv[keep].sort_values("rank_test_score").to_csv(paths["cv_results_file"], index=False)

    info = {
        "model": MODEL_NAMES[model_key],
        "cv_scoring": CV_SCORING,
        "cv_folds": get_cv().get_n_splits(),
        "best_cv_score": round(float(search.best_score_), 4),
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in search.best_params_.items()},
        "n_training_rows": int(len(y_train)),
    }
    paths["params_file"].write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"  Combinations tried : {len(cv)} x {info['cv_folds']}-fold CV (training data only)")
    print(f"  Best params        : {info['best_params']}")
    print(f"  Best CV {CV_SCORING:<10} : {info['best_cv_score']:.4f}")
    print(f"  Saved pipeline     : {paths['model_file'].relative_to(PROJECT_ROOT)}")
    return best


def load_model(model_key: str) -> Pipeline:
    path = model_paths(model_key)["model_file"]
    if not path.exists():
        folder = path.parents[1].name
        raise FileNotFoundError(f"No saved model at {path}.\nRun:  python {folder}/train.py")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    annot = np.array([[f"TN\n{cm[0, 0]}", f"FP\n{cm[0, 1]}"],
                      [f"FN\n{cm[1, 0]}", f"TP\n{cm[1, 1]}"]])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"fontsize": 13})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_proba, label: str, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall / sensitivity)")
    ax.set_title(f"ROC curve - {label}")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full evaluation of one model
# ---------------------------------------------------------------------------
def evaluate_model(model_key: str, pipeline: Pipeline, X_test, y_test) -> dict:
    """Scores the pipeline on the held-out test set and saves CSV/TXT/PNG results."""
    name = MODEL_NAMES[model_key]
    out = model_paths(model_key)["results_dir"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # metrics.csv
    pd.DataFrame([{"Algorithm": name, **metrics}]).round(4).to_csv(out / "metrics.csv", index=False)

    # classification report (CSV + readable TXT)
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                        output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).T.round(4).to_csv(out / "classification_report.csv")
    report_txt = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                       digits=4, zero_division=0)
    (out / "classification_report.txt").write_text(
        f"{name} - test set ({len(y_test)} patients)\n\n{report_txt}\n\n{EDUCATIONAL_WARNING}\n",
        encoding="utf-8")

    # per-patient predictions
    pd.DataFrame({"true_label": y_test, "predicted_label": y_pred,
                  "probability_disease": np.round(y_proba, 4)}).to_csv(
        out / "test_predictions.csv", index=False)

    # plots
    plot_confusion_matrix(y_test, y_pred, f"Confusion matrix - {name}", out / "confusion_matrix.png")
    plot_roc(y_test, y_proba, name, out / "roc_curve.png")

    print(f"\n{name} - held-out test set ({len(y_test)} patients)")
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1-score  : {metrics['F1-score']:.4f}")
    print(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}")
    print(f"  Confusion : TN={metrics['TN']}  FP={metrics['FP']}  "
          f"FN={metrics['FN']} (missed disease cases)  TP={metrics['TP']}")
    print("\nClassification report:\n" + report_txt)
    print(f"Results saved in {out.relative_to(PROJECT_ROOT)}")
    return metrics
