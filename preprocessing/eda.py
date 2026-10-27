"""
Exploratory Data Analysis (EDA) on the cleaned dataset.

EDA is descriptive only: it learns nothing that is fed into the models,
so looking at the full dataset here does not cause leakage.

Run from the project root (after clean_data.py):
    python preprocessing/eda.py
Outputs: data/processed/eda/*.png and eda_summary.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from preprocessing.preprocessing_utils import (  # noqa: E402
    CATEGORICAL_FEATURES, CATEGORY_LABELS, EDA_DIR, FEATURE_DESCRIPTIONS, NUMERICAL_FEATURES, TARGET,
    ensure_dirs, load_clean_data, load_raw_data,
)

TARGET_LABELS = {0: "No disease (0)", 1: "Disease (1)"}
PALETTE = {"No disease (0)": "#4C72B0", "Disease (1)": "#C44E52"}


def _save(fig, name: str) -> Path:
    path = EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")
    return path


def plot_missing_raw() -> None:
    raw = load_raw_data()
    missing = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax, color="#8172B2")
    ax.set_title("Missing values per column in the RAW data ('?' in the UCI file)")
    ax.set_ylabel("Number of missing values")
    for i, v in enumerate(missing.values):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom")
    _save(fig, "01_missing_values_raw.png")


def plot_class_balance(df: pd.DataFrame) -> None:
    labels = df[TARGET].map(TARGET_LABELS)
    fig, ax = plt.subplots(figsize=(6, 4))
    order = list(TARGET_LABELS.values())
    sns.countplot(x=labels, order=order, hue=labels, palette=PALETTE, legend=False, ax=ax)
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{int(h)} ({h / len(df):.0%})", (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom")
    ax.set_title("Target class distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Patients")
    _save(fig, "02_class_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    plot_df = df.assign(Diagnosis=df[TARGET].map(TARGET_LABELS))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), NUMERICAL_FEATURES):
        sns.histplot(data=plot_df, x=col, hue="Diagnosis", kde=True, palette=PALETTE,
                     element="step", stat="count", ax=ax)
        ax.set_title(FEATURE_DESCRIPTIONS[col], fontsize=10)
    fig.suptitle("Numerical features by diagnosis", fontsize=14)
    _save(fig, "03_numerical_distributions.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    plot_df = df.assign(Diagnosis=df[TARGET].map(TARGET_LABELS))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), NUMERICAL_FEATURES):
        sns.boxplot(data=plot_df, x="Diagnosis", y=col, hue="Diagnosis", palette=PALETTE,
                    legend=False, ax=ax)
        ax.set_title(FEATURE_DESCRIPTIONS[col], fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Box plots (dots beyond whiskers = IQR outliers, kept in the data)", fontsize=14)
    _save(fig, "04_numerical_boxplots.png")


def plot_categorical(df: pd.DataFrame) -> None:
    plot_df = df.assign(Diagnosis=df[TARGET].map(TARGET_LABELS))
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        order = [v for v in CATEGORY_LABELS[col].values() if v in set(plot_df[col].dropna())]
        sns.countplot(data=plot_df, x=col, order=order, hue="Diagnosis", palette=PALETTE, ax=ax)
        ax.set_title(FEATURE_DESCRIPTIONS[col], fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Categorical features by diagnosis", fontsize=14)
    _save(fig, "05_categorical_distributions.png")


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df[NUMERICAL_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix (numerical features + target)")
    _save(fig, "06_correlation_heatmap.png")


def write_summary(df: pd.DataFrame) -> None:
    lines = ["EXPLORATORY DATA ANALYSIS SUMMARY", "=" * 60,
             f"Rows: {len(df)}   Features: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}",
             "", "Numerical feature statistics:",
             df[NUMERICAL_FEATURES].describe().round(2).to_string(), "",
             "Mean of numerical features by target (0 = no disease, 1 = disease):",
             df.groupby(TARGET)[NUMERICAL_FEATURES].mean().round(2).to_string(), "",
             "Disease rate (% target = 1) within each category:"]
    for col in CATEGORICAL_FEATURES:
        rate = (df.groupby(col)[TARGET].agg(["count", "mean"])
                  .rename(columns={"mean": "disease_rate"}))
        rate["disease_rate"] = (100 * rate["disease_rate"]).round(1)
        lines += [f"\n[{col}] {FEATURE_DESCRIPTIONS[col]}", rate.to_string()]
    corr = df[NUMERICAL_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    lines += ["", "Correlation of numerical features with target:", corr.round(3).to_string()]
    path = EDA_DIR / "eda_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  saved {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ensure_dirs(EDA_DIR)
    sns.set_theme(style="whitegrid")
    df = load_clean_data()
    print("Generating EDA plots ...")
    plot_missing_raw()
    plot_class_balance(df)
    plot_numeric_distributions(df)
    plot_boxplots(df)
    plot_categorical(df)
    plot_correlation(df)
    write_summary(df)


if __name__ == "__main__":
    main()
