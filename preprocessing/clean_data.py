"""
Data cleaning + audit report for the UCI Heart Disease (Cleveland) dataset.

What this script does (and does NOT do):
  * Inspects shape, columns, dtypes, missing values, duplicates.
  * Removes exact duplicate rows (they would double-count a patient).
  * Converts values outside the documented code lists / physically impossible
    measurements into missing values (rule-based, no statistics learned).
  * Creates the binary target: 0 = no disease, 1 = disease (original 1-4)
    and drops the original 'num' column (keeping it would leak the answer).
  * Replaces integer category codes with readable labels.
  * Reports outliers (IQR rule) but does NOT remove them - they are
    plausible patient measurements, and deleting real data to improve
    accuracy would be scientifically wrong.
  * Does NOT impute, scale or encode. Those steps learn statistics (median,
    mean, std, categories) and are therefore fitted later inside sklearn
    Pipelines on the TRAINING split only, to avoid data leakage.

Run from the project root:
    python preprocessing/clean_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from preprocessing.preprocessing_utils import (  # noqa: E402
    CATEGORICAL_FEATURES, CATEGORY_LABELS, CLEAN_DATA_PATH, FEATURE_DESCRIPTIONS,
    NUMERICAL_FEATURES, ORIGINAL_TARGET, PLAUSIBLE_RANGES, PROCESSED_DIR, RAW_DATA_PATH, TARGET,
    VALID_CODES, ensure_dirs, load_raw_data,
)

REPORT_PATH = PROCESSED_DIR / "cleaning_report.txt"
BEFORE_SUMMARY_PATH = PROCESSED_DIR / "summary_before_cleaning.csv"
AFTER_SUMMARY_PATH = PROCESSED_DIR / "summary_after_cleaning.csv"
OUTLIER_REPORT_PATH = PROCESSED_DIR / "outlier_report.csv"


class Report:
    """Collects lines, prints them and writes them to a text file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, title: str) -> None:
        self.add()
        self.add("=" * 70)
        self.add(title)
        self.add("=" * 70)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines), encoding="utf-8")


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "column": col,
            "description": FEATURE_DESCRIPTIONS.get(col, "Binary target (0 = no disease, 1 = disease)"),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "unique_values": int(s.nunique(dropna=True)),
            "min": s.min() if numeric else "",
            "max": s.max() if numeric else "",
            "mean": round(s.mean(), 3) if numeric else "",
        })
    return pd.DataFrame(rows)


def find_invalid_values(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks of values that break the documented coding / physical limits."""
    masks = {}
    for col, allowed in VALID_CODES.items():
        masks[col] = df[col].notna() & ~df[col].isin(allowed)
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        masks[col] = df[col].notna() & ((df[col] < low) | (df[col] > high))
    return masks


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERICAL_FEATURES:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        rows.append({"feature": col, "q1": q1, "q3": q3, "lower_fence": low,
                     "upper_fence": high, "n_outliers": n_out,
                     "action": "kept (plausible clinical value)"})
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    ensure_dirs(PROCESSED_DIR)
    rep = Report()

    # ------------------------------------------------------------------ load
    df = load_raw_data()
    rep.section("1. RAW DATA - BEFORE CLEANING")
    rep.add(f"Source file : {RAW_DATA_PATH.relative_to(PROJECT_ROOT)}")
    rep.add(f"Shape       : {df.shape[0]} rows x {df.shape[1]} columns")
    rep.add(f"Columns     : {list(df.columns)}")
    rep.add("\nData types:")
    for col, dt in df.dtypes.items():
        rep.add(f"  {col:<10} {dt}")

    missing = df.isna().sum()
    rep.add(f"\nMissing values (originally '?'): {int(missing.sum())} in total")
    for col, n in missing[missing > 0].items():
        rep.add(f"  {col:<10} {n} missing ({n / len(df):.1%})")

    before = column_summary(df)
    before.to_csv(BEFORE_SUMMARY_PATH, index=False)

    rep.add("\nOriginal target 'num' distribution (0 = no disease, 1-4 = disease):")
    for val, cnt in df[ORIGINAL_TARGET].value_counts().sort_index().items():
        rep.add(f"  num = {int(val)} : {cnt}")

    # ------------------------------------------------------------ duplicates
    rep.section("2. DUPLICATES")
    n_dup = int(df.duplicated().sum())
    rep.add(f"Exact duplicate rows found: {n_dup}")
    if n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        rep.add(f"Removed {n_dup} duplicate rows (identical records would be double-counted).")
    else:
        rep.add("No duplicates - nothing removed.")

    # -------------------------------------------------------- invalid values
    rep.section("3. INVALID VALUES")
    masks = find_invalid_values(df)
    total_invalid = 0
    for col, mask in masks.items():
        n = int(mask.sum())
        total_invalid += n
        if n:
            rep.add(f"  {col:<10} {n} invalid value(s): {sorted(df.loc[mask, col].unique())} -> set to missing")
            df.loc[mask, col] = np.nan
    if total_invalid == 0:
        rep.add("All values are within documented codes and physically plausible ranges.")
    rep.add("Invalid values are converted to missing (imputed later inside the pipeline), "
            "not deleted.")

    # ---------------------------------------------------------------- target
    rep.section("4. TARGET CONVERSION (multi-class -> binary)")
    n_no_target = int(df[ORIGINAL_TARGET].isna().sum())
    if n_no_target:
        df = df[df[ORIGINAL_TARGET].notna()].reset_index(drop=True)
        rep.add(f"Dropped {n_no_target} rows with no diagnosis (cannot be used for supervised learning).")
    df[TARGET] = (df[ORIGINAL_TARGET] > 0).astype(int)
    df = df.drop(columns=[ORIGINAL_TARGET])
    counts = df[TARGET].value_counts().sort_index()
    rep.add("target = 0 if num == 0 (no heart disease); target = 1 if num in 1-4 (disease present)")
    rep.add(f"  target = 0 : {counts.get(0, 0)} ({counts.get(0, 0) / len(df):.1%})")
    rep.add(f"  target = 1 : {counts.get(1, 0)} ({counts.get(1, 0) / len(df):.1%})")
    rep.add("Original 'num' column dropped - keeping it would leak the answer into the features.")

    # ------------------------------------------------- variable types/labels
    rep.section("5. VARIABLE TYPES")
    rep.add(f"Numerical   ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
    rep.add(f"Categorical ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
    rep.add("'ca' (0-3 vessels) is an ordered count, so it is treated as numerical.")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].map(CATEGORY_LABELS[col]).astype(object)
    rep.add("Categorical codes replaced with readable labels, e.g. cp 4 -> 'asymptomatic'.")

    # -------------------------------------------------------------- outliers
    rep.section("6. OUTLIERS (IQR rule, reported only)")
    out = outlier_report(df)
    out.to_csv(OUTLIER_REPORT_PATH, index=False)
    for _, r in out.iterrows():
        rep.add(f"  {r.feature:<10} {r.n_outliers} value(s) outside "
                f"[{r.lower_fence:.1f}, {r.upper_fence:.1f}]")
    rep.add("Decision: outliers are KEPT. They are plausible patient values "
            "(e.g. high cholesterol), and removing real data to improve accuracy is not justified.")

    # ----------------------------------------------------------------- after
    rep.section("7. CLEANED DATA - AFTER CLEANING")
    rep.add(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    remaining = df.isna().sum()
    rep.add(f"Missing values still present: {int(remaining.sum())}")
    for col, n in remaining[remaining > 0].items():
        rep.add(f"  {col:<10} {n} -> will be imputed inside the model pipeline "
                "(median / most-frequent, fitted on TRAINING data only)")
    after = column_summary(df)
    after.to_csv(AFTER_SUMMARY_PATH, index=False)

    df.to_csv(CLEAN_DATA_PATH, index=False)
    rep.add(f"\nSaved cleaned dataset -> {CLEAN_DATA_PATH.relative_to(PROJECT_ROOT)}")
    rep.add(f"Saved summaries       -> {BEFORE_SUMMARY_PATH.name}, {AFTER_SUMMARY_PATH.name}, "
            f"{OUTLIER_REPORT_PATH.name}")

    rep.section("LEAKAGE NOTE")
    rep.add("Only rule-based steps were applied to the full dataset (duplicates, invalid codes, "
            "target conversion, label mapping). Imputation, scaling and one-hot encoding are "
            "fitted after the train/test split, on the training data only.")
    rep.save(REPORT_PATH)
    print(f"\nCleaning report written to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return df


if __name__ == "__main__":
    main()
