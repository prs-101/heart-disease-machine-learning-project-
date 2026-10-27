"""
Run the complete project end to end:

    python run_all.py              # full pipeline
    python run_all.py --skip-eda   # skip EDA plots (faster)

Steps:
  1. Download the dataset (skipped if data/raw/heart_disease.csv exists)
  2. Clean the data and write the audit report
  3. Exploratory data analysis plots
  4. Create the common stratified 80/20 train/test split
  5. Train all three models (CV tuning on training data only)
  6. Evaluate each model on the held-out test set
  7. Save model files (done during training)
  8. Generate comparison charts, final table and winner
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import download_dataset  # noqa: E402
from preprocessing import clean_data, eda  # noqa: E402
from preprocessing.preprocessing_utils import EDUCATIONAL_WARNING, create_train_test_split  # noqa: E402

MODEL_FOLDERS = [
    "model_1_logistic_regression",
    "model_2_decision_tree",
    "model_3_random_forest",
]


def banner(step: int, text: str) -> None:
    print("\n" + "#" * 78)
    print(f"# STEP {step}: {text}")
    print("#" * 78)


def main(skip_eda: bool = False) -> None:
    start = time.time()
    print(EDUCATIONAL_WARNING)

    banner(1, "DATASET DOWNLOAD")
    download_dataset.main()

    banner(2, "DATA CLEANING + AUDIT REPORT")
    clean_data.main()

    if not skip_eda:
        banner(3, "EXPLORATORY DATA ANALYSIS")
        eda.main()

    banner(4, "COMMON TRAIN/TEST SPLIT (stratified 80/20, random_state=42)")
    # force=True: always rebuild from the freshly cleaned data
    create_train_test_split(force=True)

    banner(5, "TRAINING (and saving) ALL THREE MODELS")
    for folder in MODEL_FOLDERS:
        importlib.import_module(f"{folder}.train").main()

    banner(6, "EVALUATING EACH MODEL ON THE HELD-OUT TEST SET")
    for folder in MODEL_FOLDERS:
        importlib.import_module(f"{folder}.evaluate").main()

    banner(7, "MODEL COMPARISON")
    importlib.import_module("comparison.compare_models").main()

    print(f"\nAll done in {time.time() - start:.1f} s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the whole heart disease ML project.")
    parser.add_argument("--skip-eda", action="store_true", help="skip the EDA plots")
    main(skip_eda=parser.parse_args().skip_eda)
