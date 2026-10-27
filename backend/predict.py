"""
Predict heart disease for a patient from the terminal, using the 3 trained models.

Run from the project root (after `python backend/run_all.py`):
    python backend/predict.py              # type in a new patient's details
    python backend/predict.py --test 5     # use test patient no. 5 and compare with the real answer
"""

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import pandas as pd

from preprocessing.preprocessing_utils import (
    CATEGORY_LABELS, EDUCATIONAL_WARNING, FEATURE_DESCRIPTIONS, FEATURES, MODEL_DIRS,
    MODEL_NAMES, TARGET, TEST_PATH, _fix_dtypes,
)

# min, max, typical value (used if you just press Enter)
NUMBER_INPUTS = {
    "age": (18, 100, 55),
    "trestbps": (80, 220, 130),
    "chol": (100, 600, 240),
    "thalach": (60, 220, 150),
    "oldpeak": (0.0, 7.0, 1.0),
}
ORDER = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
         "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def load_models():
    models = {}
    for key, name in MODEL_NAMES.items():
        path = MODEL_DIRS[key] / "saved_model" / f"{key}_pipeline.joblib"
        if not path.exists():
            sys.exit(f"Model not found: {path}\nRun first:  python backend/run_all.py")
        models[name] = joblib.load(path)
    return models


def ask_number(feature):
    low, high, typical = NUMBER_INPUTS[feature]
    while True:
        text = input(f"  {FEATURE_DESCRIPTIONS[feature]} [{low}-{high}, Enter = {typical}]: ").strip()
        if text == "":
            return float(typical)
        try:
            value = float(text)
        except ValueError:
            print("    Please type a number.")
            continue
        if low <= value <= high:
            return value
        print(f"    Please enter a value between {low} and {high}.")


def ask_choice(feature, options, allow_unknown=False):
    print(f"  {FEATURE_DESCRIPTIONS[feature]}:")
    for i, opt in enumerate(options, start=1):
        print(f"     {i}. {str(opt).replace('_', ' ')}")
    if allow_unknown:
        print("     u. unknown / not measured")
    while True:
        text = input("    Choose a number: ").strip().lower()
        if allow_unknown and text == "u":
            return np.nan
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        print("    Invalid choice, try again.")


def ask_vessels():
    while True:
        text = input(f"  {FEATURE_DESCRIPTIONS['ca']} [0-3, u = unknown]: ").strip().lower()
        if text == "u":
            return np.nan
        if text in ("0", "1", "2", "3"):
            return float(text)
        print("    Please type 0, 1, 2, 3 or u.")


def enter_patient():
    print("\nEnter the patient's details:")
    patient = {}
    for f in ORDER:
        if f in NUMBER_INPUTS:
            patient[f] = ask_number(f)
        elif f == "ca":
            patient[f] = ask_vessels()
        else:
            patient[f] = ask_choice(f, list(CATEGORY_LABELS[f].values()), allow_unknown=(f == "thal"))
    return _fix_dtypes(pd.DataFrame([patient])[FEATURES])


def show_patient(X):
    print("\nPatient details:")
    for f in ORDER:
        v = X.iloc[0][f]
        if pd.isna(v):
            v = "unknown"
        elif isinstance(v, float) and v.is_integer():
            v = str(int(v))
        else:
            v = str(v).replace("_", " ")
        print(f"  {FEATURE_DESCRIPTIONS[f]:<55} {v}")


def predict(models, X, actual=None):
    print("\n" + "=" * 62)
    print(f"{'Model':<22}{'Probability of disease':<26}Prediction")
    print("-" * 62)
    votes = 0
    for name, model in models.items():
        prob = model.predict_proba(X)[0, 1]
        sick = prob >= 0.5
        votes += sick
        label = "HEART DISEASE" if sick else "No heart disease"
        mark = ""
        if actual is not None:
            mark = "  (correct)" if int(sick) == actual else "  (wrong)"
        print(f"{name:<22}{prob:>8.1%}{'':<18}{label}{mark}")
    print("=" * 62)
    if votes == len(models):
        print("All 3 models predict HEART DISEASE.")
    elif votes == 0:
        print("All 3 models predict NO heart disease.")
    else:
        print(f"Models disagree: {votes} of {len(models)} predict heart disease (borderline case).")
    if actual is not None:
        print(f"Actual diagnosis: {'HEART DISEASE' if actual == 1 else 'No heart disease'}")
    print("(A probability of 50% or more is counted as heart disease.)")


def main():
    parser = argparse.ArgumentParser(description="Predict heart disease from the terminal.")
    parser.add_argument("--test", type=int, metavar="N",
                        help="use test patient number N (1-61) instead of typing details")
    args = parser.parse_args()

    print("HEART DISEASE PREDICTION - Logistic Regression, Decision Tree, Random Forest")
    print(EDUCATIONAL_WARNING)
    models = load_models()

    if args.test is not None:
        test = _fix_dtypes(pd.read_csv(TEST_PATH))
        if not 1 <= args.test <= len(test):
            sys.exit(f"Choose a test patient between 1 and {len(test)}.")
        row = test.iloc[[args.test - 1]]
        X = row[FEATURES]
        show_patient(X)
        predict(models, X, actual=int(row[TARGET].iloc[0]))
        return

    while True:
        X = enter_patient()
        show_patient(X)
        predict(models, X)
        if input("\nPredict another patient? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
