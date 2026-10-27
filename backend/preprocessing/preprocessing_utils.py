"""
Shared configuration, data loading, train/test splitting and preprocessing.

Everything that more than one script needs lives here, so the three models
always use exactly the same data, the same split and the same preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths (pathlib -> works on Windows, macOS and Linux)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = PROCESSED_DIR / "eda"

RAW_DATA_PATH = RAW_DIR / "heart_disease.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "heart_disease_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

COMPARISON_DIR = PROJECT_ROOT / "comparison" / "results"

# ---------------------------------------------------------------------------
# Reproducibility settings (identical for every model)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# If the minority class is below this share of the training set,
# models are trained with class_weight="balanced".
IMBALANCE_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Dataset schema (UCI Heart Disease, processed.cleveland.data)
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
ORIGINAL_TARGET = "num"   # 0 = no disease, 1-4 = disease present (severity)
TARGET = "target"         # binary target created during cleaning

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of the peak exercise ST segment",
    "ca": "Number of major vessels (0-3) coloured by fluoroscopy",
    "thal": "Thallium stress test result",
    "num": "Original diagnosis (0 = <50% narrowing, 1-4 = >50% narrowing)",
}

# Integer codes in the UCI file -> readable labels.
# Readable labels make the one-hot feature names, the decision tree plot and
# the feature-importance chart understandable during a demonstration.
CATEGORY_LABELS = {
    "sex": {0: "female", 1: "male"},
    "cp": {1: "typical_angina", 2: "atypical_angina",
           3: "non_anginal_pain", 4: "asymptomatic"},
    "fbs": {0: "no", 1: "yes"},
    "restecg": {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"},
    "exang": {0: "no", 1: "yes"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed_defect", 7: "reversible_defect"},
}

# Valid ranges for codes and physiologically possible ranges for measurements.
# Values outside these are treated as invalid (set to missing), NOT removed.
VALID_CODES = {
    "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
    "exang": {0, 1}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7}, "num": {0, 1, 2, 3, 4},
}
PLAUSIBLE_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),   # 0 would be physically impossible
    "chol": (50, 700),       # 0 would be physically impossible
    "thalach": (40, 250),
    "oldpeak": (0, 10),
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}
MODEL_DIRS = {
    "logistic_regression": PROJECT_ROOT / "model_1_logistic_regression",
    "decision_tree": PROJECT_ROOT / "model_2_decision_tree",
    "random_forest": PROJECT_ROOT / "model_3_random_forest",
}

EDUCATIONAL_WARNING = (
    "EDUCATIONAL USE ONLY: this project is a machine-learning exercise on a "
    "small 1988 research dataset. It is NOT a clinically validated diagnostic "
    "system and must never be used to make medical decisions."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def model_paths(model_key: str) -> dict[str, Path]:
    """Standard file locations for one model."""
    base = MODEL_DIRS[model_key]
    paths = {
        "base": base,
        "saved_model_dir": base / "saved_model",
        "results_dir": base / "results",
        "model_file": base / "saved_model" / f"{model_key}_pipeline.joblib",
        "params_file": base / "saved_model" / "best_params.json",
        "cv_results_file": base / "saved_model" / "cv_results.csv",
    }
    ensure_dirs(paths["saved_model_dir"], paths["results_dir"])
    return paths


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features -> float, categorical features -> object (str + NaN)."""
    df = df.copy()
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object)
        df.loc[df[col].isna(), col] = np.nan
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)
    return df


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}.\n"
            "Run:  python download_dataset.py"
        )
    return pd.read_csv(path, na_values=["?"])


def load_clean_data(path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}.\n"
            "Run:  python preprocessing/clean_data.py"
        )
    return _fix_dtypes(pd.read_csv(path))


# ---------------------------------------------------------------------------
# Train / test split (created ONCE, reused by every model)
# ---------------------------------------------------------------------------
def create_train_test_split(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split of the cleaned data, saved to data/processed/."""
    if TRAIN_PATH.exists() and TEST_PATH.exists() and not force:
        return _fix_dtypes(pd.read_csv(TRAIN_PATH)), _fix_dtypes(pd.read_csv(TEST_PATH))

    df = load_clean_data()
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    ensure_dirs(PROCESSED_DIR)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Split saved: {len(train_df)} training rows -> {TRAIN_PATH.name}, "
          f"{len(test_df)} test rows -> {TEST_PATH.name}")
    return _fix_dtypes(train_df), _fix_dtypes(test_df)


def get_train_test_data():
    """Returns X_train, X_test, y_train, y_test from the shared saved split."""
    train_df, test_df = create_train_test_split(force=False)
    return (
        train_df[FEATURES], test_df[FEATURES],
        train_df[TARGET].to_numpy(), test_df[TARGET].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Preprocessing pipeline (fitted ONLY on training data inside each model)
# ---------------------------------------------------------------------------
def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Numerical: median imputation (+ StandardScaler if scale_numeric).
    Categorical: most-frequent imputation + OneHotEncoder(handle_unknown="ignore").

    Tree models do not need scaling, so they pass scale_numeric=False.
    The transformer is unfitted here; it is fitted inside Pipeline.fit()
    on training data only, so no test-set information leaks into it.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERICAL_FEATURES),
            ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def choose_class_weight(y_train) -> tuple[str | None, str]:
    """Use class_weight='balanced' only if the training data is imbalanced."""
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
    minority_share = counts.min() / counts.sum()
    if minority_share < IMBALANCE_THRESHOLD:
        return "balanced", (f"Imbalance detected (minority share {minority_share:.1%}) "
                            "-> class_weight='balanced'")
    return None, (f"Classes roughly balanced (class 0: {counts[0]}, class 1: {counts[1]}, "
                  f"minority share {minority_share:.1%}) -> no re-weighting needed")


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names after one-hot encoding, e.g. 'cp_asymptomatic'."""
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
