"""
Web interface for the Heart Disease Prediction project (Streamlit).

Run from the project root, AFTER `python run_all.py` has trained the models:
    python -m streamlit run app.py

A browser tab opens at http://localhost:8501
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from preprocessing.preprocessing_utils import (  # noqa: E402
    CATEGORICAL_FEATURES, CATEGORY_LABELS, CLEAN_DATA_PATH, COMPARISON_DIR, EDA_DIR,
    EDUCATIONAL_WARNING, FEATURE_DESCRIPTIONS, FEATURES, MODEL_DIRS, MODEL_NAMES,
    NUMERICAL_FEATURES, PROCESSED_DIR, TARGET, TEST_PATH, _fix_dtypes,
)

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.result-card {border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
               border: 1px solid rgba(128,128,128,.25);}
.result-card h4 {margin: 0 0 6px 0; font-size: 1rem; opacity: .8;}
.result-card .prob {font-size: 2.2rem; font-weight: 700; line-height: 1.1;}
.result-card .label {font-weight: 600; margin-top: 4px;}
.pos {background: rgba(214, 69, 65, .10); border-color: rgba(214, 69, 65, .45);}
.neg {background: rgba(46, 139, 87, .10); border-color: rgba(46, 139, 87, .45);}
.pos .label {color: #d64541;} .neg .label {color: #2e8b57;}
.small-note {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

# Friendly display text for category labels used in the cleaned data
NICE = {
    "female": "Female", "male": "Male",
    "typical_angina": "Typical angina", "atypical_angina": "Atypical angina",
    "non_anginal_pain": "Non-anginal pain", "asymptomatic": "Asymptomatic",
    "no": "No", "yes": "Yes",
    "normal": "Normal", "st_t_abnormality": "ST-T wave abnormality",
    "lv_hypertrophy": "Left ventricular hypertrophy",
    "upsloping": "Upsloping", "flat": "Flat", "downsloping": "Downsloping",
    "fixed_defect": "Fixed defect", "reversible_defect": "Reversible defect",
}
UNKNOWN = "Unknown / not measured"

NUMERIC_INPUTS = {  # min, max, default, step
    "age": (18, 100, 55, 1),
    "trestbps": (80, 220, 130, 1),
    "chol": (100, 600, 240, 1),
    "thalach": (60, 220, 150, 1),
    "oldpeak": (0.0, 7.0, 1.0, 0.1),
}


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
def model_file(key: str) -> Path:
    return MODEL_DIRS[key] / "saved_model" / f"{key}_pipeline.joblib"


@st.cache_resource(show_spinner=False)
def _load_pipeline(path: str, mtime: float):
    return joblib.load(path)


def load_models() -> dict:
    models = {}
    for key, name in MODEL_NAMES.items():
        f = model_file(key)
        if f.exists():
            models[name] = _load_pipeline(str(f), f.stat().st_mtime)
    return models


@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv(path: Path) -> pd.DataFrame | None:
    return _read_csv(str(path), path.stat().st_mtime) if path.exists() else None


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def show_image(path: Path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"Missing: {path.relative_to(PROJECT_ROOT)}")


def predict_all(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, pipe in models.items():
        p = float(pipe.predict_proba(X)[0, 1])
        rows.append({"Model": name, "Probability": p, "Prediction": int(p >= 0.5)})
    return pd.DataFrame(rows)


def result_cards(results: pd.DataFrame) -> None:
    cols = st.columns(len(results))
    for col, (_, r) in zip(cols, results.iterrows()):
        pos = r["Prediction"] == 1
        label = "Heart disease likely" if pos else "No heart disease predicted"
        col.markdown(
            f"""<div class="result-card {'pos' if pos else 'neg'}">
                <h4>{r['Model']}</h4>
                <div class="prob">{r['Probability']:.0%}</div>
                <div class="small-note">probability of disease</div>
                <div class="label">{label}</div></div>""",
            unsafe_allow_html=True,
        )
    n_pos = int(results["Prediction"].sum())
    n = len(results)
    if n_pos == n:
        st.error(f"All {n} models predict **heart disease** for this patient.")
    elif n_pos == 0:
        st.success(f"All {n} models predict **no heart disease** for this patient.")
    else:
        st.warning(f"Models disagree: {n_pos} of {n} predict heart disease. "
                   "This patient is a borderline case.")
    st.caption("Decision threshold: probability ≥ 50 % → disease. " + EDUCATIONAL_WARNING)


# ---------------------------------------------------------------------------
# Header + status
# ---------------------------------------------------------------------------
st.title("❤️ Heart Disease Prediction")
st.markdown("Comparative study of **Logistic Regression**, **Decision Tree** and "
            "**Random Forest** on the UCI Heart Disease (Cleveland) dataset.")
st.info(EDUCATIONAL_WARNING, icon="⚠️")

models = load_models()
with st.sidebar:
    st.header("Project status")
    st.write(("✅" if CLEAN_DATA_PATH.exists() else "❌") + " Cleaned dataset")
    for key, name in MODEL_NAMES.items():
        st.write(("✅" if model_file(key).exists() else "❌") + f" {name}")
    st.write(("✅" if (COMPARISON_DIR / "model_comparison.csv").exists() else "❌")
             + " Comparison results")
    if len(models) < 3:
        st.warning("Some outputs are missing. In the terminal run:\n\n`python run_all.py`\n\n"
                   "then refresh this page.")
    st.divider()
    st.caption("Dataset: Janosi, Steinbrunn, Pfisterer & Detrano (1989), "
               "UCI Machine Learning Repository, doi:10.24432/C52P4X")

tab_predict, tab_compare, tab_models, tab_data = st.tabs(
    ["🩺 Predict", "📊 Model comparison", "🔍 Model details", "📁 Data & EDA"])

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    if not models:
        st.error("No trained models found. Run `python run_all.py` first, then refresh.")
    else:
        st.subheader("Enter patient details")
        with st.form("patient_form"):
            c1, c2, c3 = st.columns(3)
            values = {}
            with c1:
                st.markdown("**Demographics & vitals**")
                values["age"] = st.number_input(FEATURE_DESCRIPTIONS["age"], *NUMERIC_INPUTS["age"][:3],
                                                step=NUMERIC_INPUTS["age"][3])
                values["sex"] = st.selectbox(FEATURE_DESCRIPTIONS["sex"],
                                             list(CATEGORY_LABELS["sex"].values()),
                                             index=1, format_func=NICE.get)
                values["trestbps"] = st.number_input(FEATURE_DESCRIPTIONS["trestbps"],
                                                     *NUMERIC_INPUTS["trestbps"][:3], step=1)
                values["chol"] = st.number_input(FEATURE_DESCRIPTIONS["chol"],
                                                 *NUMERIC_INPUTS["chol"][:3], step=1)
                values["fbs"] = st.selectbox(FEATURE_DESCRIPTIONS["fbs"],
                                             list(CATEGORY_LABELS["fbs"].values()),
                                             format_func=NICE.get)
            with c2:
                st.markdown("**Symptoms & ECG**")
                values["cp"] = st.selectbox(FEATURE_DESCRIPTIONS["cp"],
                                            list(CATEGORY_LABELS["cp"].values()),
                                            format_func=NICE.get)
                values["restecg"] = st.selectbox(FEATURE_DESCRIPTIONS["restecg"],
                                                 list(CATEGORY_LABELS["restecg"].values()),
                                                 format_func=NICE.get)
                values["exang"] = st.selectbox(FEATURE_DESCRIPTIONS["exang"],
                                               list(CATEGORY_LABELS["exang"].values()),
                                               format_func=NICE.get)
                values["thalach"] = st.number_input(FEATURE_DESCRIPTIONS["thalach"],
                                                    *NUMERIC_INPUTS["thalach"][:3], step=1)
            with c3:
                st.markdown("**Exercise test & imaging**")
                values["oldpeak"] = st.number_input(FEATURE_DESCRIPTIONS["oldpeak"],
                                                    *NUMERIC_INPUTS["oldpeak"][:3],
                                                    step=NUMERIC_INPUTS["oldpeak"][3])
                values["slope"] = st.selectbox(FEATURE_DESCRIPTIONS["slope"],
                                               list(CATEGORY_LABELS["slope"].values()),
                                               format_func=NICE.get)
                ca = st.selectbox(FEATURE_DESCRIPTIONS["ca"], [0, 1, 2, 3, UNKNOWN])
                values["ca"] = np.nan if ca == UNKNOWN else float(ca)
                thal = st.selectbox(FEATURE_DESCRIPTIONS["thal"],
                                    list(CATEGORY_LABELS["thal"].values()) + [UNKNOWN],
                                    format_func=lambda v: NICE.get(v, v))
                values["thal"] = np.nan if thal == UNKNOWN else thal
            st.caption("'Unknown' values are filled in by each pipeline's imputer, "
                       "learned from the training data.")
            submitted = st.form_submit_button("Predict", type="primary", width="stretch")

        if submitted:
            X = _fix_dtypes(pd.DataFrame([values])[FEATURES])
            st.subheader("Prediction")
            result_cards(predict_all(models, X))

        st.divider()
        st.subheader("Try a real patient from the held-out test set")
        test_df = read_csv(TEST_PATH)
        if test_df is None:
            st.caption("Test split not found. Run `python run_all.py`.")
        else:
            test_df = _fix_dtypes(test_df)
            idx = st.selectbox(
                "Test patient", range(len(test_df)),
                format_func=lambda i: (f"Patient {i + 1}  ·  age {int(test_df.iloc[i]['age'])}, "
                                       f"{test_df.iloc[i]['sex']}  ·  actual: "
                                       f"{'disease' if test_df.iloc[i][TARGET] == 1 else 'no disease'}"))
            row = test_df.iloc[[idx]]
            with st.expander("Show this patient's measurements"):
                show = row[FEATURES].T.rename(columns={row.index[0]: "Value"})
                show.insert(0, "Feature", [FEATURE_DESCRIPTIONS[f] for f in FEATURES])
                st.dataframe(show.astype(str), width="stretch", hide_index=True)
            res = predict_all(models, row[FEATURES])
            actual = int(row[TARGET].iloc[0])
            st.markdown(f"**Actual diagnosis:** {'Heart disease (1)' if actual else 'No heart disease (0)'}")
            result_cards(res)
            correct = (res["Prediction"] == actual)
            st.caption("Correct: " + ", ".join(
                f"{m} {'✅' if c else '❌'}" for m, c in zip(res["Model"], correct)))

# ---------------------------------------------------------------------------
# Tab 2: Comparison
# ---------------------------------------------------------------------------
with tab_compare:
    comp = read_csv(COMPARISON_DIR / "model_comparison.csv")
    if comp is None:
        st.error("No comparison results yet. Run `python run_all.py`.")
    else:
        main = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        best_acc = comp["Accuracy"].max()
        winners = comp.loc[np.isclose(comp["Accuracy"], best_acc), "Algorithm"].tolist()

        def kpi(col, title: str, column: str, fmt, higher: bool = True) -> None:
            best = comp[column].max() if higher else comp[column].min()
            names = comp.loc[np.isclose(comp[column], best), "Algorithm"].tolist()
            with col.container(border=True):
                st.metric(title, fmt(best))
                st.markdown(f"**{' / '.join(names)}**")

        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Highest accuracy", "Accuracy", lambda v: f"{v:.1%}")
        kpi(k2, "Highest recall", "Recall", lambda v: f"{v:.1%}")
        kpi(k3, "Highest ROC-AUC", "ROC-AUC", lambda v: f"{v:.3f}")
        kpi(k4, "Fewest missed cases (FN)", "FN", lambda v: f"{int(v)} patients", higher=False)
        if len(winners) > 1:
            st.warning(f"Tie on accuracy between {', '.join(winners)}.")

        st.markdown("#### Results on the same held-out test set")
        st.dataframe(
            comp[["Algorithm"] + main + ["Specificity", "FN", "FP"]]
            .style.format({m: "{:.4f}" for m in main + ["Specificity"]})
            .highlight_max(subset=main + ["Specificity"], color="rgba(46,139,87,.25)")
            .highlight_min(subset=["FN", "FP"], color="rgba(46,139,87,.25)"),
            width="stretch", hide_index=True)
        st.caption("Green = best value in each column. FN = sick patients predicted healthy.")

        c1, c2 = st.columns(2)
        with c1:
            show_image(COMPARISON_DIR / "07_all_metrics_comparison.png", "All metrics")
        with c2:
            show_image(COMPARISON_DIR / "06_combined_roc_curves.png", "ROC curves")
        show_image(COMPARISON_DIR / "08_confusion_matrices.png", "Confusion matrices")

        with st.expander("Individual metric charts"):
            charts = sorted(COMPARISON_DIR.glob("0[1-5]_*.png"))
            cols = st.columns(3)
            for i, p in enumerate(charts):
                with cols[i % 3]:
                    show_image(p)
        with st.expander("Why accuracy alone is not enough"):
            st.text(read_text(COMPARISON_DIR / "comparison_report.txt") or "")

# ---------------------------------------------------------------------------
# Tab 3: Model details
# ---------------------------------------------------------------------------
with tab_models:
    key = st.radio("Model", list(MODEL_NAMES), format_func=MODEL_NAMES.get, horizontal=True)
    res_dir = MODEL_DIRS[key] / "results"
    params = read_text(MODEL_DIRS[key] / "saved_model" / "best_params.json")
    metrics = read_csv(res_dir / "metrics.csv")
    if params is None or metrics is None:
        st.error(f"No results for {MODEL_NAMES[key]} yet. Run `python run_all.py`.")
    else:
        info = json.loads(params)
        m = metrics.iloc[0]
        cols = st.columns(6)
        for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
            col.metric(name, f"{m[name]:.3f}")
        cols[5].metric(f"CV {info['cv_scoring']}", f"{info['best_cv_score']:.3f}",
                       help="Mean score over 5-fold cross-validation on the training data")

        st.markdown("**Best hyperparameters** (chosen by cross-validation on training data only)")
        st.json({k.replace("classifier__", ""): v for k, v in info["best_params"].items()})

        c1, c2 = st.columns(2)
        with c1:
            show_image(res_dir / "confusion_matrix.png")
        with c2:
            show_image(res_dir / "roc_curve.png")

        extra = {"logistic_regression": ("coefficients.png", "Red = raises disease risk, blue = lowers it"),
                 "decision_tree": ("decision_tree.png", "Left branch = condition true"),
                 "random_forest": ("feature_importance.png", "Which features the forest relied on most")}[key]
        show_image(res_dir / extra[0], extra[1])

        with st.expander("Classification report"):
            st.text(read_text(res_dir / "classification_report.txt") or "")
        if key == "decision_tree":
            rules = read_text(res_dir / "decision_tree_rules.txt")
            if rules:
                with st.expander("Full decision tree as text rules"):
                    st.code(rules, language=None)

# ---------------------------------------------------------------------------
# Tab 4: Data & EDA
# ---------------------------------------------------------------------------
with tab_data:
    clean = read_csv(CLEAN_DATA_PATH)
    if clean is None:
        st.error("Cleaned data not found. Run `python run_all.py`.")
    else:
        n_pos = int(clean[TARGET].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients (after cleaning)", len(clean))
        c2.metric("Features", len(FEATURES))
        c3.metric("With heart disease", f"{n_pos} ({n_pos / len(clean):.0%})")
        c4.metric("Missing values left", int(clean[FEATURES].isna().sum().sum()),
                  help="Imputed inside each model pipeline using training-data statistics")

        st.markdown(f"Numerical: `{', '.join(NUMERICAL_FEATURES)}`  ·  "
                    f"Categorical: `{', '.join(CATEGORICAL_FEATURES)}`")
        with st.expander("Cleaned dataset preview"):
            st.dataframe(clean.head(50), width="stretch")
        with st.expander("Cleaning audit report"):
            st.text(read_text(PROCESSED_DIR / "cleaning_report.txt") or "")

        st.markdown("#### Exploratory data analysis")
        eda_imgs = sorted(EDA_DIR.glob("*.png"))
        if not eda_imgs:
            st.caption("EDA plots not found (run `python run_all.py` without --skip-eda).")
        for i in range(0, len(eda_imgs), 2):
            cols = st.columns(2)
            for col, p in zip(cols, eda_imgs[i:i + 2]):
                with col:
                    show_image(p, p.stem[3:].replace("_", " ").capitalize())