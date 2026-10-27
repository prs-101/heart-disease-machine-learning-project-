# Heart Disease Prediction Using Three Machine Learning Algorithms — A Comparative Study

> ⚠️ **Educational use only.** This is a machine-learning exercise on a small research dataset collected in 1988. It is **not** a clinically validated diagnostic system and must never be used to make medical decisions.

## 1. Project overview

This project predicts whether a patient has heart disease (yes / no) from 13 clinical measurements, and compares three algorithms on exactly the same data:

| # | Algorithm | Why it is included |
|---|-----------|--------------------|
| 1 | Logistic Regression | Simple, interpretable linear baseline |
| 2 | Decision Tree | Human-readable if/else rules |
| 3 | Random Forest | Ensemble of many trees, usually more robust |

The pipeline covers: dataset download → cleaning + audit report → exploratory data analysis (EDA) → leakage-free preprocessing → hyperparameter tuning with cross-validation → evaluation on a held-out test set → comparison charts → winner.

## 2. Dataset

* **Name:** Heart Disease (Cleveland subset, `processed.cleveland.data`)
* **Source page:** https://archive.ics.uci.edu/dataset/45/heart+disease
* **Direct file:** https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
* **Size:** 303 patients, 13 features + 1 target; 6 missing values marked `?` (4 in `ca`, 2 in `thal`)
* **License:** CC BY 4.0

**Citation**

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

**Target conversion.** The original column `num` is 0 (no disease) or 1–4 (disease of increasing severity). It is converted to binary in `preprocessing/clean_data.py`:

* `target = 0` → no heart disease (`num = 0`)
* `target = 1` → heart disease present (`num = 1, 2, 3 or 4`)

The original `num` column is then dropped, because keeping it would give the answer away.

**Features**

| Column | Meaning | Type |
|--------|---------|------|
| age | Age in years | numerical |
| sex | female / male | categorical |
| cp | Chest pain type (typical angina, atypical angina, non-anginal pain, asymptomatic) | categorical |
| trestbps | Resting blood pressure (mm Hg) | numerical |
| chol | Serum cholesterol (mg/dl) | numerical |
| fbs | Fasting blood sugar > 120 mg/dl (yes/no) | categorical |
| restecg | Resting ECG (normal, ST-T abnormality, LV hypertrophy) | categorical |
| thalach | Maximum heart rate achieved | numerical |
| exang | Exercise-induced angina (yes/no) | categorical |
| oldpeak | ST depression induced by exercise | numerical |
| slope | Slope of peak exercise ST segment (up, flat, down) | categorical |
| ca | Number of major vessels (0–3) coloured by fluoroscopy | numerical (ordered count) |
| thal | Thallium test (normal, fixed defect, reversible defect) | categorical |

## 3. Install Python (Windows)

1. Download Python 3.11 or newer from https://www.python.org/downloads/
2. Run the installer and **tick "Add python.exe to PATH"** on the first screen.
3. Check it worked — open *Command Prompt* and type:
   ```
   python --version
   ```

## 4. VS Code setup

1. Install VS Code from https://code.visualstudio.com/
2. Open VS Code → Extensions (Ctrl+Shift+X) → install **Python** (by Microsoft).
3. Extract the project zip somewhere simple, e.g. `C:\Projects\heart-disease-ml`.
4. *File → Open Folder…* → select the `heart-disease-ml` folder (the one that contains `run_all.py`).
5. Open a terminal: *Terminal → New Terminal* (Ctrl+`). It opens in the project root.

## 5. Virtual environment (Windows)

In the VS Code terminal:

```
python -m venv .venv
.venv\Scripts\activate
```

You should now see `(.venv)` at the start of the prompt.

* If PowerShell says *"running scripts is disabled"*, run this once and try again:
  ```
  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
  ```
* Then press Ctrl+Shift+P → **Python: Select Interpreter** → choose the one inside `.venv`.

## 6. Install dependencies

```
pip install -r requirements.txt
```

## 7. Folder structure

```
heart-disease-ml/
├── data/
│   ├── raw/                     heart_disease.csv (downloaded, untouched)
│   └── processed/               cleaned data, train/test split, reports, eda/ plots
├── preprocessing/
│   ├── clean_data.py            cleaning + before/after audit report
│   ├── eda.py                   exploratory data analysis plots
│   └── preprocessing_utils.py   shared paths, split, ColumnTransformer
├── common/
│   └── evaluation_utils.py      shared tuning, metrics and plotting code
├── model_1_logistic_regression/ train.py, evaluate.py, saved_model/, results/
├── model_2_decision_tree/       train.py, evaluate.py, saved_model/, results/
├── model_3_random_forest/       train.py, evaluate.py, saved_model/, results/
├── comparison/
│   ├── compare_models.py
│   └── results/
├── download_dataset.py
├── run_all.py
├── requirements.txt
└── README.md
```

`common/` and `preprocessing/eda.py` were added to the requested structure so that tuning, evaluation and EDA code is written once and shared, instead of being copied into every model folder.

## 8. Run everything (one command)

```
python run_all.py
```

Takes about 1–2 minutes (the Random Forest grid search is the slowest part). Use `python run_all.py --skip-eda` to skip the EDA plots.

**Always run commands from the project root** (the folder containing `run_all.py`).

## 9. Run the steps separately

```
python download_dataset.py                       # 1. download (skips if already there)
python preprocessing/clean_data.py               # 2. clean + audit report
python preprocessing/eda.py                      # 3. EDA plots

python model_1_logistic_regression/train.py      # 4. train one model
python model_1_logistic_regression/evaluate.py   # 5. evaluate it

python model_2_decision_tree/train.py
python model_2_decision_tree/evaluate.py

python model_3_random_forest/train.py
python model_3_random_forest/evaluate.py

python comparison/compare_models.py              # 6. compare all three
```

The train/test split is created automatically the first time a model script runs and saved to `data/processed/train.csv` and `test.csv`, so every model uses the identical split. `run_all.py` rebuilds it from the freshly cleaned data.

### If the automatic download fails

1. Open https://archive.ics.uci.edu/dataset/45/heart+disease and click **Download**.
2. Extract `heart+disease.zip` and find `processed.cleveland.data`.
3. Run:
   ```
   python download_dataset.py --from-file "C:\path\to\processed.cleveland.data"
   ```

The script validates the file (14 columns, ~303 rows, target values 0–4). It never generates fake data.

## 10. How leakage is prevented

| Step | Where it happens | Uses statistics from data? |
|------|------------------|---------------------------|
| Remove duplicates, flag impossible values, convert target, readable labels | `clean_data.py`, whole dataset | No – fixed rules only |
| Stratified 80/20 split, `random_state=42` | `preprocessing_utils.py` | — |
| Median / most-frequent imputation, StandardScaler, OneHotEncoder | Inside each model's `Pipeline`, fitted on **training data only** | Yes |
| Hyperparameter tuning | `GridSearchCV`, 5-fold stratified CV on **training data only** | Yes |
| Test set | Used **once**, for final evaluation | — |

Other choices:

* **Outliers are reported, not removed.** They are plausible patient values; deleting real data to improve accuracy is not justified.
* **Class imbalance** is checked automatically. If the minority class is below 40 % of the training data, `class_weight="balanced"` is used. The Cleveland data is roughly 54 % / 46 %, so no re-weighting is normally needed (the training log says which was chosen).
* **The whole fitted pipeline** (preprocessing + model) is saved with joblib, so a saved model can take raw patient rows directly.
* Scaling is applied for Logistic Regression only; trees split on thresholds and don't need it.

## 11. Where to find results

| Folder | Contents |
|--------|----------|
| `data/processed/` | `cleaning_report.txt`, before/after summaries, `outlier_report.csv`, cleaned data, train/test split |
| `data/processed/eda/` | 6 EDA plots + `eda_summary.txt` |
| `model_X/saved_model/` | fitted pipeline (`.joblib`), `best_params.json` (best CV score), `cv_results.csv` |
| `model_X/results/` | `metrics.csv`, `classification_report.csv/.txt`, `test_predictions.csv`, `confusion_matrix.png`, `roc_curve.png` + model-specific plot |
| `comparison/results/` | 8 comparison charts, `model_comparison.csv`, `model_comparison.md`, `comparison_report.txt` |

Open PNG files by clicking them in the VS Code Explorer.

## 12. Metrics explained

TP = sick patient predicted sick, TN = healthy predicted healthy, FP = healthy predicted sick, **FN = sick predicted healthy (missed case)**.

| Metric | Formula | Plain meaning |
|--------|---------|---------------|
| Accuracy | (TP+TN) / all | Share of all predictions that were right |
| Precision | TP / (TP+FP) | Of patients flagged as sick, how many really are |
| Recall (sensitivity) | TP / (TP+FN) | Of sick patients, how many were caught |
| Specificity | TN / (TN+FP) | Of healthy patients, how many were correctly cleared |
| F1-score | 2·P·R / (P+R) | Balance between precision and recall |
| ROC-AUC | Area under ROC curve | Probability the model ranks a random sick patient above a random healthy one (0.5 = guessing, 1.0 = perfect) |

**Why accuracy alone is not enough.** A false negative sends a sick patient home; a false positive only triggers more tests. Accuracy treats both as equal. For screening, **recall** (few missed cases) matters more, and **ROC-AUC** shows whether the decision threshold could be lowered to catch more cases. Also, with ~61 test patients, **one patient = ~1.6 % accuracy**, so small gaps between models are not statistically meaningful. Compare against the cross-validation scores in `best_params.json`.

## 13. Graphs explained

**EDA (`data/processed/eda/`)**

1. `01_missing_values_raw.png` – missing values per column in the raw file.
2. `02_class_balance.png` – patients with / without disease.
3. `03_numerical_distributions.png` – histograms of each numerical feature split by diagnosis.
4. `04_numerical_boxplots.png` – median, spread and outliers per diagnosis.
5. `05_categorical_distributions.png` – disease counts within each category.
6. `06_correlation_heatmap.png` – correlation between numerical features and the target.

**Per model (`model_X/results/`)**

* `confusion_matrix.png` – TN, FP, FN, TP counts on the test set.
* `roc_curve.png` – true-positive rate vs false-positive rate over all thresholds; closer to the top-left corner = better.
* Logistic Regression: `coefficients.png` – red bars raise disease risk, blue bars lower it.
* Decision Tree: `decision_tree.png` (first 4 levels) and `decision_tree_rules.txt` (full tree). For one-hot features like `thal_normal <= 0.5`, the left branch means "thal is NOT normal".
* Random Forest: `feature_importance.png` – which transformed features the forest relied on most.

**Comparison (`comparison/results/`)**

* `01`–`05` – one bar chart per metric (accuracy, precision, recall, F1, ROC-AUC).
* `06_combined_roc_curves.png` – all three ROC curves on one plot.
* `07_all_metrics_comparison.png` – all metrics side by side.
* `08_confusion_matrices.png` – the three confusion matrices next to each other.

## 14. Presenting to faculty (suggested 8–10 minute flow)

1. **Problem (1 min):** binary classification of heart disease; state the educational-use warning.
2. **Dataset (1 min):** UCI Cleveland, 303 patients, 13 features, citation. Show `02_class_balance.png`.
3. **Cleaning (1 min):** open `data/processed/cleaning_report.txt` — missing `?` values, duplicates, target conversion, why outliers were kept.
4. **EDA (1 min):** show `05_categorical_distributions.png` and `06_correlation_heatmap.png`; point out the features that separate the classes.
5. **No leakage (1 min):** split first; imputation, scaling and encoding fitted inside a pipeline on training data; tuning with 5-fold CV on training data only.
6. **Live run (2 min):** run `python run_all.py --skip-eda` in the terminal.
7. **Results (2 min):** show `07_all_metrics_comparison.png`, `06_combined_roc_curves.png` and `08_confusion_matrices.png`; read the winner from the terminal.
8. **Interpretation (1 min):** compare FN counts and recall, not just accuracy; mention the small test set; show the decision tree and feature importance plots.

**Likely viva questions**

* *Why stratified split?* Keeps the disease/no-disease ratio the same in train and test.
* *Why not scale for trees?* Trees compare one feature to a threshold; scaling does not change the splits.
* *Why save the pipeline and not just the model?* New data must go through the same imputation/encoding learned from training data.
* *Why is the test set not used for tuning?* Then the reported score would be optimistically biased.
* *Why can results differ slightly between computers?* Different library versions; seeds are fixed (`random_state=42`) so results are repeatable on the same setup.

## 15. Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'sklearn'` | Activate `.venv` and run `pip install -r requirements.txt` |
| `FileNotFoundError: ... heart_disease.csv` | Run `python download_dataset.py` (or the manual steps above) |
| `No saved model at ...` | Run that model's `train.py` before `evaluate.py` |
| SSL / certificate error while downloading | Use the manual download option |
| `python` not recognised | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python` |
