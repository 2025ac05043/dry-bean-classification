"""
Dry Bean Multi-Class Classification - Model Training Script
===========================================================

Dataset : UCI Dry Bean Dataset (13,611 instances x 16 features, 7 classes)
Task    : Multi-class classification of dry bean varieties

This script:
  1. Loads and inspects the dataset
  2. Splits it into a stratified 80/20 train/test set
  3. Trains 6 classification models
  4. Evaluates each with Accuracy, AUC, Precision, Recall, F1 and MCC
  5. Cross-validates every model with 5-fold CV on the training split
  6. Persists the fitted pipelines to model/*.joblib
  7. Writes model/metrics.csv and test_data.csv for the Streamlit app

Run:  python model/train_models.py          (~35 seconds, fully deterministic)
      python model/train_models.py --no-cv  (~15 seconds, skips cross-validation)
"""

import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_CSV = os.path.join(HERE, "Dry_Bean_Dataset.csv")
TARGET = "Class"


# ---------------------------------------------------------------- 1. Load data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV)
    print(f"Shape                : {df.shape}")
    print(f"Feature count        : {df.shape[1] - 1}")
    print(f"Missing values       : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows       : {int(df.duplicated().sum())}")
    print("\nClass distribution:")
    print(df[TARGET].value_counts().to_string())
    return df


# ---------------------------------------------------- 2. Model zoo definitions
def build_models() -> dict:
    """Six classifiers. Distance/gradient based learners are wrapped in a
    pipeline with StandardScaler; tree-based learners are scale-invariant so
    they are fitted on the raw features."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        multi_class="multinomial",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                (
                    "clf",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=15, weights="distance", metric="minkowski", p=2
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", GaussianNB(var_smoothing=1e-9)),
            ]
        ),
        "Random Forest (Ensemble)": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "Gradient Boosting (Ensemble)": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.1,
                        max_depth=None,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


# ------------------------------------------------------------ 3. Metric helper
def evaluate(y_true, y_pred, y_proba, labels) -> dict:
    """All averages are macro so that every one of the 7 bean classes counts
    equally, which matters because the dataset is imbalanced
    (BOMBAY = 522 rows vs DERMASON = 3,546 rows)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------- 4. Main
def main() -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate six dry bean classifiers.")
    ap.add_argument("--no-cv", action="store_true", help="Skip the 5-fold cross-validation step")
    args = ap.parse_args()

    df = load_data()

    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain / Test split   : {X_train.shape[0]} / {X_test.shape[0]}")

    # test_data.csv is what gets uploaded to the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = le.inverse_transform(y_test)
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    print(f"Wrote test_data.csv  : {test_df.shape}")

    labels = np.arange(len(le.classes_))
    rows, reports = [], {}

    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        scores = evaluate(y_test, y_pred, y_proba, labels)
        scores["Model"] = name
        scores["Fit time (s)"] = round(fit_s, 2)
        rows.append(scores)

        reports[name] = {
            "classification_report": classification_report(
                y_test, y_pred, target_names=le.classes_, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        fname = name.split(" (")[0].lower().replace(" ", "_") + ".joblib"
        # compress=3 keeps the Random Forest small enough for GitHub / Streamlit Cloud
        joblib.dump(model, os.path.join(HERE, fname), compress=3)
        print(f"  {name:<30} acc={scores['Accuracy']:.4f}  auc={scores['AUC']:.4f}  ({fit_s:.1f}s) -> {fname}")

    joblib.dump(le, os.path.join(HERE, "label_encoder.joblib"))

    results = pd.DataFrame(rows)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC", "Fit time (s)"]
    ].sort_values("F1", ascending=False)
    results.to_csv(os.path.join(HERE, "metrics.csv"), index=False)

    with open(os.path.join(HERE, "detailed_reports.json"), "w") as fh:
        json.dump(
            {"classes": le.classes_.tolist(), "reports": reports}, fh, indent=2
        )

    print("\n=== Comparison table (held-out test set) ===")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # A single 2,723-row split can flatter a model by chance, so re-check the
    # ranking with 5-fold CV on the training fold only.
    if not args.no_cv:
        print("\nRunning 5-fold cross-validation on the training split...")
        cv_rows = []
        for name, model in build_models().items():
            scores = cross_val_score(model, X_train, y_train, cv=5,
                                     scoring="f1_macro", n_jobs=-1)
            cv_rows.append({"Model": name,
                            "CV mean f1_macro": scores.mean(),
                            "CV std": scores.std()})

        cv = pd.DataFrame(cv_rows).sort_values("CV mean f1_macro", ascending=False)
        cv.to_csv(os.path.join(HERE, "cv_scores.csv"), index=False)

        print("\n=== 5-fold cross-validation ===")
        print(cv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
