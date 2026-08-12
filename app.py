"""
Dry Bean Classifier - Streamlit front-end
=========================================
Interactive demo for six classification models trained on the UCI Dry Bean
Dataset (13,611 samples x 16 morphological features, 7 bean varieties).

Run locally:  streamlit run app.py
"""

import io
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------- configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model")
TARGET = "Class"

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
    "ShapeFactor4",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest.joblib",
    "Gradient Boosting (Ensemble)": "gradient_boosting.joblib",
}

BEAN_NOTES = {
    "BARBUNYA": "Large mottled cranberry bean - often confused with CALI.",
    "BOMBAY": "Rarest class (522 rows) but by far the largest grain - trivially separable.",
    "CALI": "Large white kidney bean, geometrically close to BARBUNYA.",
    "DERMASON": "Majority class (3,546 rows), small and round.",
    "HOROZ": "Elongated grain - high aspect ratio is the giveaway.",
    "SEKER": "Round white bean, high roundness / compactness.",
    "SIRA": "The hard one - sits between DERMASON and HOROZ in feature space.",
}

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .bean-hero {
          background: linear-gradient(90deg,#3f6212 0%,#65a30d 55%,#a3a635 100%);
          padding: 1.1rem 1.4rem; border-radius: 12px; color: #f7fee7;
          margin-bottom: 1.1rem;
      }
      .bean-hero h1 {margin:0; font-size:1.75rem; color:#f7fee7;}
      .bean-hero p  {margin:.35rem 0 0 0; font-size:.92rem; opacity:.92;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bean-hero">
      <h1>🫘 Dry Bean Variety Classifier</h1>
      <p>Six supervised models &bull; UCI Dry Bean Dataset &bull; 16 morphological features &bull; 7 classes</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    models, missing = {}, []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            missing.append(fname)
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    return models, encoder, missing


@st.cache_data(show_spinner=False)
def load_bundled_test_data():
    path = os.path.join(APP_DIR, "test_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data(show_spinner=False)
def load_training_metrics():
    path = os.path.join(MODEL_DIR, "metrics.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


models, encoder, missing = load_models()

if missing:
    st.error(
        "Missing model files in `model/`: "
        + ", ".join(missing)
        + ". Run `python model/train_models.py` to regenerate them."
    )
    st.stop()


# ----------------------------------------------------------------- data intake
st.sidebar.header("1 . Test data")
st.sidebar.caption(
    "Upload a CSV of test samples. Include a `Class` column to see full "
    "evaluation metrics; without it the app runs in prediction-only mode."
)
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

bundled = load_bundled_test_data()
use_bundled = st.sidebar.checkbox(
    "Use bundled test_data.csv (2,723 held-out rows)",
    value=uploaded is None,
    disabled=bundled is None,
)

data, source = None, None
if uploaded is not None and not use_bundled:
    try:
        data = pd.read_csv(uploaded)
        source = f"uploaded file - {uploaded.name}"
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read that CSV: {exc}")
elif use_bundled and bundled is not None:
    data = bundled.copy()
    source = "bundled test_data.csv"

if data is None:
    st.info("⬅️ Upload a CSV in the sidebar, or tick *Use bundled test_data.csv* to get started.")
    st.stop()

missing_cols = [c for c in FEATURES if c not in data.columns]
if missing_cols:
    st.error(
        "The uploaded file is missing these required feature columns: "
        + ", ".join(missing_cols)
    )
    st.stop()

has_labels = TARGET in data.columns
X = data[FEATURES]
y_true_raw = data[TARGET].astype(str).str.upper().str.strip() if has_labels else None

if has_labels:
    known = set(encoder.classes_)
    unknown = sorted(set(y_true_raw) - known)
    if unknown:
        st.warning(
            f"Dropping {len(unknown)} unrecognised label(s): {', '.join(unknown)}"
        )
        keep = y_true_raw.isin(known)
        X, y_true_raw, data = X[keep], y_true_raw[keep], data[keep]
    y_true = encoder.transform(y_true_raw)

# ----------------------------------------------------------- model / threshold
st.sidebar.header("2 . Model")
model_name = st.sidebar.selectbox("Choose a classifier", list(models.keys()), index=5)
model = models[model_name]

st.sidebar.header("3 . Display")
normalise_cm = st.sidebar.checkbox("Normalise confusion matrix (row %)", value=False)
show_probs = st.sidebar.checkbox("Show per-row class probabilities", value=False)

st.sidebar.divider()
st.sidebar.caption(
    f"**Source:** {source}  \n**Rows:** {len(X):,}  \n"
    f"**Mode:** {'evaluation' if has_labels else 'prediction only'}"
)

# --------------------------------------------------------------- run inference
y_pred = model.predict(X)
y_proba = model.predict_proba(X)
pred_labels = encoder.inverse_transform(y_pred)

tab_eval, tab_compare, tab_explore, tab_single = st.tabs(
    ["📊 Evaluation", "🏁 Model comparison", "🔍 Data & predictions", "🎯 Single sample"]
)

# ------------------------------------------------------------------- tab: eval
with tab_eval:
    st.subheader(f"{model_name} — held-out performance")

    if has_labels:
        labels = np.arange(len(encoder.classes_))
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC (OvR macro)": roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=labels
            ),
            "Precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.4f}")

        st.caption(
            "Macro averaging is used so each of the 7 bean classes contributes "
            "equally despite the 7:1 imbalance between DERMASON and BOMBAY."
        )

        left, right = st.columns([1.05, 1])

        with left:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(
                y_true, y_pred, labels=np.arange(len(encoder.classes_)),
                normalize="true" if normalise_cm else None,
            )
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalise_cm else "d",
                cmap="YlGn",
                cbar=False,
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                ax=ax,
                annot_kws={"size": 8},
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y_true, y_pred, target_names=encoder.classes_,
                zero_division=0, output_dict=True,
            )
            rep_df = pd.DataFrame(report).T.round(4)
            st.dataframe(rep_df, width="stretch", height=330)

            st.markdown("**Per-class ROC (one-vs-rest)**")
            fig2, ax2 = plt.subplots(figsize=(5.4, 3.9))
            for i, cls in enumerate(encoder.classes_):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_proba[:, i])
                ax2.plot(fpr, tpr, lw=1.4, label=cls)
            ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax2.set_xlabel("False positive rate")
            ax2.set_ylabel("True positive rate")
            ax2.legend(fontsize=6.5, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.info(
            "No `Class` column found, so metrics cannot be computed. "
            "Showing the predicted class distribution instead."
        )
        st.bar_chart(pd.Series(pred_labels).value_counts())

# ---------------------------------------------------------------- tab: compare
with tab_compare:
    st.subheader("All six models on the current data")

    if has_labels:
        labels = np.arange(len(encoder.classes_))
        rows = []
        for name, mdl in models.items():
            p = mdl.predict(X)
            pr = mdl.predict_proba(X)
            rows.append(
                {
                    "ML Model Name": name,
                    "Accuracy": accuracy_score(y_true, p),
                    "AUC": roc_auc_score(y_true, pr, multi_class="ovr", average="macro", labels=labels),
                    "Precision": precision_score(y_true, p, average="macro", zero_division=0),
                    "Recall": recall_score(y_true, p, average="macro", zero_division=0),
                    "F1": f1_score(y_true, p, average="macro", zero_division=0),
                    "MCC": matthews_corrcoef(y_true, p),
                }
            )
        comp = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
        st.dataframe(comp.round(4), width="stretch")
        best = comp.iloc[0]
        st.success(
            f"🏆 Best on this data: **{best['ML Model Name']}** "
            f"(F1 = {best['F1']:.4f}, MCC = {best['MCC']:.4f})"
        )

        metric_choice = st.selectbox("Chart a metric", list(comp.columns[1:]), index=4)
        chart_df = comp.set_index("ML Model Name")[[metric_choice]]
        st.bar_chart(chart_df)

        st.download_button(
            "⬇️ Download comparison table (CSV)",
            comp.to_csv(index=False).encode(),
            file_name="model_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload data with a `Class` column to compare models.")

    ref = load_training_metrics()
    if ref is not None:
        with st.expander("Reference scores from the original 80/20 training run"):
            st.dataframe(ref.round(4), width="stretch")

# ---------------------------------------------------------------- tab: explore
with tab_explore:
    st.subheader("Data preview and predictions")

    out = data.copy()
    out["Predicted"] = pred_labels
    out["Confidence"] = y_proba.max(axis=1).round(4)
    if has_labels:
        out["Correct"] = np.where(out["Predicted"] == y_true_raw.values, "✅", "❌")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{len(out):,}")
    c2.metric("Mean confidence", f"{out['Confidence'].mean():.3f}")
    if has_labels:
        c3.metric("Misclassified", f"{(out['Correct'] == '❌').sum():,}")

    if has_labels:
        only_wrong = st.checkbox("Show only misclassified rows", value=False)
        view = out[out["Correct"] == "❌"] if only_wrong else out
    else:
        view = out

    display_cols = (["Predicted", "Confidence"] + (["Class", "Correct"] if has_labels else []) + FEATURES)
    if show_probs:
        prob_df = pd.DataFrame(y_proba.round(4), columns=[f"P({c})" for c in encoder.classes_], index=out.index)
        view = view.join(prob_df)
        display_cols += list(prob_df.columns)

    st.dataframe(view[display_cols], width="stretch", height=380)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download predictions (CSV)",
        buf.getvalue().encode(),
        file_name=f"predictions_{model_name.split(' (')[0].lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

    with st.expander("Feature importance / coefficients for the selected model"):
        est = model.named_steps["clf"]
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=FEATURES).sort_values(ascending=False)
            st.bar_chart(imp)
        elif hasattr(est, "coef_"):
            imp = pd.Series(np.abs(est.coef_).mean(axis=0), index=FEATURES).sort_values(ascending=False)
            st.caption("Mean absolute standardised coefficient across the 7 one-vs-rest problems.")
            st.bar_chart(imp)
        else:
            st.info(f"{model_name} does not expose feature importances.")

# ----------------------------------------------------------------- tab: single
with tab_single:
    st.subheader("Score a single grain")
    st.caption("Values are pre-filled from a row of the current dataset — nudge them and watch the prediction move.")

    idx = st.number_input(
        "Seed the form from row #", min_value=0, max_value=len(X) - 1, value=0, step=1
    )
    seed = X.iloc[int(idx)]

    values, cols = {}, st.columns(4)
    for i, feat in enumerate(FEATURES):
        with cols[i % 4]:
            values[feat] = st.number_input(
                feat, value=float(seed[feat]), format="%.6f", key=f"in_{feat}"
            )

    sample = pd.DataFrame([values])[FEATURES]
    proba = model.predict_proba(sample)[0]
    pred = encoder.inverse_transform([int(np.argmax(proba))])[0]

    st.markdown(f"### Prediction: `{pred}`  —  {proba.max():.1%} confidence")
    st.caption(BEAN_NOTES.get(pred, ""))
    st.bar_chart(pd.Series(proba, index=encoder.classes_))

    if has_labels:
        st.caption(f"Actual label for row {int(idx)}: **{y_true_raw.iloc[int(idx)]}**")

st.divider()
st.caption(
    "Built with scikit-learn and Streamlit · UCI Dry Bean Dataset "
    "(Koklu & Ozkan, 2020) · models trained on a stratified 80/20 split, random_state=42."
)
