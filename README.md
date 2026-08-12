# 🫘 Dry Bean Variety Classification

Multi-class classification of seven dry bean varieties from computer-vision measurements,
with six supervised models and an interactive Streamlit front-end.

**🔗 GitHub Repository:** https://github.com/2025ac05043/dry-bean-classification
**🚀 Live Streamlit App:** https://dry-bean-classification-dfvwawmh4b2dsij2tv9iwy.streamlit.app

---

## a. Problem statement

Dry beans are graded and priced by variety, but seven of the most common Turkish varieties
look similar enough that manual sorting is slow, subjective and inconsistent between
graders. A single mis-sorted sack changes the market value of the lot.

The task is therefore a **supervised multi-class classification problem**: given 16
morphological measurements extracted automatically from a high-resolution image of a
single grain, predict which of the seven varieties it belongs to — accurately enough to
replace a human grader, and fast enough to run inline on a sorting conveyor.

Two properties make this harder than it first looks:

1. **Class imbalance.** DERMASON contributes 3,546 grains, BOMBAY only 522 — a 6.8:1 ratio.
   Plain accuracy would reward a model that quietly ignores the rare classes, so every
   metric below is **macro-averaged** and MCC is reported as the single balanced summary.
2. **Overlapping classes.** SIRA sits geometrically between DERMASON and HOROZ, and
   BARBUNYA overlaps CALI. Nearly all residual error in every model lives in those two pairs.

## b. Dataset description

| Property | Value |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository (ID 602) — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| Also on | Kaggle — `muratkokludataset/dry-bean-dataset` |
| Instances | **13,611** (requirement: ≥ 500 ✅) |
| Features | **16** numeric, all continuous or integer (requirement: ≥ 12 ✅) |
| Target | `Class` — 7 varieties (multi-class) |
| Missing values | 0 |
| Duplicate rows | 68 (retained — they are legitimately distinct grains with identical rounded measurements) |
| Citation | Koklu, M. & Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques.* Computers and Electronics in Agriculture, 174, 105507. |

Images of 13,611 grains were captured with a high-resolution camera and a computer-vision
pipeline extracted 12 dimensional features and 4 shape factors per grain.

**Features (16):**

| # | Feature | Meaning |
|---|---|---|
| 1 | `Area` | Pixel count inside the grain boundary |
| 2 | `Perimeter` | Length of the grain border |
| 3 | `MajorAxisLength` | Longest line that can be drawn through the grain |
| 4 | `MinorAxisLength` | Longest line perpendicular to the major axis |
| 5 | `AspectRation` | MajorAxisLength / MinorAxisLength |
| 6 | `Eccentricity` | Eccentricity of the ellipse with the same moments |
| 7 | `ConvexArea` | Pixel count of the smallest convex hull |
| 8 | `EquivDiameter` | Diameter of a circle with the same area |
| 9 | `Extent` | Area / bounding-box area |
| 10 | `Solidity` | Area / ConvexArea (convexity) |
| 11 | `roundness` | 4π·Area / Perimeter² |
| 12 | `Compactness` | EquivDiameter / MajorAxisLength |
| 13–16 | `ShapeFactor1–4` | Four derived dimensionless shape descriptors |

**Target distribution:**

| Class | Count | Share |
|---|---|---|
| DERMASON | 3,546 | 26.1 % |
| SIRA | 2,636 | 19.4 % |
| SEKER | 2,027 | 14.9 % |
| HOROZ | 1,928 | 14.2 % |
| CALI | 1,630 | 12.0 % |
| BARBUNYA | 1,322 | 9.7 % |
| BOMBAY | 522 | 3.8 % |

**Preprocessing.** Labels are integer-encoded with `LabelEncoder`. The data is split
80/20 with `stratify=y, random_state=42`, giving 10,888 training and **2,723 test** rows;
that held-out slice is saved as `test_data.csv` and is exactly what the Streamlit app
scores. Feature scales differ by five orders of magnitude (`Area` ≈ 30,000 vs
`ShapeFactor2` ≈ 0.003), so Logistic Regression, kNN and Naive Bayes are wrapped in a
`Pipeline` with `StandardScaler`. Tree-based models split on thresholds and are
scale-invariant, so they are fitted on raw features. **The scaler is fitted on the
training fold only**, inside the pipeline, so no test information leaks into training.

## c. GitHub Repository Link

https://github.com/2025ac05043/dry-bean-classification

```
dry-bean-classification/
├── app.py                        # Streamlit application
├── requirements.txt              # Pinned dependencies
├── README.md                     # This file
├── test_data.csv                 # 2,723 held-out rows (upload this in the app)
└── model/
    ├── train_models.py           # Training, evaluation and cross-validation script
    ├── Dry_Bean_Dataset.csv      # Full source dataset (13,611 rows)
    ├── label_encoder.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── gradient_boosting.joblib
```

## d. Models used

Six classifiers, all trained on the same 10,888-row training split and evaluated on the
same 2,723-row held-out test split.

| # | Model | Key hyper-parameters |
|---|---|---|
| 1 | Logistic Regression | `multinomial`, `lbfgs`, `C=1.0`, `max_iter=2000`, standardised |
| 2 | Decision Tree | `gini`, `max_depth=12`, `min_samples_leaf=5` |
| 3 | k-Nearest Neighbours | `k=15`, distance weighting, Euclidean, standardised |
| 4 | Naive Bayes | Gaussian, `var_smoothing=1e-9`, standardised |
| 5 | Random Forest (Ensemble) | `n_estimators=300`, `min_samples_leaf=2` |
| 6 | Gradient Boosting (Ensemble) | Histogram-based, `max_iter=300`, `learning_rate=0.1` |

### Comparison table

All values computed on the held-out test set. Precision, Recall and F1 are **macro**
averages; AUC is **one-vs-rest macro** from `predict_proba`; MCC is the multi-class
Matthews correlation coefficient.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | 0.9948 | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.9119 | 0.9710 | 0.9262 | 0.9250 | 0.9255 | 0.8935 |
| kNN | 0.9188 | 0.9892 | 0.9350 | 0.9303 | 0.9323 | 0.9019 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Random Forest (Ensemble) | 0.9192 | 0.9940 | 0.9338 | 0.9295 | 0.9315 | 0.9023 |
| **Gradient Boosting (Ensemble)** | **0.9243** | **0.9953** | **0.9395** | **0.9350** | **0.9372** | **0.9085** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Far stronger than expected for a linear model — F1 0.9335 and the second-best AUC (0.9948), within 0.4 points of the winner. Once the features are standardised the seven varieties are close to linearly separable in 16-D space, so the extra capacity of the tree models buys very little. It also trains in ≈0.3 s and serialises to 2 KB, making it the best accuracy-per-byte model here. Its weakness is the SIRA/DERMASON boundary, which is curved rather than flat. |
| **Decision Tree** | Lowest AUC by a wide margin (0.9710) even though its accuracy (0.9119) is mid-pack. That gap is the signature of a single tree: predictions are confident and near-binary, so the probability ranking that AUC measures is coarse. Depth had to be capped at 12 with `min_samples_leaf=5`; unrestricted it memorised the training set and lost ≈2 accuracy points on test. Useful mainly for interpretability, not for deployment. |
| **kNN** | Solid and unfussy — F1 0.9323 with k=15 and distance weighting, essentially tied with Random Forest. Performance was sensitive to scaling: without `StandardScaler` accuracy collapses from 0.919 to **0.734**, because `Area` (≈10⁴) completely drowns out every shape factor (≈10⁻³) in the Euclidean distance. The practical drawback is inference cost: the model carries all 10,888 training rows and is the slowest to score a batch. |
| **Naive Bayes** | Weakest model on every metric (F1 0.9091, MCC 0.8773) and the only one below 0.90 accuracy. The cause is visible in the correlation heatmap: `Area`, `Perimeter`, `ConvexArea` and `EquivDiameter` correlate above 0.98, so the conditional-independence assumption is badly violated and the redundant size evidence is counted four times. Notably its AUC is still high (0.9916) — it ranks classes well but is poorly calibrated at the decision boundary. |
| **Random Forest (Ensemble)** | F1 0.9315, the most stable model across 5-fold CV (lowest std, 0.0035), and the only one that yields directly usable feature importances (`ShapeFactor3` 0.105, `ShapeFactor1` 0.095 and `Compactness` 0.093 dominate — shape matters more than raw size). Bagging fixes the single tree's AUC problem — 0.9710 → 0.9940 — by averaging 300 trees into smooth probabilities. Costs are real though: ≈6.5 MB on disk and the longest fit time (≈4 s), for accuracy no better than logistic regression. |
| **Gradient Boosting (Ensemble)** | **Best on all six metrics** — accuracy 0.9243, AUC 0.9953, F1 0.9372, MCC 0.9085 — while fitting in under a second thanks to histogram binning of the features. Sequential boosting targets exactly the residual SIRA/DERMASON and BARBUNYA/CALI confusions that every other model leaves on the table, which is where its margin comes from. |
| **Overall winner for this dataset** | **Gradient Boosting (Ensemble)** — it wins every metric outright, trains ≈10× faster than Random Forest, and serialises to 0.5 MB versus 6.5 MB. Caveat worth stating: the spread between the top four models is under 0.6 F1 points, which is within the noise of a single 2,723-row split. If interpretability or model size mattered more than a fraction of a point, Logistic Regression would be the pragmatic pick. |

### Sanity check — 5-fold cross-validation on the training set

Because a single 2,723-row test split can flatter a model by chance, every model was also
scored with 5-fold CV on the training fold only (reproduce with `python model/train_models.py`):

| Model | Mean F1 (macro) | Std |
|---|---|---|
| Gradient Boosting (Ensemble) | 0.9396 | 0.0069 |
| kNN | 0.9370 | 0.0038 |
| Logistic Regression | 0.9365 | 0.0063 |
| Random Forest (Ensemble) | 0.9362 | 0.0035 |
| Decision Tree | 0.9215 | 0.0057 |
| Naive Bayes | 0.9062 | 0.0048 |

The ordering agrees with the held-out table — Gradient Boosting first, Naive Bayes last —
which is reassurance that the ranking is not an artefact of one lucky split. The top four
are separated by less than one standard deviation, so they should be treated as roughly tied.

### Where the error actually is

Across every model, the confusion matrix concentrates error in two places:

- **SIRA ↔ DERMASON** — the largest single error cell in all six models. The two varieties overlap almost completely in size and differ only marginally in roundness.
- **BARBUNYA ↔ CALI** — both are large kidney-shaped beans; the separating signal is subtle shape-factor differences.

**BOMBAY is classified perfectly by the winning model** (precision = recall = 1.000 on all
104 test grains, despite being the rarest class) because its grains are several times larger than any other variety — a
reminder that a rare class is not automatically a hard class.

---

## Streamlit app features

**🚀 Live app:** https://dry-bean-classification-dfvwawmh4b2dsij2tv9iwy.streamlit.app

| Requirement | Implementation |
|---|---|
| a. Dataset upload option (CSV) | Sidebar file uploader, plus a checkbox to fall back to the bundled `test_data.csv`. Column validation and unknown-label handling included. |
| b. Model selection dropdown | Sidebar `selectbox` over all six trained models; every panel updates instantly. |
| c. Display of evaluation metrics | Six metric cards — Accuracy, AUC, Precision, Recall, F1, MCC — computed live on the uploaded data. |
| d. Confusion matrix / classification report | Seaborn heatmap (raw counts or row-normalised) side by side with the full per-class classification report. |

Additional features beyond the minimum:

- **Model comparison tab** — runs all six models on the uploaded data, ranks them, highlights the winner and charts any chosen metric.
- **Per-class ROC curves** (one-vs-rest) for the selected model.
- **Data & predictions tab** — per-row predictions with confidence, a "show only misclassified rows" filter, optional per-class probability columns, and a CSV download of the predictions.
- **Feature importance / coefficient** chart that adapts to the selected model.
- **Single-sample tab** — 16 editable numeric inputs seeded from any row, with a live probability bar chart.

## How to run locally

```bash
git clone https://github.com/2025ac05043/dry-bean-classification.git
cd dry-bean-classification
pip install -r requirements.txt

# optional — retrain everything from the raw dataset (~35 s, fully deterministic)
python model/train_models.py           # add --no-cv to skip cross-validation

streamlit run app.py
```

Then open http://localhost:8501, tick *Use bundled test_data.csv* (or upload it), and pick a model.

## Reproducibility

`random_state=42` is fixed for the split and for every stochastic model, so
`python model/train_models.py` reproduces the table above exactly. `scikit-learn` is
pinned to `1.7.2` in `requirements.txt` so the serialised `.joblib` pipelines load
without version warnings on Streamlit Cloud.

## Deployment

Deployed on Streamlit Community Cloud from the `main` branch, entry point `app.py`.

## Tech stack

Python 3.11 · scikit-learn 1.7.2 · pandas · NumPy · Matplotlib · seaborn · Streamlit · joblib
