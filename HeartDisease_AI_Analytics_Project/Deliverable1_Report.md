# Deliverable 1: Data Preprocessing and Initial Model Development

**Project title:** Predicting Heart Disease Risk from Clinical Measurements
**Course:** Real-World AI-Driven Analytics Solution Using Python
**Dataset:** Heart Disease UCI (Cleveland Clinic Foundation)
**Author:** Fred Kabore
**Companion notebook:** [`Heart_Disease_Analytics_Deliverable1.ipynb`](Heart_Disease_Analytics_Deliverable1.ipynb)

---

## 1. Introduction & Problem Framing

Cardiovascular disease is the leading cause of death worldwide, and a large
share of that burden is preventable if at-risk patients are identified early
enough for lifestyle changes, medication, or further diagnostic testing.
Clinics already collect a set of inexpensive, routine measurements for every
patient — resting blood pressure, cholesterol, resting ECG results, exercise
test outcomes, and so on — but converting those raw numbers into an
actionable risk signal still depends heavily on manual physician judgment.
This project frames that gap as a **binary classification problem**: given a
patient's clinical measurements, predict whether the patient is likely to
have heart disease, so that the prediction can serve as a decision-support
signal that helps clinicians prioritize scarce diagnostic resources (such as
angiography) toward the patients most likely to need them.

The primary stakeholders are (1) primary-care physicians and cardiologists,
who would use the model's output to support — never replace — their own
diagnosis; (2) patients, whose health outcomes and personal medical data are
directly at stake; and (3) hospital administrators, who are responsible for
allocating limited diagnostic capacity efficiently. Because a **false
negative** (telling an at-risk patient they are healthy) is far more costly
in this domain than a false positive (sending a healthy patient for an extra
test), the project's success criteria emphasize **recall on the positive
(disease) class** and overall discriminative power (ROC-AUC), rather than
accuracy alone.

The **Heart Disease UCI dataset** (Cleveland Clinic Foundation, 303 patients,
13 clinical features plus a binary diagnosis label) was chosen because it is
a well-documented, widely used benchmark that is small enough to explore
exhaustively in this deliverable, while still containing the kinds of
real-world messiness — mixed feature scales, categorical codes, mild class
imbalance, and a duplicate record — that a genuine analytics pipeline has to
handle correctly.

## 2. Dataset Exploration & Preprocessing

### 2.1 Structure and data quality

The raw dataset contains 303 rows and 14 columns: 13 predictor features
(`age`, `sex`, `cp` — chest pain type, `trestbps` — resting blood pressure,
`chol` — serum cholesterol, `fbs` — fasting blood sugar flag, `restecg` —
resting ECG result, `thalach` — maximum heart rate achieved, `exang` —
exercise-induced angina, `oldpeak` — ST depression induced by exercise,
`slope` — slope of the peak exercise ST segment, `ca` — number of major
vessels colored by fluoroscopy, `thal` — thalassemia test result) and one
binary label, `target` (1 = heart disease present, 0 = absent).

A systematic data-quality pass found:

- **Zero missing values** across all 14 columns — a known property of this
  particular UCI release, since rows with missing values were already
  excluded upstream by the dataset curators.
- **One exact duplicate row.** This was removed before splitting the data,
  reducing the working set to 302 rows, so that the same patient record
  could not appear in both the training and test partitions and inflate the
  apparent test performance.
- **A roughly balanced target:** 54.4% positive (disease) vs. 45.6% negative
  after de-duplication (164 vs. 138 patients), which is close enough to
  balanced that no resampling technique (e.g., SMOTE) was needed for this
  first modeling pass.
- **A handful of outliers in `chol`** (serum cholesterol): using the
  standard 1.5×IQR rule, five patients sit above the upper fence of
  ~370 mg/dl. These were kept rather than dropped, since unusually high
  cholesterol is clinically real and potentially predictive information,
  not a data-entry error — but it does motivate scaling features before
  feeding them into a distance/gradient-sensitive model.

### 2.2 Exploratory data analysis

Visual exploration (see `figures/target_distribution.png`,
`figures/numeric_distributions.png`, `figures/chol_boxplot.png`, and
`figures/correlation_heatmap.png` in the companion notebook) showed that:

- The two classes are visibly separated along `cp` (chest pain type),
  `thalach` (maximum heart rate achieved), `exang` (exercise-induced
  angina), and `oldpeak` (ST depression) — these were expected, on
  clinical grounds and on the correlation matrix, to be among the more
  predictive features.
- No pair of features showed a correlation strong enough (|r| > 0.9) to
  raise multicollinearity concerns for this initial pass.
- Feature scales vary enormously — `chol` ranges from 126 to 564 while
  `oldpeak` ranges from 0 to 6.2 — which would bias any
  distance-or-gradient-based model (like Logistic Regression) toward the
  large-magnitude features unless the data is standardized first.

### 2.3 Preprocessing pipeline

Based on the findings above, the following preprocessing steps were applied,
in order, before any model was trained:

1. **De-duplication** — the one exact duplicate row was dropped.
2. **Categorical encoding** — `cp`, `restecg`, `slope`, and `thal` are
   *nominal* codes (their numeric values do not represent an ordinal
   magnitude), so they were one-hot encoded with `pandas.get_dummies`
   (dropping the first level of each to avoid the dummy-variable trap).
   `sex`, `fbs`, and `exang` are already binary and were left as-is. This
   expanded the feature matrix from 13 to 19 columns.
3. **Stratified train/test split** — an 80/20 split, stratified on
   `target`, was used so that both partitions preserve the original
   ~54/46 class balance (241 training rows, 61 test rows).
4. **Feature scaling** — `StandardScaler` was fit on the training set only
   (to avoid leaking test-set statistics into training) and then applied
   to both the training and test sets, for use by the Logistic Regression
   baseline. The Random Forest baseline was trained on the unscaled
   features, since tree-based splits are invariant to monotonic scaling.

## 3. Initial Model Development

Two baseline classifiers were trained on the preprocessed data: a
**Logistic Regression** model (as an interpretable linear reference point)
and a **Random Forest** (as a non-linear ensemble comparison and a source of
feature-importance rankings). Both were evaluated on the held-out 20% test
set using accuracy, precision, recall, F1-score, and ROC-AUC, since
accuracy alone would obscure the recall/precision trade-off that matters
most in this domain.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.820 | 0.806 | 0.879 | 0.841 | 0.868 |
| Random Forest | 0.754 | 0.737 | 0.848 | 0.789 | 0.904 |

Both models perform well above chance on this small, fairly separable
dataset. Logistic Regression achieved the higher accuracy, precision, and F1
on this split, and its coefficients are directly interpretable as
(standardized) log-odds effects — useful for a clinical audience that wants
to know *why* a prediction was made. The Random Forest achieved the higher
ROC-AUC (0.904 vs. 0.868), suggesting it ranks patients by risk slightly
more reliably across all thresholds even though its default-threshold
accuracy was lower on this particular split. Both models' recall (0.879 and
0.848, respectively) is comfortably above their precision, which is the
desired direction given that false negatives are costlier than false
positives in this domain.

The Random Forest's top features by importance — `thal` (thalassemia
result), `oldpeak` (ST depression), `thalach` (max heart rate), `chol`
(cholesterol), and `ca` (vessels colored by fluoroscopy) — line up well with
both clinical intuition and the correlation heatmap from Section 2.2, which
is a useful sanity check that the model has learned genuine physiological
signal rather than noise in a 302-row dataset.

**Limitations of this initial pass**, to be addressed in later iterations of
the project: no hyperparameter tuning or cross-validation has been performed
yet, so the single 80/20 split is more sensitive to sampling noise than a
k-fold estimate would be; class-imbalance handling is minimal (justified
only because the imbalance is mild); and no fairness audit (e.g., checking
whether recall differs meaningfully across `sex` or `age` subgroups) has yet
been performed. Because this model would ultimately support real clinical
decisions, that fairness and calibration audit — along with a clear
statement that the model's output is a decision-support signal and never a
standalone diagnosis — must be completed before any deployment is
considered.

---

### Tools used
`pandas`, `NumPy` for data manipulation; `Matplotlib`, `Seaborn` for
visualization; `scikit-learn` for preprocessing, modeling, and evaluation.
The notebook was developed and can be re-run in Jupyter or Google Colab
(upload `data/heart.csv` to the Colab runtime, or mount Google Drive, and
adjust the read path if needed).
