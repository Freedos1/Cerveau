# Predicting Heart Disease Risk from Clinical Measurements

**Deliverable 1: Data Preprocessing and Initial Model Development**
Real-World AI-Driven Analytics Solution Using Python

## Overview

This project builds a binary classification model that predicts whether a
patient is likely to have heart disease from routine clinical measurements
(blood pressure, cholesterol, ECG results, exercise test outcomes, etc.),
using the [Heart Disease UCI dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Cleveland Clinic Foundation, 303 patients, 13 features).

## Contents

| File | Description |
|---|---|
| `Deliverable1_Report.md` | Written report: problem framing, dataset exploration & preprocessing, initial model development and results |
| `Heart_Disease_Analytics_Deliverable1.ipynb` | Fully executed Jupyter notebook with all code, EDA, preprocessing, and modeling |
| `data/heart.csv` | The dataset (303 rows x 14 columns, provided in this repo) |
| `figures/` | Charts exported from the notebook (target distribution, EDA plots, correlation heatmap, confusion matrices, ROC curves, feature importances) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
cd HeartDisease_AI_Analytics_Project
pip install -r requirements.txt
jupyter notebook Heart_Disease_Analytics_Deliverable1.ipynb
```

Or open it in **Google Colab**: upload `Heart_Disease_Analytics_Deliverable1.ipynb`
together with `data/heart.csv` (or mount Google Drive) and run all cells.

## Pipeline summary

1. **Load & inspect** the raw 303-row dataset; confirm no missing values,
   find and remove 1 duplicate row.
2. **EDA** — class balance, per-feature distributions by class, outlier
   check on cholesterol, correlation heatmap.
3. **Preprocess** — one-hot encode nominal categorical features
   (`cp`, `restecg`, `slope`, `thal`), stratified 80/20 train/test split,
   `StandardScaler` fit on the training data only.
4. **Model** — Logistic Regression and Random Forest baselines, evaluated
   with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC
   curves, and Random Forest feature importances.

## Headline results (held-out test set, n=61)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.820 | 0.806 | 0.879 | 0.841 | 0.868 |
| Random Forest | 0.754 | 0.737 | 0.848 | 0.789 | 0.904 |

See `Deliverable1_Report.md` for the full discussion, including limitations
and the fairness/ethical follow-ups planned for the next iteration.
