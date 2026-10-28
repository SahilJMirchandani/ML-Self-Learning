# Classification Threshold Tuning — ML Virtual Lab

A hands-on Machine Learning virtual lab for the VESIT Computer Engineering curriculum.

## Focus concept

**Classification threshold tuning in Logistic Regression.** Students adjust the probability cut-off and observe its effect on the confusion matrix, accuracy, precision, recall (sensitivity), F1-score, specificity and ROC-AUC.

## What the lab demonstrates

- Trains Logistic Regression on the scikit-learn Breast Cancer Wisconsin (Diagnostic) benchmark dataset.
- Uses a stratified 75/25 train/test split with `random_state=42` (426 training and 143 test samples).
- Fits `StandardScaler` only on the training set through a scikit-learn pipeline.
- Generates held-out test probabilities once; changing the threshold changes predicted class labels without retraining.
- Treats **malignant as the positive class**, so recall is the fraction of malignant cases detected.
- Includes a live confusion matrix, metric cards, threshold presets, a threshold comparison table, individual test-case explorer and viva questions.
- ROC-AUC remains constant as the threshold is changed because it measures ranking quality across all cut-offs, rather than a single cut-off.

## Run locally

Python 3.10+ is recommended.

```bash
cd ml-threshold-lab
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000

## Method

1. Load the dataset from `sklearn.datasets.load_breast_cancer`.
2. Convert the target so malignant is 1 (positive) and benign is 0 (negative).
3. Stratify a 75/25 train/test split using random seed 42.
4. Fit feature scaling and Logistic Regression on the training data.
5. Score held-out test rows to obtain malignant probabilities.
6. Predict malignant when probability >= selected threshold.
7. Recalculate confusion-matrix counts and metrics on the same test set for each threshold.

Metrics:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)
- Specificity = TN / (TN + FP)

When a denominator is zero, the lab displays the corresponding ratio as 0.

## Endpoints

- `/` — interactive web lab
- `/api/data` — benchmark metadata and held-out predictions
- `/api/health` — lightweight health response

## Important limitation

This is an educational demonstration of classification metrics, not a clinical decision tool. It must not be used for real patient diagnosis or treatment decisions. Model performance on this split does not guarantee performance on new data.
