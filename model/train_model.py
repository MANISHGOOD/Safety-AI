import os
import pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# SIH26165 - SIF PRECURSOR DETECTION MODEL
# ============================================================

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset", "safety_reports_expanded_3018.csv"
)
MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "safety_model.pkl"
)
EVAL_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_evaluation_report.txt"
)

CLASS_LABELS = ["HIGH", "MEDIUM", "LOW"]
RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 70)
print("SIH26165 - SIF PRECURSOR DETECTION MODEL")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
print("\n[1] Loading dataset...")
data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully!")
print("Total reports:", len(data))
print("Columns:", list(data.columns))
print("\nClass distribution:")
print(data["sif_potential"].value_counts())

X = data["report_text"]
y = data["sif_potential"]

# ------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (80/20, stratified, fixed seed)
# ------------------------------------------------------------
print("\n[2] Splitting dataset 80/20 (stratified, random_state=42)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Training reports:", len(X_train))
print("Testing reports:", len(X_test))

# ------------------------------------------------------------
# 3. CREATE AI MODEL
# ------------------------------------------------------------
print("\n[3] Creating NLP + Machine Learning model...")
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )),
])

# ------------------------------------------------------------
# 4. TRAIN MODEL (fit on TRAIN only)
# ------------------------------------------------------------
print("\n[4] Training model on TRAIN split only...")
model.fit(X_train, y_train)
print("Training completed!")

# ------------------------------------------------------------
# 5. PREDICT TEST DATA (evaluate on untouched TEST only)
# ------------------------------------------------------------
print("\n[5] Testing model on held-out TEST split...")
predictions = model.predict(X_test)

# ------------------------------------------------------------
# 6. METRICS
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
precision_macro = precision_score(y_test, predictions, average="macro", zero_division=0)
recall_macro = recall_score(y_test, predictions, average="macro", zero_division=0)
f1_macro = f1_score(y_test, predictions, average="macro", zero_division=0)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Precision (macro): {precision_macro * 100:.2f}%")
print(f"Recall (macro): {recall_macro * 100:.2f}%")
print(f"F1 (macro): {f1_macro * 100:.2f}%")

print("\nClassification Report:")
report_data = classification_report(
    y_test, predictions, labels=CLASS_LABELS, zero_division=0, output_dict=True
)
print(classification_report(y_test, predictions, labels=CLASS_LABELS, zero_division=0))

high_recall = float(report_data["HIGH"]["recall"])

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)
print("\n                Predicted")
print("              HIGH  MEDIUM  LOW")
print("--------------------------------")
for i, label in enumerate(CLASS_LABELS):
    print(
        f"Actual {label:<6} "
        f"{cm[i][0]:>4}  "
        f"{cm[i][1]:>6}  "
        f"{cm[i][2]:>4}"
    )

# ------------------------------------------------------------
# 7. SAVE MODEL
# ------------------------------------------------------------
print(f"\n[6] Saving model to {MODEL_SAVE_PATH}...")
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model, f)
print("Model saved successfully.")

# ------------------------------------------------------------
# 8. SAVE EVALUATION REPORT
# ------------------------------------------------------------
print(f"\n[7] Writing evaluation report to {EVAL_REPORT_PATH}...")

train_distribution = y_train.value_counts().to_dict()
test_distribution = y_test.value_counts().to_dict()

report_lines = [
    "=" * 70,
    "SIH26165 - MODEL EVALUATION REPORT",
    "=" * 70,
    "",
    f"Dataset size overall:      {len(data)} reports",
    f"Train size (80%):          {len(X_train)} reports",
    f"Test size (20%):           {len(X_test)} reports",
    "",
    f"Train class distribution:  {train_distribution}",
    f"Test class distribution:   {test_distribution}",
    "",
    "Features:                  TF-IDF unigrams and bigrams (1,2)-gram",
    "Vectorizer:                TfidfVectorizer (lowercase, stop_words=english, min_df=1)",
    "Model:                     LogisticRegression (max_iter=2000)",
    "Pipeline:                  TfidfVectorizer -> LogisticRegression",
    f"Random seed:               {RANDOM_STATE}",
    f"Test split (percentage):   {int(TEST_SIZE * 100)}%",
    "",
    "=" * 70,
    "EVALUATION METRICS (held-out test set)",
    "=" * 70,
    f"Accuracy:                  {accuracy * 100:.2f}%",
    f"Precision (macro):         {precision_macro * 100:.2f}%",
    f"Recall (macro):            {recall_macro * 100:.2f}%",
    f"F1 score (macro):          {f1_macro * 100:.2f}%",
    f"HIGH-risk recall:          {high_recall * 100:.2f}%",
    "",
    "Per-class metrics:",
]

for cls in CLASS_LABELS:
    cls_metrics = report_data[cls]
    report_lines.append(
        f"  {cls:<8} precision={cls_metrics['precision']*100:.2f}%  "
        f"recall={cls_metrics['recall']*100:.2f}%  "
        f"f1={cls_metrics['f1-score']*100:.2f}%  "
        f"support={int(cls_metrics['support'])}"
    )

report_lines.append("")
report_lines.append("Confusion matrix (rows=actual, cols=predicted):")
col_header = " ".join(f"{cls:>8}" for cls in CLASS_LABELS)
report_lines.append(f"{'':>12} {col_header}")
for i, cls in enumerate(CLASS_LABELS):
    row = " ".join(f"{cm[i][j]:>8}" for j in range(len(CLASS_LABELS)))
    report_lines.append(f"{cls:<12} {row}")
report_lines.append("")

with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")

print("Evaluation report written successfully.")

# ------------------------------------------------------------
# 9. FINISHED
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION COMPLETED")
print("=" * 70)