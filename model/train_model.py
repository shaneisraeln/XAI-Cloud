"""
train_model.py
--------------
Phase 1 - Train and save the Random Forest classifier on the
Breast Cancer Wisconsin dataset.

Run this script once before starting the API. It produces:
  model/model.pkl       - the trained RandomForestClassifier
  model/metadata.json   - feature names, class names, and evaluation scores
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ── 1. Load dataset ──────────────────────────────────────────────────────────
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)          # 0 = malignant, 1 = benign

print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {list(data.target_names)}")   # ['malignant', 'benign']

# ── 2. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

# ── 3. Scaling ────────────────────────────────────────────────────────────────
# Random Forests are scale-invariant, but we scale anyway so the same
# scaler can be reused by SHAP/LIME in later phases without inconsistency.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 4. Train model ────────────────────────────────────────────────────────────
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,          # use all CPU cores
)
model.fit(X_train_scaled, y_train)
print("Training complete.")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall    = recall_score(y_test, y_pred, average="weighted")
f1        = f1_score(y_test, y_pred, average="weighted")
cm        = confusion_matrix(y_test, y_pred)

print("\n── Evaluation ─────────────────────────────────────────")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print("\nConfusion Matrix (rows = actual, cols = predicted):")
print(f"  Labels: {list(data.target_names)}")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# ── 6. Save model and scaler ──────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"\nModel  saved → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")

# ── 7. Save metadata ──────────────────────────────────────────────────────────
# Store everything the API will need at load time (no hardcoding in app code).
metadata = {
    "feature_names": list(data.feature_names),
    "class_names": list(data.target_names),   # index 0 = malignant, 1 = benign
    "n_features": int(X.shape[1]),
    "n_classes": int(len(data.target_names)),
    "evaluation": {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
    },
    "model_info": {
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "dataset": "Breast Cancer Wisconsin",
    },
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved → {METADATA_PATH}")

print("\nPhase 1 complete. Run verify_model.py to confirm the saved model works.")
