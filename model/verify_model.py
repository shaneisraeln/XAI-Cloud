"""
verify_model.py
---------------
Phase 1 - Load the saved model and scaler, then make a sample prediction.

Purpose: confirm that model.pkl and scaler.pkl were saved correctly and
that the inference pipeline works end-to-end before building the API.

Run AFTER train_model.py.
"""

import json
import os
import joblib
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ── Load artifacts ────────────────────────────────────────────────────────────
print("Loading saved artifacts...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]
class_names   = metadata["class_names"]

print(f"  Model   : {metadata['model_info']['algorithm']}")
print(f"  Dataset : {metadata['model_info']['dataset']}")
print(f"  Classes : {class_names}")
print(f"  Saved accuracy: {metadata['evaluation']['accuracy']}")

# ── Sample input ──────────────────────────────────────────────────────────────
# This is the FIRST sample from the Breast Cancer dataset (known label: malignant).
# Values taken directly from sklearn so results are reproducible.
sample_raw = np.array([[
    1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
    3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
    8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
    3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
    1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01
]])

# ── Inference pipeline ────────────────────────────────────────────────────────
sample_scaled = scaler.transform(sample_raw)
prediction_idx = model.predict(sample_scaled)[0]          # 0 or 1
probabilities  = model.predict_proba(sample_scaled)[0]    # [p_malignant, p_benign]

predicted_class = class_names[prediction_idx]
confidence      = probabilities[prediction_idx]

# ── Output ────────────────────────────────────────────────────────────────────
print("\n── Sample Prediction ───────────────────────────────────")
print(f"  Predicted class : {predicted_class.upper()}")
print(f"  Confidence      : {confidence * 100:.2f}%")
print(f"  Probabilities   : malignant={probabilities[0]:.4f}, benign={probabilities[1]:.4f}")
print(f"  Expected        : MALIGNANT  (sample #0 in the dataset)")

status = "PASS" if predicted_class == "malignant" else "CHECK (unexpected result)"
print(f"\n  Verification    : {status}")
print("\nModel is ready for Phase 2 (XAI Engine).")
