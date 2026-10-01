"""
config.py
---------
Central place for all paths and constants.
Every other module imports from here — no hardcoded paths elsewhere.
"""

import os

# ── Root paths ────────────────────────────────────────────────────────────────
# This file lives in  xai-cloud-service/app/config.py
# APP_DIR  → xai-cloud-service/app/
# BASE_DIR → xai-cloud-service/
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ── Model artifacts ───────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")

# ── API settings ──────────────────────────────────────────────────────────────
API_TITLE   = "XAI Cloud Service"
API_VERSION = "1.0.0"
API_DESC    = (
    "Explainable AI as a Cloud Service — "
    "provides Random Forest predictions with SHAP and LIME explanations "
    "for the Breast Cancer Wisconsin dataset."
)

# ── LIME settings ─────────────────────────────────────────────────────────────
LIME_NUM_FEATURES   = 10   # top features to include in LIME explanation
LIME_NUM_SAMPLES    = 1000 # perturbation samples (higher = more stable, slower)
