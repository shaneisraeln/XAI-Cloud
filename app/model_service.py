"""
model_service.py
----------------
Loads the trained RandomForest + scaler + metadata once at startup,
then exposes a single predict() function used by both the API and
the XAI services.

Design rule: the model is NEVER retrained per request.
"""

import json
import time
import warnings
import joblib
import numpy as np

from app.config import MODEL_PATH, SCALER_PATH, METADATA_PATH

# ── Suppress sklearn feature-name warning (we pass raw arrays intentionally) ─
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ── Load artifacts once (module-level, loaded when app starts) ─────────────
print("[model_service] Loading model artifacts...")
_model    = joblib.load(MODEL_PATH)
_scaler   = joblib.load(SCALER_PATH)

with open(METADATA_PATH, "r") as f:
    _metadata = json.load(f)

FEATURE_NAMES = _metadata["feature_names"]   # list of 30 strings
CLASS_NAMES   = _metadata["class_names"]     # ['malignant', 'benign']
print(f"[model_service] Ready — {_metadata['model_info']['algorithm']} loaded.")


def get_model():
    """Return the raw sklearn model (needed by SHAP/LIME services)."""
    return _model


def get_scaler():
    """Return the fitted StandardScaler (needed by SHAP/LIME services)."""
    return _scaler


def get_metadata():
    """Return the full metadata dict."""
    return _metadata


def predict(features: list[float]) -> dict:
    """
    Run inference on a single sample.

    Parameters
    ----------
    features : list of 30 floats (raw, unscaled values)

    Returns
    -------
    dict with keys:
        prediction       – class label string ('malignant' or 'benign')
        prediction_index – 0 or 1
        probability      – confidence for the predicted class (0–1)
        probabilities    – [p_malignant, p_benign]
        inference_time_ms – wall-clock time for scaling + inference
    """
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {len(features)}"
        )

    x_raw    = np.array(features, dtype=float).reshape(1, -1)

    t_start  = time.perf_counter()
    x_scaled = _scaler.transform(x_raw)
    pred_idx = int(_model.predict(x_scaled)[0])
    proba    = _model.predict_proba(x_scaled)[0].tolist()
    t_end    = time.perf_counter()

    return {
        "prediction":        CLASS_NAMES[pred_idx],
        "prediction_index":  pred_idx,
        "probability":       round(proba[pred_idx], 4),
        "probabilities":     [round(p, 4) for p in proba],
        "inference_time_ms": round((t_end - t_start) * 1000, 4),
    }
