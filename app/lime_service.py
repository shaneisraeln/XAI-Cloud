"""
lime_service.py
---------------
Wraps LIME's LimeTabularExplainer for the RandomForest model.

Key design decisions:
  - LimeTabularExplainer is created ONCE and reused.
  - We use the training data distribution (reconstructed from the dataset)
    so LIME's perturbation is statistically grounded.
  - The explainer wraps the model's predict_proba on SCALED input,
    because the model was trained on scaled data.
  - Returns plain Python types for easy JSON serialisation.
"""

import time
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_breast_cancer

from app.config import LIME_NUM_FEATURES, LIME_NUM_SAMPLES
from app.model_service import get_model, get_scaler, FEATURE_NAMES, CLASS_NAMES

# ── Build explainer once at import time ───────────────────────────────────────
print("[lime_service] Initialising LIME LimeTabularExplainer...")

_model  = get_model()
_scaler = get_scaler()

# Load the original dataset to give LIME the training distribution.
# This is read-only metadata usage — the model is NOT retrained.
_data      = load_breast_cancer()
_X_train   = _scaler.transform(_data.data)   # use scaled space to match model

# The explainer works in SCALED space because our predict function
# receives already-scaled input.
_explainer = LimeTabularExplainer(
    training_data  = _X_train,
    feature_names  = FEATURE_NAMES,
    class_names    = CLASS_NAMES,
    mode           = "classification",
    discretize_continuous = True,
    random_state   = 42,
)
print("[lime_service] LIME LimeTabularExplainer ready.")


def _predict_fn(x: np.ndarray) -> np.ndarray:
    """
    Prediction function passed to LIME.
    Input x is already in scaled space (LIME perturbs around the scaled point).
    """
    return _model.predict_proba(x)


def explain(features: list[float]) -> dict:
    """
    Generate a LIME explanation for a single sample.

    Parameters
    ----------
    features : list of 30 raw (unscaled) floats

    Returns
    -------
    dict with keys:
        prediction          – predicted class label
        probability         – confidence for predicted class
        explanation_method  – "LIME"
        feature_rules       – list of rule strings (e.g. "worst radius > 16.8")
        lime_weights        – corresponding contribution weights
        explanation_time_ms
        total_time_ms
    """
    t_total_start = time.perf_counter()

    # ── Scale input ───────────────────────────────────────────────────────────
    x_raw    = np.array(features, dtype=float).reshape(1, -1)
    x_scaled = _scaler.transform(x_raw)

    # ── Inference ─────────────────────────────────────────────────────────────
    pred_idx = int(_model.predict(x_scaled)[0])
    proba    = _model.predict_proba(x_scaled)[0]

    # ── LIME computation ──────────────────────────────────────────────────────
    t_lime_start = time.perf_counter()

    explanation = _explainer.explain_instance(
        data_row        = x_scaled[0],          # 1D array in scaled space
        predict_fn      = _predict_fn,
        num_features    = LIME_NUM_FEATURES,
        num_samples     = LIME_NUM_SAMPLES,
        labels          = (pred_idx,),           # explain the predicted class
    )

    t_lime_end  = time.perf_counter()
    t_total_end = time.perf_counter()

    # Extract feature rules and weights for the predicted class
    lime_list = explanation.as_list(label=pred_idx)   # [(rule_str, weight), ...]

    feature_rules = [item[0] for item in lime_list]
    lime_weights  = [round(item[1], 6) for item in lime_list]

    explanation_ms = round((t_lime_end  - t_lime_start)  * 1000, 4)
    total_ms       = round((t_total_end - t_total_start) * 1000, 4)

    return {
        "prediction":          CLASS_NAMES[pred_idx],
        "probability":         round(float(proba[pred_idx]), 4),
        "explanation_method":  "LIME",
        "feature_rules":       feature_rules,
        "lime_weights":        lime_weights,
        "explanation_time_ms": explanation_ms,
        "total_time_ms":       total_ms,
    }
