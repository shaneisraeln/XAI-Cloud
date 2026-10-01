"""
shap_service.py
---------------
Wraps SHAP's TreeExplainer for the RandomForest model.

Key design decisions:
  - TreeExplainer is created ONCE and reused (expensive to initialise).
  - We explain the prediction for the PREDICTED class only, keeping the
    API response simple and focused.
  - Returns plain Python lists so FastAPI can serialise without extra work.
"""

import time
import numpy as np
import shap

from app.model_service import get_model, get_scaler, FEATURE_NAMES, CLASS_NAMES

# ── Build explainer once at import time ───────────────────────────────────────
print("[shap_service] Initialising SHAP TreeExplainer...")
_model   = get_model()
_scaler  = get_scaler()

# TreeExplainer is the fast, exact explainer for tree-based models.
# Passing the model directly (not a wrapper) gives the best speed.
_explainer = shap.TreeExplainer(_model)
print("[shap_service] SHAP TreeExplainer ready.")


def explain(features: list[float]) -> dict:
    """
    Generate a SHAP explanation for a single sample.

    Parameters
    ----------
    features : list of 30 raw (unscaled) floats

    Returns
    -------
    dict with keys:
        prediction         – predicted class label
        probability        – confidence for predicted class
        explanation_method – "SHAP"
        feature_names      – list of feature name strings
        shap_values        – SHAP value for each feature (for predicted class)
        base_value         – SHAP expected value (baseline)
        explanation_time_ms
        total_time_ms      – includes inference + SHAP computation
    """
    t_total_start = time.perf_counter()

    # ── Scale input ───────────────────────────────────────────────────────────
    x_raw    = np.array(features, dtype=float).reshape(1, -1)
    x_scaled = _scaler.transform(x_raw)

    # ── Inference (to know which class we're explaining) ──────────────────────
    pred_idx  = int(_model.predict(x_scaled)[0])
    proba     = _model.predict_proba(x_scaled)[0]

    # ── SHAP computation ──────────────────────────────────────────────────────
    t_shap_start = time.perf_counter()

    # shap_values shape for multi-class RF:
    # Older SHAP: list of n_classes arrays, each of shape (n_samples, n_features)
    # Newer SHAP: numpy array of shape (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
    shap_vals = _explainer.shap_values(x_scaled)

    t_shap_end   = time.perf_counter()
    t_total_end  = time.perf_counter()

    if isinstance(shap_vals, list):
        class_shap = shap_vals[pred_idx][0]
    elif isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3:
            if shap_vals.shape[0] == x_scaled.shape[0] and shap_vals.shape[1] == len(FEATURE_NAMES):
                class_shap = shap_vals[0, :, pred_idx]
            else:
                class_shap = shap_vals[pred_idx, 0, :]
        elif shap_vals.ndim == 2:
            class_shap = shap_vals[0]
        else:
            class_shap = np.squeeze(shap_vals)
    else:
        class_shap = np.array(shap_vals)

    # Base value (expected model output) for the predicted class
    if isinstance(_explainer.expected_value, (list, np.ndarray)):
        base_val = float(_explainer.expected_value[pred_idx])
    else:
        base_val = float(_explainer.expected_value)

    # Build feature list sorted by absolute SHAP value (most impactful first)
    paired = sorted(
        zip(FEATURE_NAMES, class_shap.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    sorted_features   = [p[0] for p in paired]
    sorted_shap_vals  = [round(p[1], 6) for p in paired]

    explanation_ms = round((t_shap_end  - t_shap_start)  * 1000, 4)
    total_ms       = round((t_total_end - t_total_start) * 1000, 4)

    return {
        "prediction":          CLASS_NAMES[pred_idx],
        "probability":         round(float(proba[pred_idx]), 4),
        "explanation_method":  "SHAP",
        "feature_names":       sorted_features,
        "shap_values":         sorted_shap_vals,
        "base_value":          round(base_val, 6),
        "explanation_time_ms": explanation_ms,
        "total_time_ms":       total_ms,
    }
