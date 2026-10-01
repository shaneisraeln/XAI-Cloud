"""
schemas.py
----------
Pydantic request/response models for all API endpoints.

Keeping schemas separate from main.py means:
  - Clean validation with automatic error messages
  - Easy to version or extend later
  - FastAPI auto-generates accurate Swagger docs from these
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ── Request ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input for all prediction and explanation endpoints.
    Expects exactly 30 feature values matching the Breast Cancer Wisconsin
    feature order (same order as sklearn's load_breast_cancer().feature_names).
    """
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 raw (unscaled) feature values in Breast Cancer Wisconsin order",
        examples=[[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
            0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]]
    )

    @field_validator("features")
    @classmethod
    def validate_feature_count(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected exactly 30 features, got {len(v)}")
        return v


# ── Responses ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"


class RootResponse(BaseModel):
    service: str
    model: str
    dataset: str
    status: str
    version: str
    explanation_methods: list[str]


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    inference_time_ms: float


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float


class SHAPResponse(BaseModel):
    prediction: str
    probability: float
    explanation_method: str
    features: list[str]
    contributions: list[float]        # SHAP values, sorted by |value|
    base_value: float
    inference_time_ms: float
    explanation_time_ms: float
    total_time_ms: float


class LIMERule(BaseModel):
    rule: str
    weight: float


class LIMEResponse(BaseModel):
    prediction: str
    probability: float
    explanation_method: str
    feature_rules: list[str]
    contributions: list[float]        # LIME weights
    inference_time_ms: float
    explanation_time_ms: float
    total_time_ms: float


class CombinedResponse(BaseModel):
    prediction: str
    probability: float
    # SHAP section
    shap_features: list[str]
    shap_contributions: list[float]
    shap_base_value: float
    # LIME section
    lime_feature_rules: list[str]
    lime_contributions: list[float]
    # Timings
    inference_time_ms: float
    shap_explanation_time_ms: float
    lime_explanation_time_ms: float
    total_time_ms: float
    # Overhead (calculated server-side)
    xai_overhead_ms: float
    xai_overhead_pct: float
