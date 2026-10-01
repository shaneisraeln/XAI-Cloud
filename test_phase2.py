"""
test_phase2.py
--------------
Phase 2 verification script.

Tests:
  1. model_service  — predict()
  2. shap_service   — explain()
  3. lime_service   — explain()

Run from xai-cloud-service/ with:
    python test_phase2.py

Expected: all three sections print results without errors.
"""

import sys
import os

# Allow imports from app/ without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Sample: first record from Breast Cancer dataset (known: malignant) ────────
SAMPLE = [
    1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01, 2.776e-01,
    3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02, 1.095e+00, 9.053e-01,
    8.589e+00, 1.534e+02, 6.399e-03, 4.904e-02, 5.373e-02, 1.587e-02,
    3.003e-02, 6.193e-03, 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03,
    1.622e-01, 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01,
]

print("=" * 60)
print("Phase 2 — XAI Engine Verification")
print("=" * 60)

# ── 1. Model Service ──────────────────────────────────────────────────────────
print("\n[1/3] Testing model_service.predict()")
from app.model_service import predict

result = predict(SAMPLE)
print(f"  Prediction   : {result['prediction'].upper()}")
print(f"  Confidence   : {result['probability'] * 100:.2f}%")
print(f"  Probabilities: malignant={result['probabilities'][0]}, benign={result['probabilities'][1]}")
print(f"  Inference    : {result['inference_time_ms']} ms")
assert result["prediction"] == "malignant", "Unexpected prediction!"
print("  ✓ PASS")

# ── 2. SHAP Service ───────────────────────────────────────────────────────────
print("\n[2/3] Testing shap_service.explain()")
from app.shap_service import explain as shap_explain

shap_result = shap_explain(SAMPLE)
print(f"  Prediction      : {shap_result['prediction'].upper()}")
print(f"  Confidence      : {shap_result['probability'] * 100:.2f}%")
print(f"  SHAP time       : {shap_result['explanation_time_ms']} ms")
print(f"  Total time      : {shap_result['total_time_ms']} ms")
print(f"  Top 5 features  :")
for name, val in zip(shap_result["feature_names"][:5], shap_result["shap_values"][:5]):
    direction = "↑ pushes toward malignant" if val > 0 else "↓ pushes toward benign"
    print(f"    {name:<30} SHAP={val:+.4f}  {direction}")
assert len(shap_result["shap_values"]) == 30, "Expected 30 SHAP values"
print("  ✓ PASS")

# ── 3. LIME Service ───────────────────────────────────────────────────────────
print("\n[3/3] Testing lime_service.explain()")
from app.lime_service import explain as lime_explain

lime_result = lime_explain(SAMPLE)
print(f"  Prediction      : {lime_result['prediction'].upper()}")
print(f"  Confidence      : {lime_result['probability'] * 100:.2f}%")
print(f"  LIME time       : {lime_result['explanation_time_ms']} ms")
print(f"  Total time      : {lime_result['total_time_ms']} ms")
print(f"  Top 5 rules     :")
for rule, weight in zip(lime_result["feature_rules"][:5], lime_result["lime_weights"][:5]):
    direction = "↑ malignant" if weight > 0 else "↓ benign"
    print(f"    {rule:<40} weight={weight:+.4f}  {direction}")
assert len(lime_result["feature_rules"]) == 10, "Expected 10 LIME rules"
print("  ✓ PASS")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("XAI Overhead Analysis (this single sample):")
baseline_ms = result["inference_time_ms"]
shap_ms     = shap_result["total_time_ms"]
lime_ms     = lime_result["total_time_ms"]
print(f"  Baseline inference : {baseline_ms:.2f} ms")
print(f"  SHAP total         : {shap_ms:.2f} ms  (overhead: +{shap_ms - baseline_ms:.2f} ms)")
print(f"  LIME total         : {lime_ms:.2f} ms  (overhead: +{lime_ms - baseline_ms:.2f} ms)")
print("\nAll Phase 2 tests PASSED. Ready for Phase 3 (FastAPI).")
