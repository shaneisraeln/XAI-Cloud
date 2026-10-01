# Technical Requirements Document (TRD)

## Project Title
Explainable AI (XAI) as a Cloud Service for Black-Box Model Auditing

## Document Information
| Field | Detail |
|-------|--------|
| Author | M.Sc. AI & ML Student |
| Course | Cloud Computing Lab Mini Project |
| Version | 1.0 |
| Status | Complete |

---

## 1. System Architecture

```
User (Browser)
      │
      ▼
Frontend Dashboard  (HTML + CSS + JS)
      │  HTTP fetch()
      ▼
FastAPI REST API  ── app/main.py
      │
      ├── GET  /              → Service info
      ├── GET  /health        → Readiness probe
      ├── POST /predict       → Baseline inference
      ├── POST /explain/shap  → Inference + SHAP
      ├── POST /explain/lime  → Inference + LIME
      └── POST /explain       → Inference + SHAP + LIME
      │
      ▼
Model Service  ── app/model_service.py
      │  loads once at startup
      ├── RandomForestClassifier  (model/model.pkl)
      └── StandardScaler          (model/scaler.pkl)
      │
      ├── SHAP Service  ── app/shap_service.py
      │     └── shap.TreeExplainer  (singleton)
      │
      └── LIME Service  ── app/lime_service.py
            └── LimeTabularExplainer  (singleton)
      │
      ▼
Docker Container
      │
      ▼
Kubernetes (Minikube)
      ├── Deployment  (1–3 replicas)
      ├── Service     (NodePort)
      └── HPA         (optional)
```

---

## 2. Technology Stack

### Machine Learning
| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11 |
| ML Framework | scikit-learn | 1.6.1 |
| Data Processing | pandas | 2.2.3 |
| Numerical Computing | numpy | 2.2.3 |
| Model Persistence | joblib | 1.4.2 |

### Explainable AI
| Component | Technology | Version |
|-----------|-----------|---------|
| SHAP Explainer | shap | 0.46.0 |
| LIME Explainer | lime | 0.2.0.1 |

### Backend API
| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | 0.115.6 |
| ASGI Server | Uvicorn | 0.34.0 |
| Data Validation | Pydantic | 2.10.6 |
| Static Files | aiofiles | 23.2.1 |

### Frontend
| Component | Technology |
|-----------|-----------|
| Markup | HTML5 |
| Styling | CSS3 (custom, no framework) |
| Logic | Vanilla JavaScript (ES2020) |
| Charts | Custom SVG-based bar charts (no chart library) |

### Containerization & Deployment
| Component | Technology | Version |
|-----------|-----------|---------|
| Container Runtime | Docker Desktop | Latest |
| Base Image | python:3.11-slim | - |
| Orchestration | Kubernetes | - |
| Local Cluster | Minikube | Latest |

### Testing & Performance
| Component | Technology | Version |
|-----------|-----------|---------|
| Load Testing | Locust | 2.32.4 |
| HTTP Client | requests | 2.32.3 |
| Visualization | matplotlib + seaborn | 3.10.1 / 0.13.2 |

---

## 3. Machine Learning Pipeline

### Dataset
- **Name**: Breast Cancer Wisconsin (Diagnostic)
- **Source**: `sklearn.datasets.load_breast_cancer()`
- **Samples**: 569
- **Features**: 30 numerical clinical measurements
- **Classes**: 0 = malignant, 1 = benign
- **Class distribution**: 212 malignant, 357 benign

### Training Pipeline
```
load_breast_cancer()
      │
      ▼
train_test_split(test_size=0.20, stratify=y, random_state=42)
      │
      ▼
StandardScaler.fit_transform(X_train)
      │
      ▼
RandomForestClassifier(n_estimators=100, random_state=42)
      │
      ▼
joblib.dump(model, model.pkl)
joblib.dump(scaler, scaler.pkl)
json.dump(metadata, metadata.json)
```

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 95.61% |
| Precision | 95.61% |
| Recall | 95.61% |
| F1 Score | 95.60% |

### Key Design Decision
The model is loaded **once at startup** into a module-level singleton. No retraining occurs per request. The scaler is also loaded once and reused. This is critical for performance — loading a model on every request would make latency measurements meaningless.

---

## 4. XAI Engine Design

### SHAP — SHapley Additive exPlanations
- **Explainer type**: `shap.TreeExplainer`
- **Why TreeExplainer**: Exact, fast algorithm for tree-based models. Does not require sampling.
- **Output**: One SHAP value per feature, representing its contribution to the predicted class
- **Singleton**: Created once at module import time, reused for every request
- **Explanation space**: Scaled input space (same as model training)
- **Output format**: Features sorted by absolute SHAP value, most impactful first

### LIME — Local Interpretable Model-agnostic Explanations
- **Explainer type**: `LimeTabularExplainer`
- **Why LIME**: Produces human-readable rule strings. Complements SHAP.
- **Background data**: Full Breast Cancer dataset (scaled) used as training distribution
- **Perturbation samples**: 1000 per explanation (configurable via `LIME_NUM_SAMPLES`)
- **Output features**: Top 10 feature rules (configurable via `LIME_NUM_FEATURES`)
- **Singleton**: Created once at module import time
- **Key difference from SHAP**: LIME is model-agnostic (uses `predict_proba` wrapper) while SHAP uses the tree structure directly — this is why LIME is slower

### Overhead Calculation
```
baseline_ms = inference_time_ms          (POST /predict)
shap_total  = inference + shap_time      (POST /explain/shap)
lime_total  = inference + lime_time      (POST /explain/lime)

overhead_ms  = total_ms - baseline_ms
overhead_pct = (overhead_ms / baseline_ms) × 100
```

---

## 5. API Specification

### Base URL
`http://localhost:8000` (local) or `http://<minikube-ip>:<nodeport>` (Kubernetes)

### Endpoints

#### GET /
Returns service information.
```json
{
  "service": "XAI Cloud Service",
  "model": "RandomForestClassifier",
  "dataset": "Breast Cancer Wisconsin",
  "status": "running",
  "version": "1.0.0",
  "explanation_methods": ["SHAP", "LIME"]
}
```

#### GET /health
Kubernetes readiness and liveness probe.
```json
{ "status": "healthy" }
```

#### POST /predict
**Request body**:
```json
{ "features": [30 floats in Breast Cancer Wisconsin feature order] }
```
**Response**:
```json
{
  "prediction": "malignant",
  "probability": 0.97,
  "inference_time_ms": 1.24
}
```

#### POST /explain/shap
**Response**:
```json
{
  "prediction": "malignant",
  "probability": 0.97,
  "explanation_method": "SHAP",
  "features": ["worst concave points", "worst radius", "..."],
  "contributions": [0.182, 0.143, "..."],
  "base_value": 0.372,
  "inference_time_ms": 1.24,
  "explanation_time_ms": 18.5,
  "total_time_ms": 21.3
}
```

#### POST /explain/lime
**Response**:
```json
{
  "prediction": "malignant",
  "probability": 0.97,
  "explanation_method": "LIME",
  "feature_rules": ["worst radius > 16.80", "..."],
  "contributions": [0.312, "..."],
  "inference_time_ms": 1.24,
  "explanation_time_ms": 423.8,
  "total_time_ms": 426.1
}
```

#### POST /explain
**Response includes all of the above plus**:
```json
{
  "shap_explanation_time_ms": 18.5,
  "lime_explanation_time_ms": 423.8,
  "total_time_ms": 447.2,
  "xai_overhead_ms": 445.9,
  "xai_overhead_pct": 35960.0
}
```

### Input Validation
- Exactly 30 float values required
- Validated by Pydantic `field_validator`
- Returns HTTP 422 with descriptive message on invalid input

---

## 6. Project File Structure

```
xai-cloud-service/
│
├── app/
│   ├── __init__.py          # Package marker
│   ├── config.py            # All paths and constants
│   ├── model_service.py     # Model loading + predict()
│   ├── shap_service.py      # SHAP singleton + explain()
│   ├── lime_service.py      # LIME singleton + explain()
│   ├── schemas.py           # Pydantic request/response models
│   └── main.py              # FastAPI app + all routes
│
├── model/
│   ├── train_model.py       # One-time training script
│   ├── verify_model.py      # Post-training verification
│   ├── model.pkl            # Trained RandomForest (generated)
│   ├── scaler.pkl           # Fitted StandardScaler (generated)
│   └── metadata.json        # Feature names, class names, metrics
│
├── frontend/
│   ├── index.html           # Dashboard layout
│   ├── style.css            # Dark theme styling
│   └── script.js            # API calls + chart rendering
│
├── tests/
│   └── load_test.py         # Locust load testing script
│
├── results/                 # Generated graphs and CSVs
│
├── k8s/
│   ├── deployment.yaml      # Kubernetes Deployment
│   ├── service.yaml         # Kubernetes NodePort Service
│   └── hpa.yaml             # Horizontal Pod Autoscaler
│
├── docs/
│   ├── PRD.md               # This document's companion
│   └── TRD.md               # This document
│
├── Dockerfile               # Container definition
├── .dockerignore            # Docker build exclusions
├── docker-compose.yml       # Local multi-container config
├── requirements.txt         # Pinned Python dependencies
└── .gitignore               # Git exclusions
```

---

## 7. Docker Configuration

### Base Image
`python:3.11-slim` — minimal Debian-based Python image. Chosen for:
- Smaller image size than `python:3.11`
- Full `apt-get` access for C++ build tools (needed by SHAP)
- Broad wheel availability for Python 3.11

### Build Process
1. Install system C++ build tools (`gcc`, `g++`, `build-essential`)
2. Copy `requirements.txt` and install all Python packages
3. Copy `app/`, `frontend/`, `model/` directories
4. Expose port 8000
5. Start Uvicorn with 1 worker

### Why 1 Worker
SHAP and LIME explainers are module-level singletons. Multiple workers would each load their own copy, multiplying memory usage. One worker keeps explainer state predictable and memory bounded.

---

## 8. Kubernetes Configuration

### Deployment
```yaml
replicas: 1        # Start with 1, scale to 2 and 3 for experiments
image: xai-cloud-service:latest
containerPort: 8000
resources:
  requests: { cpu: "250m", memory: "512Mi" }
  limits:   { cpu: "1000m", memory: "1Gi" }
readinessProbe: GET /health
livenessProbe:  GET /health
```

### Service
```yaml
type: NodePort
port: 8000
targetPort: 8000
```

### Scaling Experiment
| Replicas | Purpose |
|----------|---------|
| 1 | Baseline Kubernetes performance |
| 2 | First scale step |
| 3 | Load distribution across pods |

---

## 9. Performance Experiment Design

### Experiment Variables
| Variable | Values |
|----------|--------|
| Request volume | 10, 50, 100, 500, 1000 |
| Endpoint | /predict, /explain/shap, /explain/lime, /explain |
| Replicas | 1, 2, 3 |

### Metrics Collected
| Metric | Tool |
|--------|------|
| Latency (avg, min, max, median) | Locust / requests |
| Throughput (req/sec) | Locust |
| CPU utilization | kubectl top pods |
| Memory utilization | kubectl top pods |
| XAI overhead % | Calculated from timings |

### Output Graphs
1. Request load vs average latency (baseline vs SHAP vs LIME)
2. Request load vs throughput
3. Replicas vs latency
4. Replicas vs throughput
5. Baseline vs SHAP vs LIME latency comparison (bar chart)

---

## 10. Security and Limitations

### Security Notes
- CORS is set to `allow_origins=["*"]` — acceptable for a local lab project, not for production
- No authentication — this is an academic prototype
- Model artifacts are copied into the container — acceptable for this use case

### Known Limitations
- Kubernetes deployment is local (Minikube), not a real cloud provider
- LIME is non-deterministic per run (controlled with `random_state=42`)
- Single-worker server limits throughput for high concurrency
- SHAP overhead is low; LIME overhead is significantly higher — this is a fundamental property of the algorithms, not a bug
