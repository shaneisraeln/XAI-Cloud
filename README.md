# XAI Cloud Service — Black-Box Model Auditing

M.Sc. AI & ML — Cloud Computing Lab Mini Project

Explainable AI as a Cloud Service. Provides Random Forest predictions with
SHAP and LIME explanations, deployed on Kubernetes with measured performance.

---

## Research Question

> How much computational overhead does explainability (SHAP and LIME) introduce
> over baseline model inference, and how does a Kubernetes-deployed XAI service
> behave under increasing workloads?

---

## Architecture

```
User (Browser)
      ↓
Frontend Dashboard  (HTML + CSS + JS)
      ↓
FastAPI REST API  (port 8000)
      ↓
Model Service → RandomForest (model.pkl)
      ↓
XAI Engine
  ├── SHAP  (TreeExplainer)
  └── LIME  (LimeTabularExplainer)
      ↓
Docker Container
      ↓
Kubernetes (Minikube)  — 1 to 3 replicas
```

---

## Project Structure

```
xai-cloud-service/
├── app/
│   ├── config.py          # paths and constants
│   ├── model_service.py   # model loading + predict()
│   ├── shap_service.py    # SHAP singleton + explain()
│   ├── lime_service.py    # LIME singleton + explain()
│   ├── schemas.py         # Pydantic request/response models
│   └── main.py            # FastAPI routes
├── model/
│   ├── train_model.py     # one-time training script
│   ├── verify_model.py    # post-training verification
│   ├── model.pkl          # trained RandomForest
│   ├── scaler.pkl         # fitted StandardScaler
│   └── metadata.json      # feature names, metrics
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   └── load_test.py       # performance testing script
├── results/               # generated graphs and CSV
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── docs/
│   ├── PRD.md
│   └── TRD.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start (Local)

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python model/train_model.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/dashboard

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Service info |
| GET | `/health` | Kubernetes probe |
| POST | `/predict` | Baseline inference only |
| POST | `/explain/shap` | Inference + SHAP |
| POST | `/explain/lime` | Inference + LIME |
| POST | `/explain` | Inference + SHAP + LIME + overhead % |

Swagger UI: http://localhost:8000/docs

---

## Docker

```cmd
docker build -t xai-cloud-service .
docker run -d -p 8000:8000 --name xai-app xai-cloud-service
docker logs xai-app
docker stop xai-app && docker rm xai-app
```

---

## Kubernetes (Minikube)

```cmd
minikube start --driver=docker
minikube image load xai-cloud-service:latest
kubectl apply -f k8s/
kubectl get pods
kubectl get services
minikube service xai-service
```

Scale:
```cmd
kubectl scale deployment xai-service --replicas=2
kubectl scale deployment xai-service --replicas=3
```

---

## Experimental Results

### XAI Overhead (1 replica, 100 requests)

| Configuration | Avg Latency | Overhead vs Baseline | Overhead % |
|--------------|-------------|---------------------|------------|
| Model only (baseline) | 239 ms | — | — |
| Model + SHAP | 776 ms | +537 ms | +225% |
| Model + LIME | 3339 ms | +3100 ms | +1296% |
| Model + SHAP + LIME | 3655 ms | +3416 ms | +1429% |

### Scaling Impact on SHAP Throughput

| Replicas | Avg Latency (100 req) | Throughput |
|----------|----------------------|------------|
| 1 pod | 776 ms | 6.4 req/s |
| 2 pods | 525 ms | 9.4 req/s |
| 3 pods | 421 ms | 11.7 req/s |

Kubernetes scaling improved SHAP throughput by **83%** (1→3 replicas).

### Generated Graphs

All graphs are saved to `results/graphs/`:
- `latency_vs_load.png` — load vs latency per endpoint
- `throughput_vs_load.png` — load vs throughput
- `latency_comparison.png` — baseline vs SHAP vs LIME at 100 requests
- `overhead_vs_load.png` — XAI overhead % vs load

---

## Model Performance

| Metric | Value |
|--------|-------|
| Dataset | Breast Cancer Wisconsin (569 samples, 30 features) |
| Algorithm | RandomForestClassifier (100 trees) |
| Accuracy | 95.61% |
| Precision | 95.61% |
| Recall | 95.61% |
| F1 Score | 95.60% |

---

## Key Findings

1. LIME introduces ~14x more overhead than baseline inference
2. SHAP introduces ~2x overhead but provides exact feature attributions
3. Kubernetes horizontal scaling reduces SHAP latency significantly
4. Zero failures across all load test scenarios
5. The XAI overhead is the dominant cost — not the model inference

---

## Load Testing

```cmd
# Test against running service
python tests/load_test.py --host http://localhost:8000 --loads 10 50 100 500

# Test specific endpoints only
python tests/load_test.py --host http://localhost:8000 --endpoints baseline shap
```
