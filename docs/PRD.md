# Product Requirements Document (PRD)

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

## 1. Product Overview

This project builds a cloud-deployed Explainable AI service that accepts clinical input data, runs it through a trained black-box machine learning model, and returns both a prediction and a human-readable explanation of why that prediction was made. The service is containerized using Docker and deployed on Kubernetes, enabling scalability and load testing.

The central research question is:

> **How much computational overhead does adding explainability (SHAP and LIME) introduce over a baseline model inference, and how does a Kubernetes-deployed XAI service behave under increasing workloads?**

---

## 2. Problem Statement

Machine learning models used in high-stakes domains like healthcare are often black boxes — they produce predictions without justification. This creates a trust and accountability gap. Clinicians, regulators, and patients cannot verify why a model made a specific decision.

Explainability methods like SHAP and LIME address this by attributing predictions to individual input features. However, these methods introduce computational cost. This project builds and evaluates a cloud service that quantifies that cost under real workloads.

---

## 3. Target Users

| User | Need |
|------|------|
| Medical professional | Understand why a model flagged a sample as malignant |
| ML engineer | Evaluate SHAP vs LIME cost trade-offs |
| Cloud architect | Understand how XAI services scale under load |
| Researcher / student | Study computational overhead of explainability |

---

## 4. Goals and Objectives

### Primary Goals
- Build a working REST API that serves Random Forest predictions with SHAP and LIME explanations
- Containerize the service using Docker
- Deploy it on Kubernetes (Minikube) and demonstrate scaling
- Experimentally measure and report the computational overhead of XAI

### Secondary Goals
- Provide a frontend dashboard for live demonstration
- Generate real performance graphs from actual experiments
- Document the system clearly enough for a viva presentation

### Non-Goals
- This system does not aim to improve model accuracy
- SHAP and LIME are used for interpretability, not performance optimization
- This is not a production medical diagnostic system

---

## 5. Core Features

### F1 — Model Prediction (Baseline)
The service accepts 30 clinical features from the Breast Cancer Wisconsin dataset and returns a classification (malignant or benign) with confidence score and inference latency.

### F2 — SHAP Explanation
The service returns a SHAP-based feature attribution for each prediction, showing which features pushed the model toward or away from the predicted class.

### F3 — LIME Explanation
The service returns a LIME-based local explanation with human-readable feature rules and their contribution weights.

### F4 — Combined Explanation
A single endpoint returns both SHAP and LIME explanations together with full latency breakdown and computed XAI overhead percentage.

### F5 — Frontend Dashboard
A web UI allowing users to input a patient sample, request predictions and explanations, and view results visually with charts and a performance panel.

### F6 — Containerization
The entire application runs inside a Docker container. The container includes the trained model, all dependencies, and the API. No retraining occurs at runtime.

### F7 — Kubernetes Deployment
The Docker image is deployed on Minikube with configurable replicas. The deployment includes resource limits, health probes, and a NodePort service.

### F8 — Performance Experiment
Load tests are run against the deployed service at increasing request volumes. Results are recorded and graphed, covering latency, throughput, CPU, and memory.

---

## 6. Performance Requirements

| Metric | Requirement |
|--------|-------------|
| Baseline inference latency | < 10 ms per request |
| SHAP explanation latency | Measured and reported (no fixed limit) |
| LIME explanation latency | Measured and reported (no fixed limit) |
| Overhead calculation | (XAI time − baseline) / baseline × 100 |
| Load test volumes | 10, 50, 100, 500, 1000 requests |
| Results | Must come from actual experiments, not estimated |

---

## 7. Constraints

- All results must be from real experiments — no fabricated data
- The model must not be retrained per request
- SHAP and LIME explainer objects must be reused across requests
- Deployment must work on Minikube (local Kubernetes), not real cloud
- The project must be explainable in a viva by a student

---

## 8. Success Criteria

- All 5 API endpoints respond correctly
- Docker image builds and runs successfully
- Kubernetes deployment shows pods running with correct replicas
- Dashboard loads and all 4 buttons produce correct results
- Performance graphs are generated from real data
- XAI overhead is quantified with actual numbers
