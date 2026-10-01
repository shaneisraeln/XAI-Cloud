# ── Dockerfile ────────────────────────────────────────────────────────────────
# XAI Cloud Service — production-style image
#
# Build:  docker build -t xai-cloud-service .
# Run:    docker run -p 8000:8000 xai-cloud-service
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies for C extensions (SHAP, scipy) ───────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Upgrade pip first, then install dependencies ──────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY app/        ./app/
COPY frontend/   ./frontend/
COPY model/      ./model/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
