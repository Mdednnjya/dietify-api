FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow dengan versi yang stabil
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    mlflow==2.8.1 \
    gunicorn==21.2.0 \
    SQLAlchemy \
    alembic

# Create directories dengan proper permissions
RUN mkdir -p /app/mlruns /app/mlartifacts && \
    chmod -R 755 /app

# Create user untuk security
RUN useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow /app

# Create health check endpoint
RUN echo 'from flask import Flask; app = Flask(__name__); @app.route("/health"); def health(): return "OK"' > /app/health.py

# Switch to mlflow user
USER mlflow

# Environment variables untuk timeout dan performance
ENV MLFLOW_BACKEND_STORE_URI=sqlite:////app/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///app/mlartifacts
ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000

# Gunicorn timeout settings untuk long-running requests
ENV GUNICORN_TIMEOUT=300
ENV GUNICORN_WORKERS=2
ENV GUNICORN_WORKER_CLASS=sync
ENV GUNICORN_WORKER_CONNECTIONS=1000
ENV GUNICORN_MAX_REQUESTS=1000
ENV GUNICORN_MAX_REQUESTS_JITTER=100

# Expose port
EXPOSE 5000

# Health check dengan longer timeout
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command dengan optimized gunicorn settings
CMD ["python", "-m", "mlflow", "server", \
     "--backend-store-uri", "sqlite:////app/mlflow.db", \
     "--default-artifact-root", "file:///app/mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--serve-artifacts", \
     "--gunicorn-opts", "--timeout=300 --workers=2 --worker-class=sync --worker-connections=1000 --max-requests=1000 --max-requests-jitter=100 --preload"]