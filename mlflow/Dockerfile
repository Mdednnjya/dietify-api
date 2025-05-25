FROM python:3.11-slim

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    boto3 \
    psycopg2-binary \
    click

# Create directories for artifacts and runs
RUN mkdir -p /mlruns /mlartifacts

# Set proper permissions
RUN chmod -R 755 /mlruns /mlartifacts

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:5000/health || exit 1' > /health_check.sh && \
    chmod +x /health_check.sh

# Expose MLflow port
EXPOSE 5000

# Set environment variables
ENV MLFLOW_BACKEND_STORE_URI=file:///mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///mlartifacts
ENV PYTHONPATH=/mlflow

# Add simple health endpoint
RUN echo 'from flask import Flask; app = Flask(__name__); @app.route("/health"); def health(): return "OK"' > /health.py

# Start MLflow server with proper error handling
CMD ["sh", "-c", "mlflow server --backend-store-uri file:///mlruns --default-artifact-root file:///mlartifacts --host 0.0.0.0 --port 5000 --serve-artifacts || (echo 'MLflow failed to start' && sleep 30)"]