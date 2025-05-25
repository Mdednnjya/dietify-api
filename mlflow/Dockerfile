FROM python:3.11-slim

WORKDIR /mlflow

# Install system dependencies including networking tools
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    iputils-ping \
    netcat-traditional \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install MLflow and dependencies with specific versions
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    boto3 \
    psycopg2-binary \
    click \
    gunicorn==21.2.0

# Create directories for artifacts and runs
RUN mkdir -p /mlruns /mlartifacts

# Set proper permissions
RUN chmod -R 755 /mlruns /mlartifacts

# Expose MLflow port
EXPOSE 5000

# Set environment variables for MLflow API
ENV MLFLOW_BACKEND_STORE_URI=file:///mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///mlartifacts
ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000
ENV PYTHONPATH=/mlflow

# Configure Gunicorn for longer timeouts
ENV GUNICORN_TIMEOUT=300
ENV GUNICORN_WORKERS=2
ENV GUNICORN_WORKER_CLASS=sync
ENV GUNICORN_MAX_REQUESTS=1000
ENV GUNICORN_MAX_REQUESTS_JITTER=50

# Start MLflow server with proper API endpoints
CMD ["sh", "-c", "mlflow server \
    --backend-store-uri file:///mlruns \
    --default-artifact-root file:///mlartifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --serve-artifacts \
    --gunicorn-opts '--timeout=300 --workers=2 --worker-class=sync --max-requests=1000 --max-requests-jitter=50'"]