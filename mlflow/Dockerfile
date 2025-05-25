FROM python:3.12-slim

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    boto3 \
    psycopg2-binary

# Create directories for artifacts and runs
RUN mkdir -p /mlruns /mlartifacts

# Set proper permissions
RUN chmod 755 /mlruns /mlartifacts

# Expose MLflow port
EXPOSE 5000

# Set environment variables
ENV MLFLOW_BACKEND_STORE_URI=file:///mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=file:///mlartifacts

# Add health check endpoint
RUN echo '#!/bin/bash\necho "MLflow Health OK"' > /usr/local/bin/health && chmod +x /usr/local/bin/health

# Start MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "file:///mlruns", \
     "--default-artifact-root", "file:///mlartifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--serve-artifacts"]