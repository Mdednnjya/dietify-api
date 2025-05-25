FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including networking tools for debugging
RUN apt-get update && apt-get install -y \
    gcc \
    libgit2-dev \
    libffi-dev \
    libssl-dev \
    build-essential \
    curl \
    iputils-ping \
    netcat-traditional \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories with proper permissions
RUN mkdir -p models output data/raw data/interim mlruns mlartifacts && \
    chmod -R 755 models output data mlruns mlartifacts

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Enhanced health check with retry logic
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application with proper error handling
CMD ["sh", "-c", "python -c 'import sys; print(f\"Python path: {sys.path}\")' && uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info --timeout-keep-alive 300"]