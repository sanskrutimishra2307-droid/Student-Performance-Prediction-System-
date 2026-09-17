# -----------------------------------------------------------------------------
# Dockerfile for Student Performance Prediction & Learning Intelligence System
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered standard out/err
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system runtime dependencies (libgomp for XGBoost/OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, static web files, and data
COPY . .

# Expose default HTTP port
EXPOSE 8000

# Health check to ensure API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch FastAPI application using Uvicorn
CMD exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}
