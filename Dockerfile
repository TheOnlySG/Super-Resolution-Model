# Multi-stage Dockerfile for Satellite Imagery Super-Resolution Platform

# --- Stage 1: Build virtual environment ---
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build-essential and GDAL dependencies for package compilation if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# --- Stage 2: Final Runtime ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install runtime GDAL libraries and curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for container security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/temp_outputs && \
    chown -R appuser:appuser /app

# Copy application source code and models
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser frontend/ /app/frontend/
COPY --chown=appuser:appuser model/ /app/model/

# Set environment variables
ENV MODEL_PATH=/app/model/RCAN_v2.pth \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

STOPSIGNAL SIGTERM
USER appuser
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
