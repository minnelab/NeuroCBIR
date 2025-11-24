# ------------------------------------------------------------------------------
# NeuroCBIR Dockerfile
# CPU-only version (compatible with deepmi/fastsurfer:cpu-v2.4.2 if needed)
# ------------------------------------------------------------------------------

# === Base image ===
# You can switch to a PyTorch image if you want preinstalled torch
FROM python:3.13-slim

LABEL maintainer="Félix Nieto-del-Amor <fenda@kth.se>"
LABEL description="NeuroCBIR: A Public Content-Based Image Retrieval System for Whole-Brain and Region-Specific MRI Across Multiple Clinical Cohorts"

# === Environment setup ===
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app \
    PYTHONPATH=/app

WORKDIR $APP_HOME

# ------------------------------------------------------------------------------
# 1. Install system dependencies
# ------------------------------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 2. Install Python dependencies
# ------------------------------------------------------------------------------

# Copy only the requirements first (for Docker caching)
COPY deploy/docker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# 3. Copy the application code
# ------------------------------------------------------------------------------

COPY deploy/neurocbir /app/neurocbir
COPY deploy/configs /app/configs
COPY deploy/data /app/data
COPY deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

# ------------------------------------------------------------------------------
# 4. Final cleanup and environment metadata
# ------------------------------------------------------------------------------

ENV PATH="$APP_HOME:$PATH"
WORKDIR "/app"

# ------------------------------------------------------------------------------
# 5. Entrypoint
# ------------------------------------------------------------------------------

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# CMD ["--help"]

