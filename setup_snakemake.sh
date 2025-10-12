#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Setup script for NeuroCBIR pipeline
# -----------------------------

# Configuration
SETUP_DIR="deploy/snakemake"

# Python virtual environment
VENV_DIR="$SETUP_DIR/venv"
SNK_VERSION="9.12.0"

# Docker configuration (for building Singularity images)
DOCKERFILE_PATH="deploy/docker/Dockerfile"
DOCKER_IMAGE_NAME="neurocbir"
DOCKER_IMAGE_TAG="latest"

# Singularity configuration
SINGULARITY_DIR="$SETUP_DIR/singularity"
NEUROCBIR_DOCKERFILE="deploy/docker/Dockerfile"
FREESURFER_DOCKER_IMAGE="freesurfer/freesurfer:7.4.1"
ANTS_DOCKER_IMAGE="antsx/ants:latest"

# SIF file paths
NEUROCBIR_SIF="$SINGULARITY_DIR/neurocbir.sif"
FREESURFER_SIF="$SINGULARITY_DIR/freesurfer.sif"
ANTS_SIF="$SINGULARITY_DIR/ants.sif"

# -----------------------------
# 1. Check Python installation
# -----------------------------
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python >=3.10"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.10"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Python version must be >= $REQUIRED_VERSION. Detected: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version OK ($PYTHON_VERSION)"

# -----------------------------
# 2. Create Python virtual environment
# -----------------------------
mkdir -p "$SETUP_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Upgrade pip
echo "Upgrading pip in virtual environment..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

# -----------------------------
# 3. Install Snakemake
# -----------------------------
echo "Installing Snakemake $SNK_VERSION..."
"$VENV_DIR/bin/pip" install "snakemake==$SNK_VERSION"

# -----------------------------
# 4. Check Apptainer/Singularity
# -----------------------------
if ! command -v apptainer &> /dev/null && ! command -v singularity &> /dev/null; then
    echo "Apptainer/Singularity not found. Please install it."
    exit 1
fi
APPTAINER_CMD=$(command -v apptainer || command -v singularity)
echo "✅ Found container runtime: $APPTAINER_CMD"

# -----------------------------
# 5. Build Singularity images
# -----------------------------
mkdir -p "$SINGULARITY_DIR"

echo "Building temporary NeuroCBIR Docker image..."
docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" -f "${DOCKERFILE_PATH}" .

echo "Building NeuroCBIR Singularity image from local Docker image..."
$APPTAINER_CMD build --fakeroot "$NEUROCBIR_SIF" "docker-daemon://${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"


echo "Building Freesurfer Singularity image from Docker Hub..."
# Using the official FreeSurfer image
$APPTAINER_CMD build --fakeroot "$FREESURFER_SIF" "docker://$FREESURFER_DOCKER_IMAGE"

echo "Building ANTs Singularity image from Docker Hub..."
# Using a common ANTs image; adjust if you use a different one
$APPTAINER_CMD build --fakeroot "$ANTS_SIF" "docker://$ANTS_DOCKER_IMAGE"

# -----------------------------
# 6. Final message
# -----------------------------
echo "✅ Setup complete!"
echo "Virtual environment: $VENV_DIR (activate with 'source $VENV_DIR/bin/activate')"
echo "Singularity images stored in $SINGULARITY_DIR"
echo "Run pipeline using: source $VENV_DIR/bin/activate && ./run_snakemake.sh"
