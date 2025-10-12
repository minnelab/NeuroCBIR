#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Setup script for NeuroCBIR pipeline
# -----------------------------

# Configuration
SETUP_DIR="setup"
SINGULARITY_DIR="$SETUP_DIR/singularity_images"
NEUROCBIR_DOCKER_IMAGE="neurocbir:latest"
NEUROCBIR_SIF="$SINGULARITY_DIR/neurocbir.sif"

# Other container builds (optional)
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
VENV_DIR="$SETUP_DIR/snakemake_venv"
mkdir -p "$SETUP_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# -----------------------------
# 3. Install Snakemake
# -----------------------------
SNK_VERSION="9.12.0"
echo "Installing Snakemake $SNK_VERSION..."
pip install "snakemake==$SNK_VERSION"

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

echo "Building NeuroCBIR Singularity image..."
$APPTAINER_CMD build --fakeroot "$NEUROCBIR_SIF" "deploy/singularity/Singularity.def"

# Optional: build Freesurfer and ANTs containers
# Adjust paths or Docker images as needed
if [[ -f "deploy/singularity/freesurfer.def" ]]; then
    echo "Building Freesurfer Singularity image..."
    $APPTAINER_CMD build --fakeroot "$FREESURFER_SIF" "deploy/singularity/freesurfer.def"
fi

if [[ -f "deploy/singularity/ants.def" ]]; then
    echo "Building ANTs Singularity image..."
    $APPTAINER_CMD build --fakeroot "$ANTS_SIF" "deploy/singularity/ants.def"
fi

# -----------------------------
# 6. Final message
# -----------------------------
echo "✅ Setup complete!"
echo "Virtual environment: $VENV_DIR (activate with 'source $VENV_DIR/bin/activate')"
echo "Singularity images stored in $SINGULARITY_DIR"
echo "Run pipeline using: source $VENV_DIR/bin/activate && snakemake --use-singularity --profile profile"
