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

echo "Building NeuroCBIR Docker image (if needed)..."
docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" -f "${DOCKERFILE_PATH}" .

# --- Helper function to check and build SIFs ---
build_sif_if_needed() {
    local sif_path="$1"
    local docker_uri="$2"
    local image_name="$3"
    local id_file="${sif_path}.id"
    local current_id=""

    # Get the unique ID/Digest for the Docker image
    if [[ "$docker_uri" == docker-daemon://* ]]; then
        # For local images, use the Image ID
        current_id=$(docker inspect --format='{{.Id}}' "$image_name")
    else
        # For remote images, pull and use the RepoDigest for immutability
        docker pull "$image_name" > /dev/null
        current_id=$(docker inspect --format='{{index .RepoDigests 0}}' "$image_name")
    fi

    # Check if a rebuild is needed
    if [ -f "$sif_path" ] && [ -f "$id_file" ] && [ "$(cat "$id_file")" == "$current_id" ]; then
        echo "✅ Singularity image for '$image_name' is up to date. Skipping build."
        return
    fi

    echo "🚀 Building Singularity image for '$image_name'..."
    if $APPTAINER_CMD build --fakeroot "$sif_path" "$docker_uri"; then
        # Store the ID on successful build
        echo "$current_id" > "$id_file"
        echo "   -> Successfully built '$sif_path'"
    else
        echo "   -> Failed to build '$sif_path'. Please check the error messages."
        # Clean up failed build artifacts if they exist
        rm -f "$sif_path" "$id_file"
        exit 1
    fi
}


# --- Build NeuroCBIR SIF ---
build_sif_if_needed "$NEUROCBIR_SIF" \
    "docker-daemon://${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" \
    "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"

# --- Build Freesurfer SIF ---
build_sif_if_needed "$FREESURFER_SIF" \
    "docker://${FREESURFER_DOCKER_IMAGE}" \
    "$FREESURFER_DOCKER_IMAGE"

# --- Build ANTs SIF ---
build_sif_if_needed "$ANTS_SIF" "docker://${ANTS_DOCKER_IMAGE}" "$ANTS_DOCKER_IMAGE"

# -----------------------------
# 6. Final message
# -----------------------------
echo "✅ Setup complete!"
echo "Virtual environment: $VENV_DIR (activate with 'source $VENV_DIR/bin/activate')"
echo "Singularity images stored in $SINGULARITY_DIR"
echo "Run pipeline using: source $VENV_DIR/bin/activate && ./run_snakemake.sh"
