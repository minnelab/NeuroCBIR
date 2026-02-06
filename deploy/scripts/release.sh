#!/bin/bash
set -e

# Check if a python environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  WARNING: No active Python environment detected."
    echo "It's recommended to activate your virtual environment before running this script."
    read -p "Do you want to continue without an active Python environment? (yes/no) " CONFIRM_ENV
    if [ "$CONFIRM_ENV" != "yes" ]; then
        echo "Aborting release."
        exit 1
    fi
fi

# ============================
# WARNING / CONFIRMATION
# ============================
echo "⚠️  WARNING: This script will:"
echo "    - Push the current branch and tag to GitHub"
echo "    - Create a new release"
echo "    - Optionally build and push a Docker image"
echo ""
read -p "Are you sure you want to continue? (yes/no) " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborting release."
    exit 1
fi

# ============================
# CONFIGURATION
# ============================

# Paths
PACKAGE_DIR="deploy/neurocbir"
DIST_DIR="$PACKAGE_DIR/dist"
DOCKER_DIR="deploy/docker"
ROOT_DATA_DIR="data_private"             # top-level folder



# Docker Hub (change to your username)
# DOCKER_REPO="your_dockerhub_username/neurocbir"

# Optional: push Docker image? yes/no
# PUSH_DOCKER="yes"

# ============================
# STEP 0: Read version
# ============================
echo "Reading version from deploy/neurocbir/version.py..."

PACKAGE_VERSION=$(python3 - <<EOF
import os

version_file = "deploy/neurocbir/version.py"
ns = {}
with open(version_file, "r") as f:
    exec(f.read(), ns)

print(ns["__version__"])
EOF
)

if [ -z "$PACKAGE_VERSION" ]; then
    echo "❌ Error: Could not read NeuroCBIR version."
    exit 1
fi

echo "📦 Releasing NeuroCBIR version $PACKAGE_VERSION"


# ============================
# WARNING / CONFIRMATION
# ============================
read -p "Is the version $PACKAGE_VERSION correct? (yes/no) " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborting release."
    exit 1
fi

# ============================
# STEP 1: Create Git tag
# ============================
git tag -a "v$PACKAGE_VERSION" -m "Release NeuroCBIR v$PACKAGE_VERSION"
git push origin "v$PACKAGE_VERSION"

# ============================
# STEP 2: Build Python package
# ============================
cd "$PACKAGE_DIR"
rm -rf dist
python3 -m build
cd -

# ============================================
# STEP 3 - ZIP data_private
# ============================================
SRC_DATA_DIR="$ROOT_DATA_DIR/data_private"   # actual data to compress
DATA_RELEASE_DIR="$ROOT_DATA_DIR/releases"

if [ ! -d "$SRC_DATA_DIR" ]; then
    echo "❌ ERROR: Expected directory '$SRC_DATA_DIR' does not exist."
    exit 1
fi

mkdir -p "$DATA_RELEASE_DIR"

ARCHIVE_NAME="data_private_v${PACKAGE_VERSION}.zip"
ARCHIVE_PATH="${DATA_RELEASE_DIR}/${ARCHIVE_NAME}"

echo "📦 Creating archive: $ARCHIVE_PATH"

cd "$SRC_DATA_DIR"
# zip everything inside the inner data_private/
zip -r "../../$ARCHIVE_PATH" . >/dev/null
cd - >/dev/null

echo "📦 Data archive created at: $ARCHIVE_PATH"


# ============================
# STEP 4: Create GitHub release
# ============================
# Requires GitHub CLI installed and authenticated (gh auth login)
gh release create "v$PACKAGE_VERSION" \
    "$DIST_DIR/neurocbir-$PACKAGE_VERSION.tar.gz" \
    "$DIST_DIR/neurocbir-$PACKAGE_VERSION-py3-none-any.whl" \
    "$ARCHIVE_PATH" \
    --title "NeuroCBIR v$PACKAGE_VERSION" \
    --notes "Automated release for version $PACKAGE_VERSION"

# ============================
# STEP 4: Build Docker image
# ============================
# if [ "$PUSH_DOCKER" = "yes" ]; then
#     docker build -t neurocbir:$PACKAGE_VERSION --build-arg NEUROCBIR_VERSION=$PACKAGE_VERSION "$DOCKER_DIR"
#     docker tag neurocbir:$PACKAGE_VERSION $DOCKER_REPO:$PACKAGE_VERSION
#     docker push $DOCKER_REPO:$PACKAGE_VERSION
# fi

# echo "Release v$PACKAGE_VERSION complete!"
