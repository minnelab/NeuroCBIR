#!/bin/bash
set -e

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

# Docker Hub (change to your username)
# DOCKER_REPO="your_dockerhub_username/neurocbir"

# Optional: push Docker image? yes/no
# PUSH_DOCKER="yes"

# ============================
# STEP 0: Read version
# ============================
PACKAGE_VERSION=$(python3 -c "from $PACKAGE_DIR.version import __version__; print(__version__)")
if [ -z "$PACKAGE_VERSION" ]; then
    echo "Error: Could not read NeuroCBIR version."
    exit 1
fi
echo "Releasing NeuroCBIR version $PACKAGE_VERSION"

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

# ============================
# STEP 3: Create GitHub release
# ============================
# Requires GitHub CLI installed and authenticated (gh auth login)
gh release create "v$PACKAGE_VERSION" \
    "$DIST_DIR/neurocbir-$PACKAGE_VERSION.tar.gz" \
    "$DIST_DIR/neurocbir-$PACKAGE_VERSION-py3-none-any.whl" \
    --title "NeuroCBIR v$PACKAGE_VERSION" \
    --notes "First full release of NeuroCBIR. Includes Python package and optional heavy dependencies."

# ============================
# STEP 4: Build Docker image
# ============================
# if [ "$PUSH_DOCKER" = "yes" ]; then
#     docker build -t neurocbir:$PACKAGE_VERSION --build-arg NEUROCBIR_VERSION=$PACKAGE_VERSION "$DOCKER_DIR"
#     docker tag neurocbir:$PACKAGE_VERSION $DOCKER_REPO:$PACKAGE_VERSION
#     docker push $DOCKER_REPO:$PACKAGE_VERSION
# fi

# echo "Release v$PACKAGE_VERSION complete!"
