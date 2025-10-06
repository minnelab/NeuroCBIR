#!/usr/bin/env bash
set -e  # Exit immediately if a command exits with a non-zero status

# ------------------------------------------------------------------------------
# NeuroCBIR Docker Entrypoint
# ------------------------------------------------------------------------------
# This script serves as the main entrypoint for running the NeuroCBIR application
# inside a Docker container.
#
# It assumes:
# - The Python environment is already installed and activated.
# - The working directory is set to /app (or wherever neurocbir/ lives).
# - Configuration files are in deploy/configs/.
# ------------------------------------------------------------------------------

echo "🚀 Starting NeuroCBIR container..."
echo "📁 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Installed packages snapshot:"
pip list | grep neurocbir || echo "(NeuroCBIR package not found — continuing anyway)"

# Allow overriding the command (e.g., to open a shell)
if [ "$1" = "bash" ] || [ "$1" = "sh" ]; then
    echo "🔧 Opening shell..."
    exec "$@"
else
    exec python -m neurocbir "$@"
fi
