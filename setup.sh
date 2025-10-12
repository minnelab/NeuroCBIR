#!/bin/bash
set -e

# This script is a wrapper for setting up different NeuroCBIR environments.
# It calls the appropriate setup script based on the user's choice.

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [docker|snakemake]"
    echo "  docker    : Sets up the environment for the Docker-based workflow."
    echo "  snakemake : Sets up the environment for the Snakemake-based workflow."
    exit 1
fi

SETUP_TYPE=$1
shift # Consume the first argument

case "$SETUP_TYPE" in
    docker)
        echo "--- 🚀 Starting Docker Setup ---"
        bash deploy/scripts/setup_docker.sh "$@"
        ;;
    snakemake)
        echo "--- 🐍 Starting Snakemake Setup ---"
        bash deploy/scripts/setup_snakemake.sh "$@"
        ;;
    *)
        echo "Error: Invalid setup type '$SETUP_TYPE'."
        echo "Usage: $0 [docker|snakemake]"
        exit 1
        ;;
esac

echo "--- ✅ Setup for '$SETUP_TYPE' completed successfully! ---"