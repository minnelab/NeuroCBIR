#!/bin/bash
set -e

# This script is a wrapper for running the NeuroCBIR pipeline.
# It calls the appropriate run script based on the user's choice.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 [docker|snakemake] [OPTIONS...]"
    echo "  docker    : Runs the pipeline using Docker Compose."
    echo "  snakemake : Runs the pipeline using Snakemake and Singularity."
    exit 1
fi

RUN_TYPE=$1
shift # Consume the first argument, pass the rest

case "$RUN_TYPE" in
    docker)
        echo "--- 🚀 Running NeuroCBIR with Docker ---"
        bash deploy/scripts/run_docker.sh "$@"
        ;;
    snakemake)
        echo "--- 🐍 Running NeuroCBIR with Snakemake ---"
        bash deploy/scripts/run_snakemake.sh "$@"
        ;;
    *)
        echo "Error: Invalid run type '$RUN_TYPE'."
        echo "Usage: $0 [docker|snakemake] [OPTIONS...]"
        exit 1
        ;;
esac