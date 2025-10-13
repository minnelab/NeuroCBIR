#!/bin/bash
set -e

# Color codes for better output
CYAN=$'\033[36m'
YELLOW=$'\033[33m'
RESET=$'\033[0m'

# Source the banner function
source "$(dirname "$0")/deploy/scripts/banner.sh"

if [ "$#" -lt 1 ]; then
    print_banner
    # Usage message
    echo "Usage: $0 [docker|snakemake] [OPTIONS...]"
    echo "  docker    : Runs the pipeline using Docker Compose."
    echo "  snakemake : Runs the pipeline using Snakemake and Singularity."
    echo -e "${YELLOW}For options specific to a run type, use the --help flag after the command:${RESET}"
    echo "  $0 docker --help"
    echo "  $0 snakemake --help"
    exit 1
fi

RUN_TYPE=$1
shift # Consume the first argument, pass the rest

case "$RUN_TYPE" in
    docker)
        echo -e "--- ${CYAN}🚀 Running NeuroCBIR with Docker${RESET} ---"
        bash deploy/scripts/run_docker.sh "$@"
        ;;
    snakemake)
        echo -e "--- ${CYAN}🐍 Running NeuroCBIR with Snakemake${RESET} ---"
        bash deploy/scripts/run_snakemake.sh "$@"
        ;;
    -h|--help)
        print_banner
        echo "Usage: $0 [docker|snakemake] [OPTIONS...]"
        echo ""
        echo "Top-level wrapper script to run the NeuroCBIR pipeline."
        echo ""
        echo "Commands:"
        echo "  docker      Runs the pipeline using Docker and Docker Compose."
        echo "  snakemake   Runs the pipeline using Snakemake and Singularity."
        echo ""
        echo -e "${YELLOW}For options specific to a run type, use the --help flag after the command:${RESET}"
        echo "  $0 docker --help"
        echo "  $0 snakemake --help"
        exit 0
        ;;
    *)
        echo "Error: Invalid run type '$RUN_TYPE'."
        echo "Usage: $0 [docker|snakemake] [OPTIONS...]"
        exit 1
        ;;
esac