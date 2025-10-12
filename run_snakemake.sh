#!/bin/bash
set -e  # Exit on any error

# Color codes for better output
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

START_TIME=$(date +%s)

# === CONFIGURATION ===
# Path to the Singularity image file.
# This is now managed by the Snakemake profile
SNAKEMAKE_DIR="deploy/snakemake"

# Path to the Snakemake configuration file
SNAKEMAKE_CONFIG="$SNAKEMAKE_DIR/config.yaml"

# Number of cores to use for Snakemake
CORES=4

if [ ! -d "$SNAKEMAKE_DIR/singularity" ]; then
    echo -e "${RED}Error: Singularity images directory not found at '$SNAKEMAKE_DIR/singularity'${RESET}"
    echo -e "${CYAN}Please build the images first by running: './setup_snakemake.sh'${RESET}"
    exit 1
fi

echo -e "${CYAN}🚀 Starting Snakemake pipeline via Singularity...${RESET}"

# Execute Snakemake using the profile
# Snakemake will handle container execution for each rule automatically
snakemake \
    --snakefile "$SNAKEMAKE_DIR/Snakefile" \
    --configfile "$SNAKEMAKE_CONFIG" \
    --cores "$CORES" \
    --profile "$SNAKEMAKE_DIR/profile" \
    --use-singularity

# End timer
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "${CYAN}✅ Snakemake pipeline completed successfully!${RESET}"
echo -e "${CYAN}Total time: $TOTAL_TIME seconds ($(date -ud "@$(printf "%.0f" $TOTAL_TIME)" +'%Hh:%Mm:%Ss'))${RESET}"