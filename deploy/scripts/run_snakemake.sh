#!/bin/bash
set -e  # Exit on any error
REPO_DIR=$(pwd) # Repository root directory

# Check if inside NeuroCBIR repo
if [ ! -f "$REPO_DIR/run_neurocbir.sh" ]; then
    echo "Error: Please run this script from the NeuroCBIR repository root directory."
    exit 1
fi  

# === DEFAULTS & CONFIGURATION ===
SNAKEMAKE_DIR="deploy/snakemake"
SNAKEMAKE_CONFIG_DEFAULT="$SNAKEMAKE_DIR/config.yaml"
CORES_DEFAULT=4
FS_LICENSE_PATH_DEFAULT="/usr/local/freesurfer/.license"

# Use defaults
SNAKEMAKE_CONFIG="$SNAKEMAKE_CONFIG_DEFAULT"
CORES="$CORES_DEFAULT"
FS_LICENSE_PATH="$FS_LICENSE_PATH_DEFAULT"

# Array to hold Snakemake config overrides
SNAKEMAKE_CONFIG_OVERRIDES=()

# === PARSE ARGUMENTS ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) SNAKEMAKE_CONFIG="$2"; shift ;;
        --cores) CORES="$2"; shift ;;
        --fs-license)
            FS_LICENSE_PATH="$2"
            SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "fs_license=$2")
            shift
            ;;
        # Specific overrides for snakemake config.yaml
        --outdir) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "outdir=$2"); shift ;;
        --guid) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "guid=$2"); shift ;;
        --raw_mri_path) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "raw_mri_path=$2"); shift ;;
        --user_config) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "user_config=$2"); shift ;;
        --internal_config) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "internal_config=$2"); shift ;;
        --region) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "region=$2"); shift ;;
        --top_k) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "top_k=$2"); shift ;;
        --device) SNAKEMAKE_CONFIG_OVERRIDES+=("--config" "device=$2"); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# === INTERNAL CONFIGURATION ===
VENV_DIR="$SNAKEMAKE_DIR/venv"

# Activate the Python virtual environment
source $VENV_DIR/bin/activate

# Color codes for better output
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

START_TIME=$(date +%s)

if [ ! -d "$SNAKEMAKE_DIR/singularity" ]; then
    echo -e "${RED}Error: Singularity images directory not found at '$SNAKEMAKE_DIR/singularity'${RESET}"
    echo -e "${CYAN}Please build the images first by running: './setup_snakemake.sh'${RESET}"
    exit 1
fi

echo -e "${CYAN}🚀 Starting Snakemake pipeline via Singularity...${RESET}"

# Show the configuration in the config file
echo -e "${CYAN}Using Snakemake configuration from '$SNAKEMAKE_CONFIG':${RESET}"
cat "$SNAKEMAKE_CONFIG"
echo "" # Blank line for readability        

# Execute Snakemake using the profile
# Snakemake will handle container execution for each rule automatically
snakemake \
    --directory "$SNAKEMAKE_DIR" \
    --snakefile "$SNAKEMAKE_DIR/Snakefile" \
    --configfile "$SNAKEMAKE_CONFIG" \
    --cores "$CORES" \
    "${SNAKEMAKE_CONFIG_OVERRIDES[@]}" \
    --use-singularity \
    --singularity-prefix "singularity" \
    --singularity-args "--home $REPO_DIR \
                        --bind $FS_LICENSE_PATH:/usr/local/freesurfer/.license"

# End timer
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "${CYAN}✅ Snakemake pipeline completed successfully!${RESET}"
echo -e "${CYAN}Total time: $TOTAL_TIME seconds ($(date -ud "@$(printf "%.0f" $TOTAL_TIME)" +'%Hh:%Mm:%Ss'))${RESET}"