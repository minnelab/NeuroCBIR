#!/bin/bash
set -e  # Exit on any error
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

START_TIME=$(date +%s)

# === INPUT PARAMETERS ===
INPUT_IMAGE=$1      # Path to input MRI
OUTPUT_DIR=$2       # Path to save outputs
GUID=$3             # FreeSurfer subject name



if [[ -z "$INPUT_IMAGE" || -z "$OUTPUT_DIR" || -z "$GUID" ]]; then
    echo -e "${RED}Usage: $0 <input_image> <subjects_dir> <guid>${RESET}"
    exit 1
fi

# Step 1: preprocess
./preprocess.sh $INPUT_IMAGE $OUTPUT_DIR $GUID

cd ./deploy/infra/

# Step 2: neurocbir
echo -e "${CYAN}Step 2 — neurocbir${RESET}"

docker compose run --rm neurocbir \
    --img_path "${OUTPUT_DIR}/${GUID}/mri/brain_talairach.nii.gz" \
    --seg_path "${OUTPUT_DIR}/${GUID}/mri/aparc+aseg_talairach.nii.gz" \
    --scope "region" \
    --region "Left-Hippocampus" \
    --top_k 30


# End timer
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "${CYAN}Pipeline completed for $GUID${RESET}"
echo -e "${CYAN}Total time: $TOTAL_TIME seconds ($(date -ud "@$(printf "%.0f" $TOTAL_TIME)" +'%Hh:%Mm:%Ss'))${RESET}"
