#!/bin/bash
set -e  # Exit on any error
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

# === INPUT PARAMETERS ===
INPUT_IMAGE=$1      # Path to input MRI
OUTPUT_DIR=$2       # Path to save outputs
GUID=$3             # FreeSurfer subject name

if [[ -z "$INPUT_IMAGE" || -z "$OUTPUT_DIR" || -z "$GUID" ]]; then
    echo -e "${RED}Usage: $0 <input_image> <subjects_dir> <guid>${RESET}"
    exit 1
fi

cd ./deploy/infra/

docker compose run --rm freesurfer \
    mkdir -p "${OUTPUT_DIR}"

docker compose run --rm freesurfer \
    mkdir -p "${OUTPUT_DIR}/${GUID}/mri/orig"

echo -e "${CYAN}Starting preprocessing pipeline${RESET}"
echo -e "${CYAN}Input image: $INPUT_IMAGE${RESET}"
echo -e "${CYAN}Output directory: $OUTPUT_DIR${RESET}"
echo -e "${CYAN}Subject ID: $GUID${RESET}"

# Start timer
START_TIME=$(date +%s)

# Step 0: copy original image
# docker compose run --rm freesurfer \
#     cp "$INPUT_IMAGE" "${OUTPUT_DIR}/${GUID}/mri/orig/001.mgz"

# Step 1: conform image
echo -e "${CYAN}Step 1 — mri_convert --conform${RESET}"

docker compose run --rm freesurfer \
    mri_convert --conform "${OUTPUT_DIR}/${GUID}/mri/orig/001.mgz" "${OUTPUT_DIR}/${GUID}/mri/orig_conform.mgz"

# Step 2: bias correction
echo -e "${CYAN}Step 2 — nu_correct${RESET}"

docker compose run --rm ants \
    N4BiasFieldCorrection \
    -i "${OUTPUT_DIR}/${GUID}/mri/orig_conform.mgz" \
    -o "${OUTPUT_DIR}/${GUID}/mri/orig_nu.mgz"

# Step 3: SynthSeg segmentation
echo -e "${CYAN}Step 3 — mri_synthseg${RESET}"

docker compose run --rm freesurfer \
    mri_synthseg \
    --i "${OUTPUT_DIR}/${GUID}/mri/orig_nu.mgz" \
    --o "${OUTPUT_DIR}/${GUID}/mri/aparc+aseg.mgz" \
    --robust \
    --cpu \
    --threads 8 \
    --parc

# Step 4: Talairach transformation
echo -e "${CYAN}Step 4 — talairach_avi${RESET}"

docker compose run --rm freesurfer \
    mkdir -p "${OUTPUT_DIR}/${GUID}/mri/transforms"

docker compose run --rm freesurfer \
    talairach_avi \
    --i "${OUTPUT_DIR}/${GUID}/mri/orig_nu.mgz" \
    --xfm "${OUTPUT_DIR}/${GUID}/mri/transforms/talairach.auto.xfm"

docker compose run --rm freesurfer \
    lta_convert \
    --src "${OUTPUT_DIR}/${GUID}/mri/orig_nu.mgz" \
    --trg "$FREESURFER_HOME/average/mni305.cor.mgz" \
    --inxfm "${OUTPUT_DIR}/${GUID}/mri/transforms/talairach.auto.xfm" \
    --outlta "${OUTPUT_DIR}/${GUID}/mri/transforms/talairach.xfm.lta"

# Step 5: Brain extraction (mri_synthstrip)
echo -e "${CYAN}Step 5 — mri_synthstrip${RESET}"

docker compose run --rm freesurfer \
    mri_synthstrip \
    --i "${OUTPUT_DIR}/${GUID}/mri/orig_nu.mgz" \
    --o "${OUTPUT_DIR}/${GUID}/mri/brain.mgz" \
    --mask "${OUTPUT_DIR}/${GUID}/mri/mask.mgz"

# Step 6: mri_vol2vol for brain
echo -e "${CYAN}Step 6 — mri_vol2vol (brain)${RESET}"

docker compose run --rm freesurfer \
    mri_vol2vol \
    --mov "${OUTPUT_DIR}/${GUID}/mri/brain.mgz" \
    --targ "$FREESURFER_HOME/average/mni305.cor.mgz" \
    --reg "${OUTPUT_DIR}/${GUID}/mri/transforms/talairach.xfm.lta" \
    --o "${OUTPUT_DIR}/${GUID}/mri/brain_talairach.nii.gz"

# Step 7: mri_vol2vol for segmentation
echo -e "${CYAN}Step 7 — mri_vol2vol (segmentation)${RESET}"

docker compose run --rm freesurfer \
    mri_vol2vol \
    --mov "${OUTPUT_DIR}/${GUID}/mri/aparc+aseg.mgz" \
    --targ "$FREESURFER_HOME/average/mni305.cor.mgz" \
    --reg "${OUTPUT_DIR}/${GUID}/mri/transforms/talairach.xfm.lta" \
    --o "${OUTPUT_DIR}/${GUID}/mri/aparc+aseg_talairach.nii.gz"

# End timer
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo -e "${CYAN}Pipeline completed for $GUID${RESET}"
echo -e "${CYAN}Total time: $TOTAL_TIME seconds ($(date -ud "@$(printf "%.0f" $TOTAL_TIME)" +'%Hh:%Mm:%Ss'))${RESET}"
