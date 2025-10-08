#!/bin/bash
set -e  # Exit on error

CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

START_TIME=$(date +%s)

# === DEFAULTS ===
PREPROCESS=false
OUT_PATH=""
RAW_MRI_PATH=""
BRAIN_PATH=""
SEG_PATH=""
GUID=""
REGION="Left-Hippocampus"
SCOPE="region"
TOP_K=30

# === PARSE ARGUMENTS ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --preprocess) PREPROCESS=true ;;
        --out_path) OUT_PATH="$2"; shift ;;
        --raw_mri_path) RAW_MRI_PATH="$2"; shift ;;
        --brain_path) BRAIN_PATH="$2"; shift ;;
        --seg_path) SEG_PATH="$2"; shift ;;
        --guid) GUID="$2"; shift ;;
        --region) REGION="$2"; shift ;;
        --scope) SCOPE="$2"; shift ;;
        --top_k) TOP_K="$2"; shift ;;
        *) echo -e "${RED}Unknown parameter passed: $1${RESET}"; exit 1 ;;
    esac
    shift
done
export OUT_PATH="$OUT_PATH"

# === VALIDATION ===
if [[ -z "$OUT_PATH" || -z "$GUID" ]]; then
    echo -e "${RED}Error: --out_path and --guid are required.${RESET}"
    exit 1
fi

if [[ "$PREPROCESS" == true && -z "$RAW_MRI_PATH" ]]; then
    echo -e "${RED}Error: --raw_mri_path is required when using --preprocess.${RESET}"
    exit 1
fi

if [[ "$PREPROCESS" == false && ( -z "$BRAIN_PATH" || -z "$SEG_PATH" ) ]]; then
    echo -e "${RED}Error: --brain_path and --seg_path are required when not preprocessing.${RESET}"
    exit 1
fi

# === PREPROCESSING ===
if [[ "$PREPROCESS" == true ]]; then
    mkdir -p "${OUT_PATH}/${GUID}/mri/orig"
    cp $RAW_MRI_PATH "${OUT_PATH}/${GUID}/mri/orig/001.mgz"

    echo -e "${CYAN}Step 1 — preprocessing${RESET}"
    ./preprocess.sh "/data/${GUID}/mri/orig/001.mgz" "/data" "$GUID"

    docker compose run --rm \
        neurocbir \
        --img_path "/data/${GUID}/mri/brain_talairach.nii.gz" \
        --seg_path "/data/${GUID}/mri/aparc+aseg_talairach.nii.gz" \
        --scope "$SCOPE" \
        --region "$REGION" \
        --top_k "$TOP_K"

else
    mkdir -p "${OUT_PATH}/${GUID}/mri"
    cp $BRAIN_PATH "${OUT_PATH}/${GUID}/mri/brain_talairach.nii.gz"
    cp $SEG_PATH "${OUT_PATH}/${GUID}/mri/aparc+aseg_talairach.nii.gz"

    docker compose run --rm \
        neurocbir \
        --img_path "${OUT_PATH}/${GUID}/mri/brain_talairach.nii.gz" \
        --seg_path "${OUT_PATH}/${GUID}/mri/aparc+aseg_talairach.nii.gz" \
        --scope "$SCOPE" \
        --region "$REGION" \
        --top_k "$TOP_K"

fi

# === NEUROCBIR ===
echo -e "${CYAN}Step 2 — neurocbir${RESET}"

# Adjust mount point dynamically based on --out_path
# cd ./deploy/infra/


# === TIMER ===
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo -e "${CYAN}Pipeline completed for $GUID${RESET}"
echo -e "${CYAN}Total time: $TOTAL_TIME seconds ($(date -ud "@$(printf "%.0f" $TOTAL_TIME)" +'%Hh:%Mm:%Ss'))${RESET}"

