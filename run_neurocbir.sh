#!/bin/bash
set -e  # Exit on error

# === Colors ===
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

START_TIME=$(date +%s)

# === DEFAULTS ===
# For this wrapper
PREPROCESS=false
RAW_MRI_PATH="" # Need to define if --preprocess
OUT_PATH=""
GUID=""
# For neurocbir package
BRAIN_PATH="" # Do not define if --preprocess
SEG_PATH="" # Do not define if --preprocess
EMB_DATASET_PATH=""
REGION="" # Left-Hippocampus
SCOPE="" # whole_brain or region
TOP_K="" # int
DEVICE=""
INTERNAL_CONFIG=""
USER_CONFIG=""

# === PARSE ARGUMENTS ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --preprocess) PREPROCESS=true ;;
        --raw_mri_path) RAW_MRI_PATH="$2"; shift ;;
        --out_path) OUT_PATH="$2"; shift ;;
        --guid) GUID="$2"; shift ;;
        --brain_path) BRAIN_PATH="$2"; shift ;;
        --seg_path) SEG_PATH="$2"; shift ;;
        --emb_dataset_path) EMB_DATASET_PATH="$2"; shift ;;
        --region) REGION="$2"; shift ;;
        --scope) SCOPE="$2"; shift ;;
        --top_k) TOP_K="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --internal_config) INTERNAL_CONFIG="$2"; shift ;;
        --user_config) USER_CONFIG="$2"; shift ;;
        
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

if [[ "$PREPROCESS" == false && -z "$BRAIN_PATH" ]]; then
    echo -e "${RED}Error: --brain_path is required when not preprocessing.${RESET}"
    exit 1
fi

# === PREPROCESSING ===
if [[ "$PREPROCESS" == true ]]; then
    mkdir -p "${OUT_PATH}/${GUID}/mri/orig"
    cp $RAW_MRI_PATH "${OUT_PATH}/${GUID}/mri/orig/001.mgz"

    echo -e "${CYAN}Step 1 — preprocessing${RESET}"
    ./preprocess.sh "/data/${GUID}/mri/orig/001.mgz" "/data" "$GUID"

    # Build dynamic argument list
    NEUROCBIR_ARGS=(
        --img_path "/data/${GUID}/mri/brain_talairach.nii.gz" \
        --seg_path "/data/${GUID}/mri/aparc+aseg_talairach.nii.gz" \
    )
    [[ -n "$EMB_DATASET_PATH" ]] && NEUROCBIR_ARGS+=(--emb_dataset_path "$EMB_DATASET_PATH")
    [[ -n "$REGION" ]] && NEUROCBIR_ARGS+=(--region "$REGION")
    [[ -n "$SCOPE" ]] && NEUROCBIR_ARGS+=(--scope "$SCOPE")
    [[ -n "$TOP_K" ]] && NEUROCBIR_ARGS+=(--top_k "$TOP_K")
    [[ -n "$DEVICE" ]] && NEUROCBIR_ARGS+=(--device "$DEVICE")
    [[ -n "$INTERNAL_CONFIG" ]] && NEUROCBIR_ARGS+=(--internal_config "$INTERNAL_CONFIG")
    [[ -n "$USER_CONFIG" ]] && NEUROCBIR_ARGS+=(--user_config "$USER_CONFIG")
    [[ -n "$O_PATH" ]] && NEUROCBIR_ARGS+=(--o_path "$O_PATH")

    # Run NeuroCBIR container
    docker compose run --rm neurocbir "${NEUROCBIR_ARGS[@]}"

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

