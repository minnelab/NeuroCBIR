#!/bin/bash
set -e  # Exit on error

# === Colors ===
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

# Source the banner function
source "$(dirname "$0")/banner.sh"

START_TIME=$(date +%s)

# === DEFAULTS ===
# For freesurfer container
FS_LICENSE_PATH="/usr/local/freesurfer/.license"

# For this wrapper
PREPROCESS=false
RAW_MRI_PATH="" # Need to define if --preprocess
O_PATH=""
GUID=""
# For neurocbir package
BRAIN_PATH="" # Do not define if --preprocess
SEG_PATH="" # Do not define if --preprocess
O_PATH=""
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
        --o_path) O_PATH="$2"; shift ;;
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
        --fs_license) FS_LICENSE_PATH="$2" ; shift ;;
        # Help
        -h|--help)
            print_banner
            echo "Usage: $0 [options]"
            echo ""
            echo "Wrapper script to run the NeuroCBIR pipeline using Docker and Docker Compose."
            echo "It can either run a full preprocessing + CBIR workflow or just the CBIR step on preprocessed data."
            echo ""
            echo "Required Arguments:"
            echo "  --o_path <dir>         Path to the main output directory. This will be mounted into the containers."
            echo "  --guid <id>            Unique identifier for the subject/run."
            echo ""
            echo "Workflow Mode (choose one):"
            echo "  1) Preprocessing + CBIR:"
            echo "     --preprocess          Flag to enable the full preprocessing pipeline."
            echo "     --raw_mri_path <file> Path to the raw input T1w MRI. Required with --preprocess."
            echo ""
            echo "  2) CBIR Only:"
            echo "     --brain_path <file>   Path to the preprocessed, skull-stripped brain image."
            echo "     --seg_path <file>     Path to the preprocessed segmentation image (required for --scope region)."
            echo ""
            echo "NeuroCBIR Parameters (for the retrieval step):"
            echo "  --scope <mode>         CBIR scope: 'whole_brain' or 'region'."
            echo "  --region <name>        Region name for region-specific CBIR (e.g., 'Left-Hippocampus')."
            echo "  --top_k <num>          Number of top similar images to retrieve."
            echo "  --device <name>        Computation device ('cpu' or 'cuda')."
            echo "  --emb_dataset_path <file> Path to the embedding dataset (HDF5 file)."
            echo "  --internal_config <file> Path to internal configuration YAML file."
            echo "  --user_config <file>   Path to user configuration YAML file."
            echo "Other Options:"
            echo "  --fs_license <file>    Path to FreeSurfer license file (default: $FS_LICENSE_PATH)."
            echo ""
            echo "Example 1: Preprocessing + CBIR:"
            echo "  ./run_neurocbir.sh docker --preprocess --o_path /output --guid guid \\"
            echo "                        --raw_mri_path /path/to/mri.nii.gz  \\"
            echo "                        --scope region --region Left-Hippocampus --top_k 5 --device cpu \\"
            echo "                        --emb_dataset_path /path/to/embeddings.h5 \\"
            echo "                        --user_config /path/to/user_config.yaml \\"
            echo "                        --internal_config /path/to/internal_config.yaml \\"
            echo "                        --fs_license /path/to/license.txt"
            echo "" 
            echo "Example 2: CBIR Only for --scope whole_brain:"           
            echo "  ./run_neurocbir.sh docker --o_path /output --guid guid \\"
            echo "                        --brain_path /path/to/brain.nii.gz"
            echo "                        --scope whole_brain --top_k 5 --device cpu \\"
            echo "                        --emb_dataset_path /path/to/embeddings.h5 \\"
            echo "                        --user_config /path/to/user_config.yaml \\"
            echo "                        --internal_config /path/to/internal_config.yaml \\"
            echo "                        --fs_license /path/to/license.txt"
            echo "" 
            echo "Example 3: CBIR Only for --scope region:"           
            echo "  ./run_neurocbir.sh docker --o_path /output --guid guid \\"
            echo "                        --brain_path /path/to/brain.nii.gz --seg_path /path/to/seg.nii.gz \\"
            echo "                        --scope region --region Left-Hippocampus --top_k 5 --device cpu \\"
            echo "                        --emb_dataset_path /path/to/embeddings.h5 \\"
            echo "                        --user_config /path/to/user_config.yaml \\"
            echo "                        --internal_config /path/to/internal_config.yaml \\"
            echo "                        --fs_license /path/to/license.txt"
            echo ""
            echo "Note:"
            echo "  - Ensure that the FreeSurfer license file is correctly set in the script (default path: $FS_LICENSE_PATH)."
            echo "  - The output directory (--o_path) will be created if it does not
            exit 0
            ;;
        *) echo -e "${RED}Unknown parameter passed: $1${RESET}"; exit 1 ;;
    esac
    shift
done

# Export variables for docker-compose
export O_PATH="$O_PATH"
export FS_LICENSE_PATH="$FS_LICENSE_PATH"

# === VALIDATION ===
if [[ -z "$O_PATH" || -z "$GUID" ]]; then
    echo -e "${RED}Error: --o_path and --guid are required.${RESET}"
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

# Prepare arguments for neurocbir
NEUROCBIR_ARGS=()

# === PREPROCESSING ===
if [[ "$PREPROCESS" == true ]]; then
    mkdir -p "${O_PATH}/${GUID}/mri/orig"
    cp $RAW_MRI_PATH "${O_PATH}/${GUID}/mri/orig/001.mgz"

    echo -e "${CYAN}Step 1 — preprocessing${RESET}"
    # Assuming preprocess.sh uses docker-compose and O_PATH is mounted as /data
    deploy/scripts/preprocess.sh "/data/${GUID}/mri/orig/001.mgz" "/data" "$GUID"

    # After preprocessing, set the paths for the neurocbir step
    NEUROCBIR_ARGS+=(--brain_path "/data/${GUID}/mri/brain_talairach.nii.gz")
    NEUROCBIR_ARGS+=(--seg_path "/data/${GUID}/mri/aparc+aseg_talairach.nii.gz")
else
    mkdir -p "${O_PATH}/${GUID}/mri"
    cp $BRAIN_PATH "${O_PATH}/${GUID}/mri/brain_talairach.nii.gz"
    if [[ "$SCOPE" == "region" ]]; then
        cp $SEG_PATH "${O_PATH}/${GUID}/mri/aparc+aseg_talairach.nii.gz"
    fi
    NEUROCBIR_ARGS+=(--brain_path "/data/${GUID}/mri/brain_talairach.nii.gz")
    NEUROCBIR_ARGS+=(--seg_path "/data/${GUID}/mri/aparc+aseg_talairach.nii.gz")
fi

# === NEUROCBIR ===
echo -e "${CYAN}Step 2 — neurocbir${RESET}"

# Append all other optional arguments
[[ -n "$EMB_DATASET_PATH" ]] && NEUROCBIR_ARGS+=(--emb_dataset_path "$EMB_DATASET_PATH")
[[ -n "$REGION" ]] && NEUROCBIR_ARGS+=(--region "$REGION")
[[ -n "$SCOPE" ]] && NEUROCBIR_ARGS+=(--scope "$SCOPE")
[[ -n "$TOP_K" ]] && NEUROCBIR_ARGS+=(--top_k "$TOP_K")
[[ -n "$DEVICE" ]] && NEUROCBIR_ARGS+=(--device "$DEVICE")
[[ -n "$INTERNAL_CONFIG" ]] && NEUROCBIR_ARGS+=(--internal_config "$INTERNAL_CONFIG")
[[ -n "$USER_CONFIG" ]] && NEUROCBIR_ARGS+=(--user_config "$USER_CONFIG")
[[ -n "$O_PATH" ]] && NEUROCBIR_ARGS+=(--o_path "/data/${GUID}/neurocbir_report/$SCOPE/") # Map o_path inside container

# Run NeuroCBIR container with all arguments
docker compose -f deploy/docker/docker-compose.yml run --rm neurocbir "${NEUROCBIR_ARGS[@]}"

# === TIMER ===
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo -e "${CYAN}Pipeline completed for $GUID${RESET}"
echo -e "${CYAN}Total time: $TOTAL_TIME seconds ($(date -ud "@$(printf "%.0f" $TOTAL_TIME)" +'%Hh:%Mm:%Ss'))${RESET}"
