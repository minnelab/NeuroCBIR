#!/bin/bash
set -e

# Color codes for better output
CYAN="\033[36m"
RED="\033[31m"
YELLOW="\033[33m"
RESET="\033[0m"

# === CONFIGURATION ===
DOCKER_IMAGE_NAME="neurocbir"
DOCKER_IMAGE_TAG="latest"
DOCKERFILE_PATH="deploy/docker/Dockerfile"

# Define required files and directories that the user must download manually.
REQUIRED_PATHS=(
    "deploy/data/data_private/region_brain/cl_ckpt.pth"
    "deploy/data/data_private/region_brain/vae_ckpt.pth"
    "deploy/data/data_private/region_brain/projected_embeddings.parquet"
    "deploy/data/data_private/whole_brain/cl_ckpt.pth"
    "deploy/data/data_private/whole_brain/vae_ckpt.pth"
    "deploy/data/data_private/whole_brain/projected_embeddings.parquet"
)

# === PRE-BUILD CHECK ===
echo -e "${CYAN}🔍 Verifying required data files...${RESET}"

ALL_FILES_PRESENT=true
for path in "${REQUIRED_PATHS[@]}"; do
    if [ ! -f "$path" ]; then
        echo -e "${RED}Error: Required file not found at: $path${RESET}"
        ALL_FILES_PRESENT=false
    fi
done

if [ "$ALL_FILES_PRESENT" = false ]; then
    echo -e "\n${YELLOW}Please download the required model weights and embedding files and place them in the 'deploy/data/data_private/' directory.${RESET}"
    echo -e "${YELLOW}The expected directory structure is:${RESET}"
    cat << 'EOF'

deploy/data/
└── data_private/
    ├── region_brain/
    │   ├── cl_ckpt.pth
    │   ├── projected_embeddings.parquet
    │   └── vae_ckpt.pth
    └── whole_brain/
        ├── cl_ckpt.pth
        ├── projected_embeddings.parquet
        └── vae_ckpt.pth

EOF
    exit 1
fi

echo -e "${CYAN}✅ All required data files are present.${RESET}"

# === DOCKER BUILD ===
echo -e "\n${CYAN}🚀 Building Docker image: ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}...${RESET}"

docker build -t "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}" -f "${DOCKERFILE_PATH}" .

echo -e "\n${CYAN}✅ Docker image built successfully!${RESET}"
echo -e "${CYAN}You can now run the container, for example:${RESET}"
echo -e "   docker run --rm ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} --help"