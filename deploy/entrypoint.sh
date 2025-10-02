#!/usr/bin/env bash
set -euo pipefail

# Basic wrapper so container can be run as:
# docker run --gpus all -v $(pwd)/data:/app/data image -- --img1 /app/data/a.nii.gz --img2 /app/data/b.nii.gz --model /app/data/classifier.pth

# If first arg is --* then call infer.py with all args
python /app/infer.py "$@"
