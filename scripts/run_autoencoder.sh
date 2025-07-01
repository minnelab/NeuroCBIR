#!/bin/bash
#SBATCH --job-name=autoencoder
#SBATCH --output=logs/autoencoder_%j.out
#SBATCH --error=logs/autoencoder_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1             # or however many you need
#SBATCH --time=24:00:00      # max time
#SBATCH --mem=64G            # or whatever you need
#SBATCH --cpus-per-task=4

# Activate your virtual environment
source /mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/venv_NeuroCBIR/bin/activate

# Run your training script
python -m scripts.train_autoencoder_puglisi
