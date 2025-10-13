# NeuroCBIR

*A Public Content-Based Image Retrieval System for Whole-Brain and Region-Specific MRI Across Multiple Clinical Cohorts.*

---

## Overview

**NeuroCBIR** is an open neuroimaging framework for **content-based image retrieval (CBIR)** on structural MRI data.  
It supports both **whole-brain** and **region-specific** searches across clinical datasets.

It can be used in multiple ways:

1. **Docker workflow** – run the full pipeline (preprocessing + CBIR) in a reproducible containerized environment.
2. **Snakemake workflow** – execute the same pipeline using Snakemake + Singularity (Apptainer) for HPC or cluster environments.
3. **Python package** – import and extend the NeuroCBIR logic in your own Python workflows.

---

## Authors

Félix Nieto-del-Amor¹, Jingru Fu¹, J.-Sebastian Muehlboeck², Eric Westman²,  
Daniel Ferreira², Rodrigo Moreno¹  

¹ Division of Biomedical Imaging, KTH Royal Institute of Technology  
² Division of Clinical Geriatrics, Karolinska Institute  

📧 **Contact:** fenda@kth.se

---

## System Requirements

| Component | Minimum Version | Purpose |
|------------|-----------------|----------|
| **Python** | ≥ 3.10 | For setup and optional local runs |
| **Docker** | ≥ 24.0 | For the default containerized workflow |
| **Apptainer / Singularity** | ≥ 1.2 | For Snakemake-based HPC workflows |
| **Git** | Any recent | To clone the repository |

---

## Directory Structure

```
deploy/
├── configs/                # Default configs (internal & user)
├── data/                   # Metadata and pretrained model data
├── docker/                 # Dockerfiles and Compose setup
├── neurocbir/              # Python package implementation
├── scripts/                # Setup & execution scripts
└── snakemake/              # Snakemake pipeline (rules, profiles, configs)
```

---

## Setup

Clone the repository and prepare the environment:

```bash
git clone https://github.com/feniede/NeuroCBIR.git
cd NeuroCBIR
```

Run the setup script corresponding to your workflow:

### Docker Mode
```bash
bash deploy/scripts/setup_docker.sh
```

### Snakemake Mode
```bash
bash deploy/scripts/setup_snakemake.sh
```

These scripts will:
- Check for required dependencies (Docker, Python, Singularity, etc.)
- Build or pull necessary containers
- Prepare configuration files
- Validate the installation

---

## Usage

### Run with Docker
```bash
./run_neurocbir.sh docker --o_path /output --guid SUBJECT_ID --preprocess --raw_mri_path /path/to/mri.nii.gz
```
For CBIR only:
```bash
./run_neurocbir.sh docker --o_path /output --guid SUBJECT_ID --brain_path /path/to/brain.nii.gz --seg_path /path/to/seg.nii.gz --scope region --region Left-Hippocampus
```

### Run with Snakemake
```bash
./run_neurocbir.sh snakemake --config deploy/snakemake/config.yaml --cores 4 --guid SUBJECT_ID
```

---

## Contact

For questions or support, contact **Félix Nieto-del-Amor** at: fenda@kth.se

