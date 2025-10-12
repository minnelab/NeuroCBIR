# NeuroCBIR

A Public Content-Based Image Retrieval System for Whole-Brain and Region-Specific MRI Across Multiple Clinical Cohorts.

---

----
## 📖 Overview

This project provides a suite of tools for content-based image retrieval (CBIR) on brain MRI scans. It can be used for both whole-brain and region-specific queries. The project is designed to be flexible and can be used in several ways:

1.  **As a Python Package**: For easy integration into other Python projects and direct command-line use.
2.  **With Docker**: For a portable, containerized environment that encapsulates all dependencies.
3.  **With Snakemake & Singularity**: For running reproducible, end-to-end pipelines, including preprocessing of raw MRI data.

---

## 🛠 Prerequisites

Before you begin, ensure you have the following installed:

- **Git**: To clone the repository.
- **Python**: Version 3.10 or higher.
- **Docker**: For the Docker-based workflow.
- **Apptainer (or Singularity)**: For the Snakemake-based workflow.

### Required Data

This project requires pre-trained model weights and embedding datasets, which must be downloaded manually. After cloning the repository, you must place these files into the `deploy/data/data_private/` directory.

The expected directory structure is:
```
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
```

---

## 🚀 Installation and Usage

First, clone the repository:
```bash
git clone https://github.com/feniede/NeuroCBIR.git
cd NeuroCBIR
```

Choose one of the following methods to run the application.

### Option 1: As a Python Package

This method installs `neurocbir` as a command-line tool in your Python environment. It's ideal for quick queries and integration into other scripts.

1.  **Navigate to the package directory:**
    ```bash
    cd deploy/neurocbir
    ```

2.  **Install the package:**
    ```bash
    pip install .
    ```

3.  **Run a query:**
    Once installed, you can use the `neurocbir` command from anywhere.

    *   **Whole-Brain Retrieval:**
        ```bash
        neurocbir --scope whole_brain --img_path /path/to/your/brain_image.nii.gz
        ```
    *   **Region-Specific Retrieval:**
        ```bash
        neurocbir --scope region --region "Left-Hippocampus" --img_path /path/to/your/brain_image.nii.gz --seg_path /path/to/your/segmentation.nii.gz
        ```

### Option 2: Using Docker

This method provides a self-contained environment with all dependencies included. It's recommended for ensuring consistent execution across different machines.

1.  **Build the Docker image:**
    Use the provided script, which verifies that all required data is present before building.
    ```bash
    chmod +x build_docker.sh
    ./build_docker.sh
    ```

2.  **Run a query:**
    Use `docker run` to execute a command inside the container. You must mount any external data directories (like your subject data) into the container.

    ```bash
    docker run --rm \
      -v /path/to/your/subjects:/data/subjects \
      neurocbir:latest \
        --scope region \
        --region "Left-Hippocampus" \
        --img_path /data/subjects/OAS30005_MR_d0143/mri/align_norm.nii.gz \
        --seg_path /data/subjects/OAS30005_MR_d0143/mri/align_aparc+aseg.nii.gz
    ```

### Option 3: Using Snakemake with Singularity

This is the most comprehensive method, designed for running reproducible, end-to-end pipelines that include preprocessing of raw data.

1.  **Run the setup script:**
    This script will create a Python virtual environment, install Snakemake, and build the required Singularity container.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

2.  **Configure the pipeline:**
    Edit the `deploy/snakemake/config.yaml` file to specify your input data (`raw_mri_path`), output directory (`outdir`), and other parameters.

3.  **Run the pipeline:**
    Execute the pipeline using the provided run script.
    ```bash
    chmod +x run_snakemake.sh
    ./run_snakemake.sh
    ```
    This will automatically handle preprocessing, feature extraction, and retrieval according to your configuration.

---

## 🧠 Preprocessing

The Snakemake pipeline (Option 3) is the recommended method for preprocessing raw MRI data. It provides a structured and reproducible workflow that handles all necessary steps, from segmentation to normalization.

To run preprocessing, simply configure the `raw_mri_path` in `deploy/snakemake/config.yaml` and execute the pipeline as described above.

