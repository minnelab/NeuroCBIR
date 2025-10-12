# NeuroCBIR Python Package

This directory contains the source code for the `neurocbir` Python package, a command-line tool for Content-Based Image Retrieval in neuroimaging.

---

## 📦 Installation

To install the package, navigate to this directory and use `pip`. It is recommended to perform the installation within a Python virtual environment.

1.  **Navigate to the package directory:**
    ```bash
    # Assuming you are in the root of the NeuroCBIR project
    cd deploy/neurocbir
    ```

2.  **Install the package:**
    This command will install `neurocbir` and all its dependencies. It also creates a command-line tool named `neurocbir` that you can run from your terminal.
    ```bash
    pip install .
    ```

---

## 🚀 Usage

Once installed, you can use the `neurocbir` command-line tool from anywhere.

### View Help

To see all available commands and options, run:
```bash
neurocbir --help
```

### Example: Whole-Brain Retrieval

Run a query on a whole-brain image. The command will use default configurations unless you provide overrides.
```bash
neurocbir --scope whole_brain --img_path /path/to/your/brain_image.nii.gz
```

### Example: Region-Specific Retrieval

Run a query on a specific brain region, such as the "Left-Hippocampus".
```bash
neurocbir --scope region --region "Left-Hippocampus" --img_path /path/to/your/brain_image.nii.gz --seg_path /path/to/your/segmentation.nii.gz
```