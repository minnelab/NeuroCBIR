# NeuroCBIR development

*NeuroCBIR: A Public Image Retrieval System for Whole-Brain and Region-Specific MRI.*

---

## Overview

**NeuroCBIR** is an open neuroimaging framework for **content-based image retrieval (CBIR)** on structural MRI data.
It supports both **whole-brain** and **region-specific** searches across clinical datasets.

This sections is only focused on reaching reproducible results.

---

## Whole-brain

### VAE-training


```mermaid
flowchart TD
    A[Start run_reproducibility_check.py] --> C[Parse Arguments]
    C --> D[Prepare Mock Dataset<br/><code>prepare_mock_dataset.py</code>]
    D --> E[Create Data Index<br/><code>create_data_index.py</code>]

    E --> F[directory: <br/><code>scripts.whole_brain</code>]
    E --> G[directory: <br/><code>scripts.region_brain</code>]

    %% Whole-Brain Subgraph
    subgraph F_group[Whole-Brain Pipeline]
        F --> F1[Train VAE<br/><code>train_autoencoder</code>]
        F1 --> F2[Extract VAE Embedding<br/><code>run_vae_embedding</code>]
        F2 --> F3[<code>create_data_index</code>]
        F3 --> F4[Train CL Model<br/><code>train_contrastive_model</code>]
        F4 --> F5[Extract CL Embedding<br/><code>run_cl_embedding</code>]
        F5 --> F6[Evaluate<br/><code>run_cbir_eval</code>]
    end

    %% Region-Brain Subgraph
    subgraph G_group[Region-Brain Pipeline]
        G --> G1[Create Bounding Boxes<br/><code>create_bounding_boxes.py</code>]
        G1 --> G2[Create Fake labels.csv]
        G2 --> G3[Train VAE<br/><code>train_autoencoder.py</code>]
        G3 --> G4[Extract VAE Embedding<br/><code>run_vae_embedding.py</code>]
        G4 --> G5[<code>create_data_index.py</code>]
        G5 --> G6[Train CL Model<br/><code>train_contrastive_model.py</code>]
        G6 --> G7[Extract CL Embedding<br/><code>run_cl_embedding</code>]
        G7 --> G8[Evaluate<br/><code>run_cbir_eval.py</code>]
    end

    F6 --> Z[End]
    G8 --> Z
```
