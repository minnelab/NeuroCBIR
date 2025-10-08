# NeuroCBIR — Docker Setup & Usage Guide

This document summarizes the necessary commands to build and run **NeuroCBIR** using Docker and `docker-compose`.

---

## 📁 Project Structure



---

## 🛠 Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (latest stable version)
- [Docker Compose](https://docs.docker.com/compose/install/) (latest stable version)
- Ensure `.dockerignore` exists in the project root to optimize builds
- Ensure your the data directory looks like that:

```
NeuroCBIR/
├── deploy/
│   ├── configs/
│   │   └── ...
│   ├── data/
│   │   ├── data_private
│   │   │   ├── example
│   │   │   │   └── OAS30001_MR_d0129
│   │   │   │       └── mri
│   │   │   │           ├── align_aparc+aseg.nii.gz
│   │   │   │           └── align_norm.nii.gz
│   │   │   ├── region_brain
│   │   │   │   ├── cl_ckpt.pth
│   │   │   │   ├── projected_embeddings.parquet
│   │   │   │   ├── README.md
│   │   │   │   └── vae_ckpt.pth
│   │   │   └── whole_brain
│   │   │       ├── cl_ckpt.pth
│   │   │       ├── projected_embeddings.parquet
│   │   │       ├── README.md
│   │   │       └── vae_ckpt.pth
│   │   ├── bounding_boxes.csv
│   │   └── labels.csv
│   ├── docker/
│   │   └── ...
│   ├── infra/
│   │   └── ...
│   └── neurocbir/
│       └── ...
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Python package

## Preprocessing
```bash
docker compose run --rm freesurfer \
mri_synthseg \
    --i "/data/data_private/example/OAS30001_MR_d0129/mri/orig/001.mgz" \
    --o "/data/data_private/example/OAS30001_MR_d0129/mri/aparc+aseg.mgz" \
    --fast \
    --cpu \
    --threads 8 \
    --parc
```
```bash
docker compose run --rm ants N4BiasFieldCorrection -i /data/data_private/example/OAS30001_MR_d0129/mri/orig/001.mgz -o /data/data_private/example/OAS30001_MR_d0129/mri/orig_nu.mgz
```


```bash
mri_synthseg \
  --i /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/example/OAS30001_MR_d0129/mri/orig/001.mgz \
  --o /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/output_synthseg/seg.mgz \
  --fast \
  --cpu \
  --threads 8 \
  --parc
```

```bash
docker run  -v /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/example:/app/data \
            -v /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/output:/app/output \
            -v /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/fs_license:/app/fs_license \
            -it --rm \
            --user $(id -u):$(id -g) \
            deepmi/fastsurfer:cpu-v2.4.2 \
            --seg_only \
            --no_cereb \
            --no_hypothal \
            --t1 /app/data/OAS30001_MR_d0129/mri/orig/001.mgz \
            --sid OAS30001_MR_d0129 \
            --sd /app/output \
            --parallel \
            --threads 10 \
            --batch 12 \
            --fs_license /app/fs_license/license.txt
```

```bash
docker run  -v /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/example:/app/data \
            -v /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/output:/app/output \
            -v /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/fs_license:/app/fs_license \
            -it --rm \
            --entrypoint /bin/bash \
            --user $(id -u):$(id -g) \
            deepmi/fastsurfer:cpu-v2.4.2
```

```bash
./run_neurocbir.sh \
    --preprocess \
    --out_path /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/test_1 \
    --raw_mri_path /home/fenda/Work/202501__NeuroCBIR/NeuroCBIR/tmp/example/OAS30001_MR_d0129/mri/orig/001.mgz \
    --guid subject1
```


## Docker

If you haven’t cloned NeuroCBIR yet:
```bash
git clone git@github.com:feniede/NeuroCBIR.git
cd NeuroCBIR
```
In NeuroCBIR folder, build the docker image:
```bash
docker build -t neurocbir:latest -f deploy/docker/Dockerfile .
```
To run the example:
```bash
docker run --rm neurocbir:latest
```
To enter bash session:
```bash
docker run -it --rm neurocbir:latest bash
```
To run with docker directly:
```bash
docker run --rm \
  --name neurocbir_test \
  -v /mnt/kth_cbh/fenda/Datasets/OASIS3/oasis3:/app/data/data_mnt \
  neurocbir:latest \
  --img_path "/app/data/data_mnt/OAS30005_MR_d0143/mri/align_norm.nii.gz" \
  --seg_path "/app/data/data_mnt/OAS30005_MR_d0143/mri/align_aparc+aseg.nii.gz" \
  --scope "region" \
  --region "Left-Hippocampus" \
  --top_k 30
```

## Docker-Compose