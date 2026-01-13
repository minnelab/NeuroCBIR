import os
from tqdm import main, tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from dev.preprocessing import LookupNPZDataset, SequentialBatchIterator

from dev.preprocessing.load_dataset import EmbBatchedDataset
from scipy.ndimage import gaussian_filter, zoom

from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity


def compute_for_query(query, r_guid, r_pyramid):
    val = np.mean([
        ssim(a, b, data_range=1.0)
        for a, b in zip(query["pyramid"], r_pyramid)
    ])
    return query["GUID"], r_guid, val


def build_pyramid(img, levels=3):
    pyr = [img]
    for _ in range(1, levels):
        img = gaussian_filter(img, sigma=1)
        img = zoom(img, 0.5, order=1)
        if min(img.shape) < 16:
            break
        pyr.append(img)
    return pyr

def ms_ssim_from_pyramids(pyr1, pyr2):
    return np.mean([
        ssim(a, b, data_range=1.0)
        for a, b in zip(pyr1, pyr2)
    ])


def ms_ssim(img1, img2, levels=3):
    scores = []

    for _ in range(levels):
        scores.append(
            ssim(img1, img2, data_range=1.0)
        )

        # Gaussian blur BEFORE downsampling
        img1 = gaussian_filter(img1, sigma=1)
        img2 = gaussian_filter(img2, sigma=1)

        # Proper downsampling
        img1 = zoom(img1, 0.5, order=1)
        img2 = zoom(img2, 0.5, order=1)

        if min(img1.shape) < 16:
            break

    return float(np.mean(scores))

def preprocess(img):
    return zoom(img, 0.5, order=1)

def main(config):
    
    # Startup config
    np.random.seed(seed=config["random_state"])
    logging.info("Starting training with config: %s", config)
    seed = config["random_state"]
    data_path = config["data_path"]

    # Load metadata
    index_ds = pd.read_csv(os.path.join(config["data_path"], config["dataset_index_file_name"]))
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file_name"]))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner")
    logging.info(f"METADATA: Original rows: {len(metadata)}")

    # Filter out rows
    metadata = metadata.query("repet == 1").reset_index(drop=True)
    metadata = metadata.query("useable == 1").reset_index(drop=True)
    metadata = metadata.query("mislabel == 0").reset_index(drop=True)
    metadata['subject'].replace('', pd.NA, inplace=True)
    metadata = metadata.dropna(subset=['subject']).reset_index(drop=True)
    logging.debug(f"METADATA: Remaining rows: {len(metadata)}") # Check result

    # Training configuration
    batch_files = sorted(metadata["batch_file"].unique())

    # Get queries
    dataset = EmbBatchedDataset(metadata, batch_files=batch_files[0])
    # LookupNPZDataset(metadata, batch_file=batch_files[0], use_segmentation=False)
    i_qs = np.random.randint(0, high=len(dataset), size=config["n_queries"], dtype=int)
    queries = []
    for i_q in i_qs:
        q = dataset[i_q].copy()
        q["mus"] = q["emb"].detach().cpu().numpy().astype(np.float32)
        queries.append(q)
    
    # Compute MS-SSIM: pairwise comparison between each query and the rest of the dataset
    res_sim = {}
    for query in queries:
        q_guid = query["guid"]
        res_sim[q_guid] = {"r_guid": [], "sim": []}


    for i, batch_file in enumerate(batch_files):
        dataset = EmbBatchedDataset(metadata, batch_files=batch_file)
        # dataset = LookupNPZDataset(metadata, batch_file=batch_file, use_segmentation=False)
        logging.info(f"Created dataset for batch_file: {batch_file} with {len(dataset)} samples")

        for data in tqdm(dataset, total=len(dataset), ncols=180):  # loop over several batches per file
            r_guid = data["guid"]
            r_mus = data["emb"].detach().cpu().numpy().astype(np.float32).flatten()

            # For each query compute MS-SSIM
            for query in queries:
                q_guid = query["guid"]
                q_mus = query["mus"].flatten()

                val_sim = cosine_similarity([q_mus], [r_mus])[0][0]
                res_sim[q_guid]["r_guid"].append(r_guid)
                res_sim[q_guid]["sim"].append(val_sim)

    logging.info(f"Computation completed!")
    
    # Convert results to a flat table
    rows = []
    for q_guid, values in res_sim.items():
        r_guids = values["r_guid"]
        sims = values["sim"]

        for r_guid, score in zip(r_guids, sims):
            rows.append({
                "query_guid": q_guid,
                "reference_guid": r_guid,
                "sim": score
            })
    df = pd.DataFrame(rows)
    print(df.head())

    # Save res_sim to csv
    output_file = os.path.join(config["output_dir"], config["output_file_name"])
    df.to_csv(output_file, index=False)
    logging.info(f"Saved to csv!: {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = {
        "random_state": 1234,
        "data_path": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/",
        "output_dir": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/whole_brain/",
        "dataset_index_file_name": "whole_brain/dataset_index.csv",
        "metadata_file_name": "combined_metadata.csv",
        "n_queries": 100,
        "n_jobs": -1,
        "output_file_name": "vae_sim.csv",
    }

    main(config)
