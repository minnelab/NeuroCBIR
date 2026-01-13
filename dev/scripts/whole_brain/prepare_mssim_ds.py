import os
from tqdm import main, tqdm
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from dev.preprocessing import LookupNPZDataset, SequentialBatchIterator

from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter, zoom

from joblib import Parallel, delayed

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
    metadata = metadata.dropna(subset=['partition']).reset_index(drop=True)
    logging.info(f"METADATA: Remaining rows: {len(metadata)}") # Check result
    logging.debug(f"METADATA columns: {metadata.columns}") 
    logging.debug(f"METADATA head: {metadata.head()}") 

    # Training configuration
    batch_files = sorted(metadata["batch_file"].unique())
    
    # Select batch_files to ensure n_queries per project and partition (minimize the number of batch files loaded).
    bias_columns = config.get("bias_columns", [])
    bias_combinations = metadata[bias_columns].drop_duplicates().values.tolist()
    
    q_batch_files = []
    for combs in bias_combinations:
        logging.debug(f"Combination: {combs}")
        # Example of groupby: ["ADNI", ]
        metadata_subset = metadata.copy()
        for col, comb in zip(bias_columns, combs):
            metadata_subset = metadata_subset.query(f"{col} == @comb").reset_index(drop=True)
        batch_files_subset = sorted(metadata_subset["batch_file"].unique())
        logging.debug(f"  → Found {len(metadata_subset)} samples in {len(batch_files_subset)} batch files.")
        # Check if in a batch, there are enough samples to extract n_queries per combination
        for batch_file in batch_files_subset:
            if len(metadata_subset.query("batch_file == @batch_file")) >= config["n_queries"]:
                q_batch_files.append([combs, batch_file])
                logging.debug(f"    ✓ Sufficient samples in batch_file: {batch_file}")
                break
            else:
                logging.debug(f"    ✗ Insufficient samples for combination: {combs} in batch_file: {batch_file}")

    # Sanity check to ensure there are enough batch files
    logging.info(f"Batch files for queries: {q_batch_files}")

    logging.info(f"Selected {len(q_batch_files)} batch files for queries.")
    # Get queries
    queries = []
    for combs, q_batch_file in q_batch_files:
        dataset = LookupNPZDataset(metadata, batch_file=q_batch_file, use_segmentation=False)
        metadata_subset = dataset.metadata.copy()
        for col, comb in zip(bias_columns, combs):
            metadata_subset = metadata_subset.query(f"{col} == @comb")
        # get possible query indices
        possible_indices = metadata_subset.index.tolist()
        i_qs = np.random.choice(possible_indices, size=config["n_queries"], replace=False)
        for i_q in i_qs:
            q = dataset[i_q].copy()
            q["image"] = preprocess(q["image"].detach().cpu().numpy().astype(np.float32)[0])
            q["pyramid"] = build_pyramid(q["image"])
            queries.append(q)
    logging.info(f"Selected total {len(queries)} queries for MS-SSIM computation.")
            
    # Compute MS-SSIM: pairwise comparison between each query and the rest of the dataset
    res_mssim = {}
    for query in queries:
        q_guid = query["GUID"]
        res_mssim[q_guid] = {"r_guid": [], "ms_ssim": []}


    for i, batch_file in enumerate(batch_files):
        dataset = LookupNPZDataset(metadata, batch_file=batch_file, use_segmentation=False)
        logging.info(f"Created dataset for batch_file: {batch_file} with {len(dataset)} samples")

        for data in tqdm(dataset, total=len(dataset), ncols=100):  # loop over several batches per file
            r_guid = data["GUID"]
            r_image = preprocess(data["image"].detach().cpu().numpy().astype(np.float32)[0])
            r_pyramid = build_pyramid(r_image)

            # For each query compute MS-SSIM
            # for query in queries:
            #     q_guid = query["GUID"]
            #     q_image = query["image"]

            #     val_ms_ssim = ms_ssim(q_image, r_image)
            #     res_mssim[q_guid]["r_guid"].append(r_guid)
            #     res_mssim[q_guid]["ms_ssim"].append(val_ms_ssim)
            results = Parallel(config["n_jobs"])(delayed(compute_for_query)(q, r_guid, r_pyramid) for q in queries)

            for q_guid, r_guid, val in results:
                res_mssim[q_guid]["r_guid"].append(r_guid)
                res_mssim[q_guid]["ms_ssim"].append(val)

    logging.info(f"Computation completed!")
    
    # Convert results to a flat table
    rows = []
    for q_guid, values in res_mssim.items():
        r_guids = values["r_guid"]
        ms_ssims = values["ms_ssim"]

        for r_guid, score in zip(r_guids, ms_ssims):
            rows.append({
                "query_guid": q_guid,
                "reference_guid": r_guid,
                "ms_ssim": score
            })
    df = pd.DataFrame(rows)

    # Save res_mssim to csv
    output_file = os.path.join(config["output_dir"], config["output_file_name"])
    df.to_csv(output_file, index=False)
    logging.info(f"Saved to csv!: {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = {
        "random_state": 1234,
        "data_path": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/",
        "output_dir": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/original/",
        # "data_path": "/mnt/alvis2/mimer_user/batched_datasets/",
        # "output_dir": "/mnt/alvis2/mimer_user/batched_datasets/original/",
        "dataset_index_file_name": "original/dataset_index.csv",
        "metadata_file_name": "combined_metadata.csv",
        "n_queries": 30,
        "n_jobs": -1,
        "output_file_name": "ms_ssim_sample2.csv",
        "bias_columns": [
            "partition",
            "project",
            ]
    }

    main(config)

    
