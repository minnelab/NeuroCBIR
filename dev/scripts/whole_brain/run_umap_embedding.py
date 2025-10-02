import os
import json
import argparse
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from utils import load_config_from_path

import torch
import umap
import numpy as np

from preprocessing import EmbBatchedDataset

def main(config):
    seed=config["random_state"]
    os.makedirs(config["output_dir"], exist_ok=True)

    if config["fit_project"]:

        # Load metadata
        index_ds = pd.read_csv(os.path.join(config["data_path"], config["dataset_index_file_name"]))
        clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file_name"]))
        metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner")
        metadata['subject'].replace('', pd.NA, inplace=True)
        metadata = metadata.dropna(subset=['subject']).reset_index(drop=True)

        # Filter by partition if specified
        partition = config.get("partition", None)
        if partition is not None:
            if partition.lower() not in ["train", "test"]:
                raise ValueError(f"Invalid partition value: {partition}. Must be 'train', 'test', or not set.")
            metadata = metadata[metadata["partition"].str.lower() == partition.lower()].reset_index(drop=True)
            print(f"🔍 Using partition: {partition} | {len(metadata)} records selected.")
        else:
            print(f"🔍 No partition specified. Using all data | {len(metadata)} records total.")

        # Load dataset
        batch_files = sorted(metadata["batch_file"].unique())
        dataset = EmbBatchedDataset(metadata, batch_files=batch_files)
        raw_embs_np = dataset.embs
        raw_embs_np = raw_embs_np.reshape([raw_embs_np.shape[0], -1]) # [N, D]
        all_guids = dataset.guids

        # Fit PCA
        print("Fitting PCA.")
        pca = PCA(n_components=config["pca_params"]["n_components"], 
                random_state=seed)
        embs_pca = pca.fit_transform(raw_embs_np)  # [N, 100]
        # Save the PCA model to disk
        joblib.dump(pca, os.path.join(config["output_dir"], "pca_model.pkl"))

        # Fit UMAP
        print("Fitting UMAP.")
        reducer = umap.UMAP(n_components=config["umap_params"]["n_components"], 
                            verbose=config["umap_params"]["verbose"],
                            random_state=seed)
        reducer.fit(embs_pca) 

        # Save the model
        joblib.dump(reducer, os.path.join(config["output_dir"], "umap_model.pkl"))
        print(f"✅ UMAP model saved.")

        del dataset, metadata, raw_embs_np, all_guids, embs_pca

    else:
            # Load PCA and UMAP models
            pca = joblib.load(os.path.join(config["output_dir"], "pca_model.pkl"))
            reducer = joblib.load(os.path.join(config["output_dir"], "umap_model.pkl"))

    # Obtaining the embeddings
    # Load metadata
    index_ds = pd.read_csv(os.path.join(config["data_path"], config["dataset_index_file_name"]))
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file_name"]))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner")
    metadata['subject'].replace('', pd.NA, inplace=True)
    metadata = metadata.dropna(subset=['subject']).reset_index(drop=True)

    # Load dataset
    batch_files = sorted(metadata["batch_file"].unique())
    dataset = EmbBatchedDataset(metadata, batch_files=batch_files)
    raw_embs_np = dataset.embs
    raw_embs_np = raw_embs_np.reshape([raw_embs_np.shape[0], -1]) # [N, D]
    all_guids = dataset.guids

    # Apply UMAP
    print("Computing UMAP embs.")
    pca_embs = pca.transform(raw_embs_np)
    umap_embs = reducer.transform(pca_embs)

    # Create DataFrame and save
    df_embs = pd.DataFrame(umap_embs)
    df_embs.insert(0, "GUID", all_guids)
    df_embs.columns = df_embs.columns.astype(str)

    output_path = os.path.join(config["output_dir"], "projected_embeddings.parquet")
    df_embs.to_parquet(output_path, index=False)
    print(f"✅ UMAP embeddings saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)