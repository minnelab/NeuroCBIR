import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import umap
import numpy as np
from monai.utils import set_determinism

from preprocessing import EmbBatchedDataset, SequentialBatchIterator

def main(config):
    set_determinism(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load metadata
    index_ds = pd.read_csv(os.path.join(config["data_path"], config["dataset_index_file_name"]))
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file_name"]))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner")
    metadata['subject'].replace('', pd.NA, inplace=True)
    metadata = metadata.dropna(subset=['subject']).reset_index(drop=True)

    # Get unique .npz files to loop over
    batch_files = sorted(metadata["batch_file"].unique())

    # Collect raw embeddings
    all_raw_embs = []
    all_guids = []

    for batch_file in batch_files:
        dataset = EmbBatchedDataset(metadata, batch_files=batch_file)
        batch_iter = SequentialBatchIterator(dataset, batch_size=config["batch_size"], device=device)

        progress_bar = tqdm(batch_iter, total=len(batch_iter), ncols=150)
        progress_bar.set_description(f'Collecting {os.path.basename(batch_file)}')

        for batch in progress_bar:
            embs, guids = batch["emb"], batch["guid"]
            all_raw_embs.append(embs.cpu())
            all_guids.extend(guids)

    # Stack raw embeddings
    raw_embs_tensor = torch.cat(all_raw_embs, dim=0)  # [N, D]
    raw_embs_np = raw_embs_tensor.numpy()

    # Apply UMAP
    reducer = umap.UMAP(n_components=32, random_state=42)
    umap_embs = reducer.fit_transform(raw_embs_np)  # [N, 32]

    # Create DataFrame and save
    df_embs = pd.DataFrame(umap_embs)
    df_embs.insert(0, "GUID", all_guids)
    df_embs.columns = df_embs.columns.astype(str)

    output_path = os.path.join(config["output_dir"], "umap_embeddings.parquet")
    df_embs.to_parquet(output_path, index=False)
    print(f"✅ UMAP embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    main(config)
