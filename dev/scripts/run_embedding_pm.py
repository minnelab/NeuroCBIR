'''
This script is to extract the features from pretrained models, to perform the multicomparisons.
'''

import argparse
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging

from monai.utils import set_determinism
import torch

from dev.utils import load_config_from_path


def process_batch(data, encoder, device, batch_size=16):
    """
    Process images in batches through the encoder.

    Args:
        data: dict with keys "images" (list/array of images) and "GUID" (list of IDs).
        encoder: torch.nn.Module model.
        device: torch.device.
        batch_size: number of images per batch.

    Returns:
        embs: numpy array of shape (N, D) with embeddings.
        ids: numpy array of shape (N,) with sample IDs.
    """
    images = data['images']
    sample_ids = data['GUID']

    embs = []
    ids = []

    # iterate in batches
    for start in tqdm(range(0, len(images), batch_size), ncols=150, desc="Processing batches"):
        end = start + batch_size
        batch_imgs = images[start:end]
        batch_ids = sample_ids[start:end]

        # preprocess batch: (B, C, H, W, D)
        batch_tensors = []
        for img in batch_imgs:
            img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # add channel, normalize
            batch_tensors.append(torch.tensor(img))

        batch_tensors = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            batch_embs = encoder(batch_tensors)  # (B, D)
            batch_embs = torch.flatten(batch_embs, start_dim=1)
        
        batch_embs = batch_embs.cpu().numpy().astype(np.float16)

        embs.append(batch_embs)
        ids.extend(batch_ids)

    embs = np.concatenate(embs, axis=0)
    ids = np.array(ids)

    return embs, ids

def main(config):
    set_determinism(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = config["data_path"]

    # Load metadata
    index_ds = pd.read_csv(os.path.join(data_path, config["dataset_index_file_name"]))
    clinical_ds = pd.read_csv(os.path.join(data_path, config["metadata_file_name"]))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner") # Merge on the 'GUID' column
    logging.info(f"METADATA: Original rows: {len(metadata)}")
    metadata['subject'].replace('', pd.NA, inplace=True) # First, ensure empty strings are treated as NaN
    metadata = metadata.dropna(subset=['subject']) # Then drop rows where subject is NaN
    metadata = metadata.reset_index(drop=True) # Reset index
    logging.info(f"METADATA: Remaining rows: {len(metadata)}") # Check result
    # Load VAE
    encoder = config["pretrained_encoder"].to(device)
    encoder.eval()

    # Training configuration
    batch_files = sorted(metadata["batch_file"].unique())

    # Evaluate and embed all batches
    all_proj_embs = []
    all_guids = []

    for batch_file in batch_files:

        # file_to_load = os.path.join(config["load_path"], file_path, file_name)
        logging.info(f"Processing {batch_file}")
        data = np.load(batch_file, allow_pickle=True)

        embs_batch, ids_batch = process_batch(data=data,
                                              encoder=encoder,
                                              device=device,
                                              batch_size=config["batch_size"]
                                             )

        all_proj_embs.append(embs_batch)
        all_guids.extend(ids_batch)  # list of strings

    # Stack everything into a DataFrame and save
    embs_tensor = np.vstack(all_proj_embs)  # shape: [N, D]
    all_guids = np.vstack(all_guids)

    df_embs = pd.DataFrame(embs_tensor)
    df_embs.insert(0, "GUID", all_guids)

    df_embs.columns = df_embs.columns.astype(str)
    output_path = os.path.join(config["output_dir"], "projected_embeddings.parquet")
    df_embs.to_parquet(output_path, index=False)
    logging.info(f"✅ Embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)
