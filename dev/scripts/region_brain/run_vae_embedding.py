# run_vae_embedding.py

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.utils import set_determinism
from preprocessing.load_dataset import SubCorBatDataset, SequentialBatchIterator
from utils import load_config_from_path

def load_vae_model(config, device, ckpt_path = None, use_old_state_dict = False):
    # VAE initialization
    vae_params = config["vae_params"]
    autoencoder = AutoencoderKL(**vae_params).to(config["device"])

    # Load weights (if checkpoint is provided)
    if ckpt_path:
        # Load full checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_key = config.get("ckpt_key", "autoencoder_state_dict")
        # Load only the autoencoder weights
        if use_old_state_dict:
            autoencoder.load_old_state_dict(checkpoint[ckpt_key])
            print("Loaded weights using load_old_state_dict().")
        else:
            autoencoder.load_state_dict(checkpoint[ckpt_key])
            print("Loaded weights using load_state_dict().")

    return autoencoder

def process_batch(sample, autoencoder, device, batch_size=16):
    z_mus = []

    images = sample["image"].to(device)  # shape: (B, C, H, W, D) or similar
    images = images.reshape(-1, *images.shape[2:])  # flatten all images
    guid = sample["GUID"][0]  # assuming same for all images in this batch
    struct_names = sample["struct_name"][0]  # list of names, length = num_images
    struct_map_id = sample["struct_map_id"][0]


    total = images.shape[0]
    for i in range(0, total, batch_size):
        img_batch = images[i:i + batch_size]

        with torch.no_grad():
            z_mu_batch, _ = autoencoder.encode(img_batch)  # shape: (B', D)

        z_mu_batch = z_mu_batch.cpu().numpy().astype(np.float16)
        z_mus.append(z_mu_batch)

    # Concatenate all batched z_mu results
    z_mus = np.concatenate(z_mus, axis=0).tolist()

    return z_mus, guid, struct_names, struct_map_id


def main(config):
    set_determinism(config["random_state"])
    data_path = config["data_path"]
    os.makedirs(config["save_path"], exist_ok=True)

    # Load metadata
    index_ds = pd.read_csv(os.path.join(data_path,"dataset_index.csv"))
    clinical_ds = pd.read_csv(config["metadata_file"])
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner") # Merge on the 'GUID' column
    print(f"METADATA: Original rows: {len(metadata)}")
    metadata['subject'].replace('', pd.NA, inplace=True) # First, ensure empty strings are treated as NaN
    metadata = metadata.dropna(subset=['subject']) # Then drop rows where subject is NaN
    metadata = metadata.reset_index(drop=True) # Reset index
    print(f"METADATA: Remaining rows: {len(metadata)}") # Check result

    # Load labels and bounding boxes for cortical/subcortical structures
    labels_df = pd.read_csv(config["labels_path"])
    bb_df = pd.read_csv(config["bb_path"])
    labels_bb_df = pd.merge(labels_df, bb_df, on="LabelName", how="inner") # Merge on the 'GUID' column

    # Load VAE
    autoencoder = load_vae_model(config=config,
                                 device=config["device"],
                                 ckpt_path=config.get("ckpt_path"),
                                 use_old_state_dict=config.get("use_old_state_dict")
                                )
    autoencoder.eval()
    
     # Training configuration
    batch_files = sorted(metadata["batch_file"].unique())

    for batch_file in batch_files:
        output_filename = os.path.join(config["save_path"], batch_file.split(data_path)[-1])
        if os.path.isfile(output_filename):
            print(f"This file already exists: {output_filename}.")
            continue

        z_mu_batch = []
        guid_batch = []
        struct_struct_name_batch = []
        struct_map_id_batch = []

        dataset = SubCorBatDataset(metadata, batch_file, labels_bb_df, config["n_structs"])
        batch_iter = SequentialBatchIterator(dataset, batch_size=1, device=config["device"])
        progress_bar = tqdm(range(len(batch_iter)), total=len(batch_iter), ncols=150)
        progress_bar.set_description(f'File {os.path.basename(batch_file)}')

        for step in progress_bar:
            try:
                sample = next(batch_iter) # loop over several batches per file
            except (ValueError, RuntimeError) as e:
                print(f"⚠️ Skipping batch at step {step} due to error: {e}")
                continue

            z_mu, guid, struct_names, struct_map_id = process_batch(
                sample=sample,
                autoencoder=autoencoder,
                device=config["device"],
                batch_size=config["batch_size"]
                )
            z_mu_batch.append(z_mu)
            guid_batch.extend(guid)
            struct_struct_name_batch.extend(struct_names)
            struct_map_id_batch.extend(struct_map_id)

        z_mu_batch = np.concatenate(z_mu_batch, axis=0)

        # Save embeddings
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)  # <-- Ensure directory exists
        np.savez_compressed(output_filename, mus=z_mu_batch, GUID=guid_batch, struct_name=struct_struct_name_batch, MapID=struct_map_id_batch)
        print(f"Saved: {output_filename} with {len(z_mu_batch)} samples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .json")
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    # Set dynamic paths
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(config)


    
