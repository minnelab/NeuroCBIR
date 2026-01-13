import os
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.utils import set_determinism
from dev.utils import load_config_from_path
import logging

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
            logging.info("Loaded weights using load_old_state_dict().")
        else:
            autoencoder.load_state_dict(checkpoint[ckpt_key])
            logging.info("Loaded weights using load_state_dict().")

    return autoencoder


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
            logging.info("Loaded weights using load_old_state_dict().")
        else:
            autoencoder.load_state_dict(checkpoint[ckpt_key])
            logging.info("Loaded weights using load_state_dict().")

    return autoencoder

def process_batch(data, autoencoder, device):
    z_mus = []
    ids = []

    images = data['images']
    sample_ids = data['GUID']

    for img, sample_id in tqdm(zip(images, sample_ids), total=len(images), ncols=150, desc="Processing batch"):
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            z_mu, _ = autoencoder.encode(img)

        z_mu = z_mu.squeeze(0).cpu().numpy().astype(np.float16)
        z_mus.append(z_mu)
        ids.append(sample_id)

    return np.stack(z_mus), np.stack(ids)

def main(config):
    set_determinism(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = config["data_path"]
    os.makedirs(config["save_path"], exist_ok=True)

    # Load metadata
    index_ds = pd.read_csv(os.path.join(data_path,"dataset_index.csv"))
    clinical_ds = pd.read_csv(config["metadata_file"])
    
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner") # Merge on the 'GUID' column
    logging.info(f"METADATA: Original rows: {len(metadata)}")
    metadata['subject'].replace('', pd.NA, inplace=True) # First, ensure empty strings are treated as NaN
    metadata = metadata.dropna(subset=['subject']) # Then drop rows where subject is NaN
    metadata = metadata.reset_index(drop=True) # Reset index
    logging.info(f"METADATA: Remaining rows: {len(metadata)}") # Check result

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
            logging.info(f"This file already exists: {output_filename}.")
            continue

        # file_to_load = os.path.join(config["load_path"], file_path, file_name)
        logging.info(f"Processing {batch_file}")
        data = np.load(batch_file, allow_pickle=True)

        z_mu_batch, ids_batch = process_batch(data=data,
                                              autoencoder=autoencoder,
                                              device=device,
                                             )

        # Save embeddings
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        np.savez_compressed(output_filename, mus=z_mu_batch, GUID=ids_batch)
        logging.info(f"Saved: {output_filename} with {len(z_mu_batch)} samples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .py")
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    # Set dynamic paths
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(config)



    
