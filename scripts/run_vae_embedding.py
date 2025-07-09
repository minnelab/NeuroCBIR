import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.utils import set_determinism
from preprocessing.load_dataset import list_files_with_extension

def load_vae_model(config, device, ckpt_path = None, use_old_state_dict = False):
    vae = AutoencoderKL(
        spatial_dims=config["spatial_dims"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        channels=tuple(config["channels"]),
        latent_channels=config["latent_channels"],
        num_res_blocks=config["num_res_blocks"],
        norm_num_groups=config["norm_num_groups"],
        norm_eps=config["norm_eps"],
        attention_levels=tuple(config["attention_levels"]),
        with_encoder_nonlocal_attn=config["with_encoder_nonlocal_attn"],
        with_decoder_nonlocal_attn=config["with_decoder_nonlocal_attn"],
        include_fc=config["include_fc"]
    ).to(device)

    # Load weights (if checkpoint is provided)
    if ckpt_path:
        # Load full checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_key = config.get("ckpt_key", "autoencoder_state_dict")
        # Load only the autoencoder weights
        if use_old_state_dict:
            vae.load_old_state_dict(checkpoint[ckpt_key])
            print("Loaded weights using load_old_state_dict().")
        else:
            vae.load_state_dict(checkpoint[ckpt_key])
            print("Loaded weights using load_state_dict().")

    return vae

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
    os.makedirs(config["save_path"], exist_ok=True)

    # Load VAE
    autoencoder = load_vae_model(config=config["vae_config"],
                                 device=device,
                                 ckpt_path=config.get("ckpt_path"),
                                 use_old_state_dict=config.get("use_old_state_dict")
                                )
    autoencoder.eval()
    
     # List input files
    file_paths, file_names = list_files_with_extension(config["load_path"], config["extension"])
    file_names.sort()

    for file_path, file_name in zip(file_paths, file_names):
        file_to_load = os.path.join(config["load_path"], file_path, file_name)
        print(f"Processing {file_to_load}")
        data = np.load(file_to_load)

        z_mu_batch, ids_batch = process_batch(data=data,
                                              autoencoder=autoencoder,
                                              device=device,
                                             )

        # Save embeddings
        filename = os.path.join(config["save_path"], file_name)
        np.savez_compressed(filename, mus=z_mu_batch, GUID=ids_batch)
        print(f"Saved: {filename} with {len(z_mu_batch)} samples.")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    main(config)


    
