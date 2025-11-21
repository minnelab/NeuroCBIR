import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from monai.networks.nets.autoencoderkl import Encoder
from monai.utils import set_determinism

from dev.model.contrastive_model import ContrastiveModel
from dev.utils import load_config_from_path
from dev.preprocessing import EmbBatchedDataset, SequentialBatchIterator

def create_encoder(config, device):
    encoder_params = config["encoder_params"]
    return Encoder(**encoder_params).to(device)

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

    # Model setup
    encoder = create_encoder(config, device)
    model = ContrastiveModel(
        encoder=encoder,
        input_shape=config["proj_params"]["input_shape"],
        projector_dims=config["proj_params"]["projector_dims"],
        final_dim=config["proj_params"]["final_dim"],
        device=device
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(config["resume_path"], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Get unique .npz files to loop over
    batch_files = sorted(metadata["batch_file"].unique())

    # Evaluate and embed all batches
    all_proj_embs = []
    all_guids = []

    for batch_file in batch_files:
        dataset = EmbBatchedDataset(metadata, batch_files=batch_file)
        batch_iter = SequentialBatchIterator(dataset, batch_size=config["batch_size"], device=device)

        progress_bar = tqdm(batch_iter, total=len(batch_iter), ncols=150)
        progress_bar.set_description(f'Embedding {os.path.basename(batch_file)}')

        for batch in progress_bar:
            embs, guids = batch["emb"], batch["guid"]
            with torch.no_grad():
                proj_embs = model(embs)
            all_proj_embs.append(proj_embs.cpu())
            all_guids.extend(guids)  # list of strings

    # Stack everything into a DataFrame and save
    embs_tensor = torch.cat(all_proj_embs, dim=0)  # shape: [N, D]
    df_embs = pd.DataFrame(embs_tensor.numpy())
    df_embs.insert(0, "GUID", all_guids)

    df_embs.columns = df_embs.columns.astype(str)
    output_path = os.path.join(config["output_dir"], "projected_embeddings.parquet")
    df_embs.to_parquet(output_path, index=False)
    print(f"✅ Embeddings saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)
