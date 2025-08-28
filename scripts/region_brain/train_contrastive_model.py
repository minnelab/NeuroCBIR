import os
import json
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets.autoencoderkl import Encoder
from monai.utils import set_determinism

from model.contrastive_model import ContrastiveModel
from model.losses import MultiPosConLoss
from training import AverageLoss
from preprocessing import RegionEmbBatchedDataset, get_balanced_batch
from utils import load_config_from_path
import random
import time


def create_encoder(config):
    encoder_params = config["encoder_params"]
    return Encoder(**encoder_params).to(config["device"])

def main(config):
    set_determinism(0)
    device = config["device"]
    os.makedirs(config["logging_path"], exist_ok=True)
    with open(os.path.join(config["logging_path"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

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

    # Model setup
    encoder = create_encoder(config)
    model = ContrastiveModel(
        encoder=encoder,
        input_shape=config["proj_params"]["input_shape"],
        projector_dims=config["proj_params"]["projector_dims"],
        final_dim=config["proj_params"]["final_dim"],
        device=device
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    grad_scaler = GradScaler()
    avgloss = AverageLoss()
    cont_loss_fn = MultiPosConLoss()

    # Resume trainig
    resume_path = config["resume_path"]
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        total_counter = checkpoint['total_counter']
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        total_counter = 0

    # Load dataset
    batch_files = sorted(metadata["batch_file"].unique())
    dataset = RegionEmbBatchedDataset(metadata, batch_files=batch_files)

    # Training
    writer = SummaryWriter(log_dir=config["logging_path"])
    for epoch in range(start_epoch, config["num_epochs"]):
        # random.shuffle(batch_files)

        # for batch_file in batch_files:
        # for i_batch in range(0, len(batch_files) - config["n_file_loaded"]):
        # dataset = RegionEmbBatchedDataset(metadata, batch_files=batch_files[i_batch:i_batch+config["n_file_loaded"]])
        progress_bar = tqdm(range(config["n_batches_per_file"]), total=config["n_batches_per_file"], ncols=150)
        progress_bar.set_description(f'Epoch {epoch} - N_batches {len(batch_files)}')

        for step in progress_bar:
            with autocast(device_type=device, enabled=True):
                batch = get_balanced_batch(
                    dataset,
                    batch_size=config["batch_size"],
                    group_size=config["group_size"],
                    groups_per_batch=config["groups_per_batch"],
                    group_key=config["group_key"],
                    device=device
                )
                embs, labels = batch["emb"], batch["label"]
                proj_embs = model(embs)
                cont_loss = cont_loss_fn(proj_embs, labels)

            loss = cont_loss
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)

            avgloss.put(os.path.join(config["logging_path"], 'cont_loss'), cont_loss.item())

            if total_counter % 10 == 0:
                global_step = total_counter // 10
                avgloss.to_tensorboard(writer, global_step)

            total_counter += 1

        
        if (epoch % 5 == 0): # and (epoch > 0):
            checkpoint = {
                'epoch': epoch,
                'total_counter': total_counter,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': grad_scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(config["logging_path"], f'checkpoint.pth'))

            writer.close()
            writer = SummaryWriter(log_dir=config["logging_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON configuration file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    # Set dynamic paths
    run_GUID = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(config)
