import os
import argparse
import warnings

import pandas as pd
import torch
from tqdm import main, tqdm
from monai import transforms
from monai.utils import set_determinism

from torch.nn import L1Loss
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator

from model import GradientAccumulation, KLDivergenceLoss
from utils import load_config_from_path


from datetime import datetime
from torch.amp import autocast, GradScaler


from training import AverageLoss
from torch.utils.tensorboard import SummaryWriter
import random
from preprocessing import LookupNPZDataset, get_balanced_batch
import matplotlib.pyplot as plt
import warnings
from utils.visualization import plot_mri_comparison

def main(config):
    # Startup config
    seed = config["random_state"]
    set_determinism(seed)
    data_path = config["data_path"]
    resume_path= config["resume_path"]
    
    
    
    # 
    run_guid = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_path = os.path.join(config["base_logging_path"], run_guid) # Logging path preparation    

    # Load metadata
    index_ds = pd.read_csv(os.path.join(data_path,"dataset_index.csv"))
    clinical_ds = pd.read_csv(os.path.join(data_path,"combined_metadata.csv"))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner") # Merge on the 'GUID' column
    print(f"METADATA: Original rows: {len(metadata)}")

    # First, ensure empty strings are treated as NaN
    metadata['subject'].replace('', pd.NA, inplace=True)

    # Then drop rows where subject is NaN
    metadata = metadata.dropna(subset=['subject'])

    # Optional: reset index
    metadata = metadata.reset_index(drop=True)
    print(f"METADATA: Remaining rows: {len(metadata)}") # Check result


    # # Training config
    # num_workers = 0
    # num_epochs = 500
    # max_batch_size = 1
    # batch_size = 8
    # n_batches_per_file = 800
    # lr = 1e-4
    # adv_weight= 0.1
    # perceptual_weight = 0.1
    # kl_weight = 1e-7

    # VAE initialization
    vae_params = config["vae_params"]
    autoencoder = AutoencoderKL(**vae_params).to(config["device"])
    
    # Patch discrimiantor initialization
    dis_params = config["dis_params"]
    discriminator = PatchDiscriminator(**dis_params).to(config["device"])
    
    # Prepare losses
    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)

    loss_perceptual.to(config["device"])
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=config["lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config["lr"])
    
    avgloss = AverageLoss()
    
    gradacc_g = GradientAccumulation(actual_batch_size=config["max_batch_size"],
                                     expect_batch_size=config["batch_size"],
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=config["max_batch_size"],
                                     expect_batch_size=config["batch_size"],
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler())
    
    # Training configuration
    batch_files = sorted(metadata["batch_file"].unique())
    
    # Resume path
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        gradacc_g.grad_scaler.load_state_dict(checkpoint['scaler_g_state_dict'])
        gradacc_d.grad_scaler.load_state_dict(checkpoint['scaler_d_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        total_counter = checkpoint['total_counter']
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        total_counter = 0

    # New logging file
    writer = SummaryWriter(log_dir=config["logging_path"])
    for epoch in range(start_epoch, config["num_epochs"]):
        # random.shuffle(batch_files)
        for batch_file in batch_files:
            dataset = LookupNPZDataset(metadata, batch_file=batch_file, use_segmentation=False)
            total_batches = len(dataset) // config["max_batch_size"]
            progress_bar = tqdm(range(total_batches), total=total_batches, ncols=180)
            progress_bar.set_description(f'Epoch {epoch} - batch_file {batch_file}')

            step = 0
            for _ in progress_bar:  # loop over several batches per file
                with autocast(device_type=config["device"], enabled=True):
                    batch = get_balanced_batch(dataset, group_size=1, groups_per_batch=config["max_batch_size"], device=config["device"])
                    images = batch["feats"].to(config["device"])
                    
                    reconstruction, z_mu, z_sigma = autoencoder(images)

                    # we use [-1] here because the discriminator also returns 
                    # intermediate outputs and we want only the final one.
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                    # Computing the loss for the generator. In the Adverarial loss, 
                    # if the discriminator works well then the logits are close to 0.
                    # Since we use `target_is_real=True`, then the target tensor used
                    # for the MSE is a tensor of 1, and minizing this loss will make 
                    # the generator better at fooling the discriminator (the discriminator
                    # weights are not optimized here).

                    rec_loss = l1_loss_fn(reconstruction.float(), images.float())
                    kld_loss = config["kl_weight"] * kl_loss_fn(z_mu, z_sigma)
                    per_loss = config["perceptual_weight"] * loss_perceptual(reconstruction.float(), images.float())
                    gen_loss = config["adv_weight"] * adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    
                    loss_g = rec_loss + kld_loss + per_loss + gen_loss
                    
                gradacc_g.step(loss_g, step)

                with autocast(device_type=config["device"], enabled=True):

                    # Here we compute the loss for the discriminator. Keep in mind that
                    # the loss used is an MSE between the output logits and the expected logits.
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    d_loss_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    d_loss_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                    loss_d = config["adv_weight"] * discriminator_loss

                gradacc_d.step(loss_d, step)

                # Logging.
                avgloss.put(os.path.join(logging_path, 'Generator/reconstruction_loss'),    rec_loss.item())
                avgloss.put(os.path.join(logging_path, 'Generator/perceptual_loss'),        per_loss.item())
                avgloss.put(os.path.join(logging_path, 'Generator/adverarial_loss'),        gen_loss.item())
                avgloss.put(os.path.join(logging_path, 'Generator/kl_regularization'),      kld_loss.item())
                avgloss.put(os.path.join(logging_path, 'Discriminator/adverarial_loss'),    loss_d.item())

                
                if total_counter % 10 == 0:
                    global_step = total_counter // 10
                    avgloss.to_tensorboard(writer, global_step)
                    
                    # Only log images every 200 steps
                    if total_counter % 200 == 0:
                        gt_image = images[0].detach().cpu()
                        recon_image = reconstruction[0].detach().cpu()

                        fig = plot_mri_comparison(gt_image, recon_image, title="GT vs Reconstruction")
                        writer.add_figure("Comparison/GroundTruth_vs_Reconstruction", fig, global_step=global_step)
                        plt.close(fig)
                            
                step += 1
                total_counter += 1

             # Save the model
            checkpoint = {
                        'epoch': epoch,
                        'total_counter': total_counter,
                        'autoencoder_state_dict': autoencoder.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                    }

            torch.save(checkpoint, os.path.join(config["logging_path"], f'checkpoint-epoch-{epoch}.pth'))

            # Close previous writer
            writer.close()
            writer = SummaryWriter(log_dir=config["logging_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)
    
    
if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dev/configs/whole_brain/train_autoencoder_config.py", 
                        type=str, required=True, help='Path to the config .py file')
    args = parser.parse_args()
    
    # Load config
    config = load_config_from_path(args.config)

    # Set dynamic paths
    run_GUID = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    config["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(config)