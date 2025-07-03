import os
import argparse
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from monai import transforms
from monai.utils import set_determinism

from torch.nn import L1Loss
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator

from model import GradientAccumulation, KLDivergenceLoss
from datetime import datetime
from torch.amp import autocast, GradScaler


from training import AverageLoss
from torch.utils.tensorboard import SummaryWriter
import random
from preprocessing import LookupNPZDataset, get_balanced_batch
import matplotlib.pyplot as plt
import warnings


def plot_mri_comparison(gt_tensor, recon_tensor, title="MRI Comparison"):
    """
    Plot 3 views (axial, coronal, sagittal) for both ground truth and reconstruction
    gt_tensor, recon_tensor: [C, D, H, W] or [D, H, W]
    """
    if gt_tensor.ndim == 4:
        gt_tensor = gt_tensor[0]
    if recon_tensor.ndim == 4:
        recon_tensor = recon_tensor[0]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(title)

    mid_slices = [
        gt_tensor.shape[0] // 2,
        gt_tensor.shape[1] // 2,
        gt_tensor.shape[2] // 2
    ]

    # Row 0: Ground Truth
    axes[0, 0].imshow(gt_tensor[mid_slices[0], :, :], cmap="gray")
    axes[0, 1].imshow(gt_tensor[:, mid_slices[1], :], cmap="gray")
    axes[0, 2].imshow(gt_tensor[:, :, mid_slices[2]], cmap="gray")
    for ax in axes[0]:
        ax.axis("off")
    axes[0, 0].set_title("Axial")
    axes[0, 1].set_title("Coronal")
    axes[0, 2].set_title("Sagittal")

    # Row 1: Reconstruction
    axes[1, 0].imshow(recon_tensor[mid_slices[0], :, :], cmap="gray")
    axes[1, 1].imshow(recon_tensor[:, mid_slices[1], :], cmap="gray")
    axes[1, 2].imshow(recon_tensor[:, :, mid_slices[2]], cmap="gray")
    for ax in axes[1]:
        ax.axis("off")

    axes[0, 0].set_ylabel("Ground Truth", rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("Reconstruction", rotation=90, labelpad=10)

    fig.tight_layout()
    return fig


set_determinism(0)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
RUN_GUID = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_PATH = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/"
# RESUME_PATH = "/cephyr/users/felixnie/Alvis/logs/20250701_125311/checkpoint-epoch-1.pth" # Fill in for resuming training
RESUME_PATH=""
LOGGING_PATH = os.path.join("/cephyr/users/felixnie/Alvis/", "logs", RUN_GUID) # Logging path preparation

if __name__ == '__main__':

    ### Input data
    # Path to dataset
    load_ds_path = DATA_PATH + "batched_adni/"
    # Files to load/save extension
    extension = ".npz"
    # Pretrained weights for the VAE
    ckpt_vae_path = ""#"./data/pretrained_models/autoencoder_puglisi.pth"
    # Pretrained weights for the Discriminator
    ckpt_dis_path = ""#"./data/pretrained_models/ckpt_dis_.pth"
    # Preparing image for using as input of the VAE
    target_shape = [1, 160, 224, 160] # Desired shape: [1, 160, 224, 160]
    


    # Load metadata
    index_ds = pd.read_csv(os.path.join(DATA_PATH,"dataset_index.csv"))
    clinical_ds = pd.read_csv(os.path.join(DATA_PATH,"combined_metadata.csv"))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner") # Merge on the 'GUID' column
    print(f"METADATA: Original rows: {len(metadata)}")

    # First, ensure empty strings are treated as NaN
    metadata['subject'].replace('', pd.NA, inplace=True)

    # Then drop rows where subject is NaN
    metadata = metadata.dropna(subset=['subject'])

    # Optional: reset index
    metadata = metadata.reset_index(drop=True)
    print(f"METADATA: Remaining rows: {len(metadata)}") # Check result


    # Training config
    num_workers = 0
    num_epochs = 500
    max_batch_size = 1
    batch_size = 8
    n_batches_per_file = 800
    lr = 1e-4
    adv_weight= 0.1
    perceptual_weight = 0.1
    kl_weight = 1e-7

    autoencoder = AutoencoderKL(spatial_dims=3,
                                in_channels=1,
                                out_channels=1,
                                latent_channels=8,
                                channels=(64, 128, 128, 128),
                                num_res_blocks=2,
                                norm_num_groups=32,
                                norm_eps=1e-06,
                                attention_levels=(False, False, False, False),
                                with_decoder_nonlocal_attn=False,
                                with_encoder_nonlocal_attn=False)
    autoencoder.to(DEVICE)

    # Load pretrained model VAE
    if os.path.isfile(ckpt_vae_path):
        state_vae_dict = torch.load(ckpt_vae_path)
        autoencoder.load_old_state_dict(state_vae_dict)
        print("VAE weights successfully loaded!")
    else:
        print("VAE weights are not available!")

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, channels=32, in_channels=1, out_channels=1, norm="INSTANCE")
    discriminator.to(DEVICE)

    # Load pretrained model VAE
    if os.path.isfile(ckpt_dis_path):
        state_dis_dict = torch.load(ckpt_dis_path)
        discriminator.load_old_state_dict(state_dis_dict)
        print("Discriminator weights successfully loaded!")
    else:
        print("Discriminator weights are not available!")

    l1_loss_fn = L1Loss()
    kl_loss_fn = KLDivergenceLoss()

    # Prepare losses
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)

    loss_perceptual.to(DEVICE)
    
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)


    gradacc_g = GradientAccumulation(actual_batch_size=max_batch_size,
                                     expect_batch_size=batch_size,
                                     loader_len=n_batches_per_file,
                                     optimizer=optimizer_g, 
                                     grad_scaler=GradScaler())

    gradacc_d = GradientAccumulation(actual_batch_size=max_batch_size,
                                     expect_batch_size=batch_size,
                                     loader_len=n_batches_per_file,
                                     optimizer=optimizer_d, 
                                     grad_scaler=GradScaler())

    avgloss = AverageLoss()

    # Training configuration
    batch_files = sorted(metadata["batch_file"].unique())

    # Resume path
    if os.path.isfile(RESUME_PATH):
        checkpoint = torch.load(RESUME_PATH)
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

    for epoch in range(start_epoch, num_epochs):
        # random.shuffle(batch_files)

        # New logging file
        writer = SummaryWriter(log_dir=LOGGING_PATH)

        for batch_file in batch_files:
            dataset = LookupNPZDataset(metadata, batch_file=batch_file, use_segmentation=False)
            progress_bar = tqdm(range(n_batches_per_file), total=n_batches_per_file, ncols=180)
            progress_bar.set_description(f'Epoch {epoch} - batch_file {batch_file}')

            for step in progress_bar:  # loop over several batches per file
                with autocast(device_type=DEVICE, enabled=True):
                    batch = get_balanced_batch(dataset, group_size=1, groups_per_batch=max_batch_size, device=DEVICE)
                    images = batch["feats"].to(DEVICE)
                    
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
                    kld_loss = kl_weight * kl_loss_fn(z_mu, z_sigma)
                    per_loss = perceptual_weight * loss_perceptual(reconstruction.float(), images.float())
                    gen_loss = adv_weight * adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    
                    loss_g = rec_loss + kld_loss + per_loss + gen_loss
                    
                gradacc_g.step(loss_g, step)

                with autocast(device_type=DEVICE, enabled=True):

                    # Here we compute the loss for the discriminator. Keep in mind that
                    # the loss used is an MSE between the output logits and the expected logits.
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    d_loss_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    d_loss_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (d_loss_fake + d_loss_real) * 0.5
                    loss_d = adv_weight * discriminator_loss

                gradacc_d.step(loss_d, step)

                # Logging.
                avgloss.put(os.path.join(LOGGING_PATH, 'Generator/reconstruction_loss'),    rec_loss.item())
                avgloss.put(os.path.join(LOGGING_PATH, 'Generator/perceptual_loss'),        per_loss.item())
                avgloss.put(os.path.join(LOGGING_PATH, 'Generator/adverarial_loss'),        gen_loss.item())
                avgloss.put(os.path.join(LOGGING_PATH, 'Generator/kl_regularization'),      kld_loss.item())
                avgloss.put(os.path.join(LOGGING_PATH, 'Discriminator/adverarial_loss'),    loss_d.item())

                
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
                            
                total_counter += 1

            # Save the model after each epoch.
            checkpoint = {
                        'epoch': epoch,
                        'total_counter': total_counter,
                        'autoencoder_state_dict': autoencoder.state_dict(),
                        'discriminator_state_dict': discriminator.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'scaler_g_state_dict': gradacc_g.grad_scaler.state_dict(),
                        'scaler_d_state_dict': gradacc_d.grad_scaler.state_dict()
                    }

            torch.save(checkpoint, os.path.join(LOGGING_PATH, f'checkpoint-epoch-{epoch}.pth'))

            # Close previous writer
            writer.close()
