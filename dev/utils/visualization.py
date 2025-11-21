import matplotlib.pyplot as plt
import torch

# def plot_slices_comparison(original, reconstructed, title_prefix=""):
#     """
#     Plots mid-slices (Axial, Coronal, Sagittal) of original and reconstructed volumes side-by-side.

#     Args:
#         original (ndarray): Original 3D volume (H x W x D).
#         reconstructed (ndarray): Reconstructed 3D volume (H x W x D).
#         title_prefix (str): Prefix for plot titles.
#     """
#     assert original.shape == reconstructed.shape, "Shapes must match!"

#     pz = original.shape[0] // 2
#     py = original.shape[1] // 2
#     px = original.shape[2] // 2

#     slices = {
#         "Axial (Z)":      (original[pz, :, :], reconstructed[pz, :, :]),
#         "Coronal (Y)":    (original[:, py, :], reconstructed[:, py, :]),
#         "Sagittal (X)":   (original[:, :, px], reconstructed[:, :, px]),
#     }

#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     for i, (title, (orig, recon)) in enumerate(slices.items()):
#         axes[0, i].imshow(orig, cmap='gray')
#         axes[0, i].set_title(f'{title_prefix}Original - {title}')
#         axes[0, i].axis('off')

#         axes[1, i].imshow(recon, cmap='gray')
#         axes[1, i].set_title(f'{title_prefix}Reconstructed - {title}')
#         axes[1, i].axis('off')

#     plt.tight_layout()
#     plt.show()

# def visualize_autoencoder_reconstruction(model, sample, device, substructure_index=0, title_prefix=""):
#     """
#     Generates and plots original vs. reconstructed images using a trained autoencoder.

#     Args:
#         model (nn.Module): Trained autoencoder model.
#         sample (dict): A sample from the dataset with keys "input" and optionally others.
#         device (torch.device): CUDA or CPU device.
#         substructure_index (int): Index to select sub-volume (e.g. for 4D input).
#         title_prefix (str): Title prefix for plots.
#     """
#     model.eval()

#     with torch.no_grad():
#         input_img = sample["input"].unsqueeze(0).to(device, dtype=torch.float32)[:, substructure_index:substructure_index+1]
#         _, reconstructed = model(input_img)

#     # Convert to NumPy
#     original_np = input_img.squeeze().cpu().numpy()
#     reconstructed_np = reconstructed.squeeze().cpu().numpy()

#     # Plot comparison
#     plot_slices_comparison(original_np, reconstructed_np, title_prefix=title_prefix)

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

