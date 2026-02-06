import matplotlib.pyplot as plt
import torch

def plot_mri_comparison(gt_tensor, recon_tensor, title="MRI Comparison"):
    """
    Plot 3 views (axial, coronal, sagittal) for both ground truth and reconstruction
    gt_tensor, recon_tensor: [C, D, H, W] or [D, H, W]
    """
    # If tensors have channel dimension, take first channel
    if gt_tensor.ndim == 4:
        gt_tensor = gt_tensor[0]
    if recon_tensor.ndim == 4:
        recon_tensor = recon_tensor[0]

    # Convert to float32 if needed
    if isinstance(gt_tensor, torch.Tensor):
        gt_tensor = gt_tensor.float()
    if isinstance(recon_tensor, torch.Tensor):
        recon_tensor = recon_tensor.float()

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
