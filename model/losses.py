import torch
import torch.nn as nn
from torch import Tensor


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        """
        input: predicted output from autoencoder, shape (B, C, D, H, W)
        target: ground truth MRI images, shape (B, C, D, H, W)
        weight: per-voxel weights, shape (B, C, D, H, W) or broadcastable to input
        """
        if weight is None:
            weight = torch.ones_like(input)

        loss = weight * (input - target) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction


class KLDivergenceLoss:
    """
    A class for computing the Kullback-Leibler divergence loss.
    """
    
    def __call__(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        """
        Computes the KL divergence loss for the given parameters.

        Args:
            z_mu (Tensor):  The mean of the distribution.
            z_sigma (Tensor): The standard deviation of the distribution.

        Returns:
            Tensor: The computed KL divergence loss, averaged over the batch size.
        """

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]