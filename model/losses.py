import torch
import torch.nn as nn

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