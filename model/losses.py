import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


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

class MultiPosConLoss(nn.Module):
    '''
    https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py
    '''
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        '''
        feats: shape: [B, D]
        labels: shape: [B]
        '''

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)

        # For single GPU: no all_gather
        all_feats = feats
        all_labels = labels

        # Compute mask: multiple positives = same label
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1), all_labels.view(1, -1)).float().to(feats.device)
            self.logits_mask = torch.ones_like(mask)
            self.logits_mask.fill_diagonal_(0)
            self.mask = mask * self.logits_mask
            self.last_local_batch_size = local_batch_size

        mask = self.mask
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # Stability
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        # Soft targets
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        q = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(p * q, dim=-1).mean()
        return loss
