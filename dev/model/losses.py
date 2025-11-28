import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import logging

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
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, feats, labels):
        """
        Args:
            feats: [B, D] - Embeddings
            labels: [B] - Categorical labels (could repeat or be unique)
        Returns:
            Scalar loss
        """
        # device = feats.device
        # feats = F.normalize(feats, dim=-1)
        # logits = torch.matmul(feats, feats.T) / self.temperature  # [B, B]

        # # Build positive mask (1 if same label, 0 otherwise)
        # label_mask = labels.view(-1, 1) == labels.view(1, -1)  # [B, B]

        # pos_mask = label_mask.float()
        # # neg_mask = (~label_mask).float()

        # # Count positives per sample
        # pos_counts = pos_mask.sum(dim=1, keepdim=True)  # [B, 1]

        # # Avoid divide-by-zero and use fallback
        # has_pos = (pos_counts > 0).float()
        # no_pos = 1.0 - has_pos

        # # Soft targets for samples with positives
        # pos_probs = pos_mask / pos_counts.clamp(min=1.0)

        # # Fallback: NT-Xent (sample treated as having self as only positive)
        # fallback_probs = F.one_hot(torch.arange(len(labels), device=device), num_classes=len(labels)).float()

        # # Final target distribution
        # target_probs = has_pos * pos_probs + no_pos * fallback_probs

        # log_probs = F.log_softmax(logits, dim=1)
        # loss = -torch.sum(target_probs * log_probs, dim=1).mean()

        # # This is to compute the minimum loss value possible in each batch (depends on the labels).
        # # min_loss = torch.sum(target_probs * -torch.log(target_probs.clamp(min=1e-9)), dim=1).mean()

        # return loss


        device = feats.device
        feats = F.normalize(feats, dim=-1)
        logits = torch.matmul(feats, feats.T) / self.temperature  # [B, B]

        # Exclude self-similarity from positive mask
        same_label = labels.view(-1, 1) == labels.view(1, -1)      # [B, B]
        not_self = ~torch.eye(len(labels), dtype=torch.bool, device=device)
        pos_mask = (same_label & not_self).float()
        neg_mask = (~same_label).float()

        # Count actual positives (excluding self)
        pos_counts = pos_mask.sum(dim=1, keepdim=True)  # [B, 1]

        has_pos = (pos_counts > 0).float()
        no_pos = 1.0 - has_pos

        # Normalize across positives
        pos_probs = pos_mask / pos_counts.clamp(min=1.0)

        # Fallback if no other positives exist
        fallback_probs = F.one_hot(torch.arange(len(labels), device=device), num_classes=len(labels)).float()

        # Final target
        target_probs = has_pos * pos_probs + no_pos * fallback_probs

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(target_probs * log_probs, dim=1).mean()
        
        logging.debug(f"Min loss: {torch.sum(target_probs * -torch.log(target_probs.clamp(min=1e-9)), dim=1).mean().item()}")
        return loss


