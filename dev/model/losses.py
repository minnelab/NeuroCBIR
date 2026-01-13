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
    """
    Multi-Positive Contrastive Loss with:
      - hard positives (same identity)
      - soft positives (high similarity but different identity)
      - hard negatives (low similarity)
    Percentile-based selection is performed per anchor.
    """

    def __init__(
        self,
        lambda_id: float = 0.2,
        lambda_soft: float = 0.1,
        m_id: float = 0.2,
        m_soft: float = 0.2,
        lower_perc: float = 0.8,
        upper_perc: float = 0.8,
    ):
        """
        Parameters
        ----------
        lambda_id : float
            Weight of the identity-based ranking loss.

        lambda_soft : float
            Weight of the soft-positive vs negative ranking loss.

        m_id : float
            Margin for identity-based ranking constraints
            (hard positives vs soft positives / negatives).

        m_soft : float
            Margin enforcing soft positives to be closer than negatives.

        lower_perc : float
            Percentile threshold (per anchor) for selecting hard negatives.
            Values below this percentile are treated as negatives.

        upper_perc : float
            Percentile threshold (per anchor) for selecting soft positives.
            Values above this percentile are treated as soft positives.
        """
        super().__init__()

        # loss weighting
        self.lambda_id = lambda_id
        self.lambda_soft = lambda_soft

        # margins (relative, later scaled by per-anchor similarity range)
        self.m_id = m_id
        self.m_soft = m_soft

        # percentile thresholds for selection
        self.upper_perc = upper_perc
        self.lower_perc = lower_perc
        # Basic sanity check (don't silently allow inverted percentiles)
        if (self.upper_perc is not None) and (self.lower_perc is not None):
            if self.upper_perc < self.lower_perc:
                raise ValueError("upper_perc must be >= lower_perc")

    def forward(self, feats, vae_feats, labels):
        device = feats.device
        B = feats.size(0)

        # -----------------------------
        # Similarity matrix
        # -----------------------------
        feats = F.normalize(feats, dim=-1)
        sim = torch.matmul(feats, feats.T)

        same_label = labels[:, None] == labels[None, :]
        not_self = ~torch.eye(B, dtype=torch.bool, device=device)

        hard_pos_mask = same_label & not_self
        neg_mask = (~same_label) & not_self

        # -----------------------------
        # Soft positives and hard negatives
        # -----------------------------
        with torch.no_grad():
            
            # Compute VAE-based similarity
            vae_feats = F.normalize(vae_feats, dim=-1)
            mu_sim = torch.matmul(vae_feats, vae_feats.T).detach().to(torch.float)

            # Detach similarity for margin scaling
            sim_detached = sim.detach().clone().to(torch.float)

            # invalidate hard positives and self-similarities
            invalid = same_label | torch.eye(B, dtype=torch.bool, device=device)
            
            # Set invalid similarities to NaN for percentile computations
            # mu_sim[invalid] = float("nan")
            mu_sim[invalid] = torch.mean(mu_sim)
            sim_detached[invalid] = torch.mean(sim_detached)

            # similarity range (valid only)
            sim_range = (sim_detached.max(dim=1).values - sim_detached.min(dim=1).values).clamp(min=1e-3)

            scaled_m_id   = self.m_id * sim_range
            scaled_m_soft = self.m_soft * sim_range

            # soft positives
            tau_soft = torch.nanquantile(mu_sim, self.upper_perc, dim=1)
            soft_mask = mu_sim >= tau_soft.unsqueeze(1)

            # hard negatives
            if self.lower_perc is not None:
                tau_neg = torch.nanquantile(mu_sim, self.lower_perc, dim=1)
                neg_mask &= mu_sim < tau_neg.unsqueeze(1)

            # guard empty rows
            has_valid = ~torch.isnan(mu_sim).all(dim=1)
            soft_mask &= has_valid[:, None]
            neg_mask  &= has_valid[:, None]
            
        # -----------------------------
        # Identity ranking loss
        # -----------------------------
        pos_inf = torch.finfo(sim.dtype).max
        neg_inf = torch.finfo(sim.dtype).min

        # hard positives min
        hard_sim = sim.masked_fill(~hard_pos_mask, pos_inf)
        hp_min = hard_sim.min(dim=1).values # hard_sim.mean(dim=1) # hard_sim.min(dim=1).values

        # soft positives max
        soft_sim = sim.masked_fill(~soft_mask, neg_inf)
        soft_max = soft_sim.max(dim=1).values

        # negatives max
        neg_sim = sim.masked_fill(~neg_mask, neg_inf)
        neg_max = neg_sim.max(dim=1).values

        # valid masks
        has_hard = hard_pos_mask.any(dim=1)
        has_soft = soft_mask.any(dim=1)
        has_neg  = neg_mask.any(dim=1)

        loss_hd = torch.zeros_like(hp_min)
            
        # hard-positive attraction (inside loss_id)
        valid_hp = has_hard
        loss_hd[valid_hp] += (1.0 - hp_min[valid_hp]).pow(2)
        # # soft-positive attraction (inside loss_id)
        # loss_id[valid_soft] += F.relu(
        #     hp_min[valid_soft] - soft_max[valid_soft]
        # )

        # -----------------------------
        # Identity ranking loss
        # -----------------------------
        
        loss_id = torch.zeros_like(hp_min)
        
        # hard > soft
        valid_soft = has_hard & has_soft
        loss_id[valid_soft] += F.relu(
            scaled_m_id[valid_soft] + soft_max[valid_soft] - hp_min[valid_soft]
        )

        # hard > negative
        valid_neg = has_hard & has_neg
        loss_id[valid_neg] += F.relu(
            scaled_m_id[valid_neg] + neg_max[valid_neg] - hp_min[valid_neg]
        )

        # -----------------------------
        # Soft > negative loss
        # -----------------------------
        soft_min = sim.masked_fill(~soft_mask, pos_inf).min(dim=1).values
        loss_soft = F.relu(scaled_m_soft + neg_max - soft_min)
        loss_soft = torch.where(has_soft & has_neg, loss_soft, 0)

        # -----------------------------
        # Final loss
        # -----------------------------
        loss = loss_hd.mean() + self.lambda_id * loss_id.mean() + self.lambda_soft * loss_soft.mean()
        return loss

