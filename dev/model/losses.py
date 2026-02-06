import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

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

class MPRCLoss(nn.Module):
    """
    Multi-Positive Ranking Contrastive Loss (MPRCL)

    Implements:
    - Hard-positive attraction (L_hp)
    - Soft-negative ranking loss (L_sn)
    - Hard-negative ranking loss (L_hn)

    Aligns with NeuroCBIR manuscript notation.
    """

    def __init__(
        self,
        lambda_sn: float = 0.2,   # Weight for soft-negative ranking
        lambda_hn: float = 0.1,   # Weight for hard-negative ranking
        alpha_sn: float = 0.2,    # Soft-negative margin scaling
        alpha_hn: float = 0.2,    # Hard-negative margin scaling
        lower_perc: float = 0.05,  # Percentile threshold for hard negatives
        upper_perc: float = 0.95,  # Percentile threshold for soft negatives
    ):
        super().__init__()
        self.lambda_sn = lambda_sn
        self.lambda_hn = lambda_hn
        self.alpha_sn = alpha_sn
        self.alpha_hn = alpha_hn
        self.lower_perc = lower_perc
        self.upper_perc = upper_perc

        if upper_perc < lower_perc:
            raise ValueError("upper_perc must be >= lower_perc")

    def forward(self, feats: Tensor, vae_feats: Tensor, labels: Tensor) -> Tensor:
        """
        feats      : embedding vectors z_i (B x N_f)
        vae_feats  : VAE latent means mu_phi(x_i) (B x N_f)
        labels     : subject or subject-region identity
        """
        device = feats.device
        B = feats.size(0)

        # -----------------------------
        # Pairwise cosine similarity
        # -----------------------------
        feats = F.normalize(feats, dim=-1)
        sim = torch.matmul(feats, feats.T)

        # Hard positives mask: same label, exclude self
        same_label = labels[:, None] == labels[None, :]
        not_self = ~torch.eye(B, dtype=torch.bool, device=device)
        hard_pos_mask = same_label & not_self
        neg_mask = (~same_label) & not_self

        # -----------------------------
        # VAE-based similarity for soft/hard negative mining
        # -----------------------------
        with torch.no_grad():
            # Compute VAE-based similarity
            vae_feats = F.normalize(vae_feats, dim=-1)
            mu_sim = torch.matmul(vae_feats, vae_feats.T).detach().to(torch.float)

            # Detach similarity for margin scaling
            sim_detached = sim.detach().clone().to(torch.float)
            invalid = same_label | torch.eye(B, dtype=torch.bool, device=device)
            mu_sim[invalid] = mu_sim.mean()
            sim_detached[invalid] = sim_detached.mean()

            # Similarity range (valid only)
            sim_range = (sim_detached.max(dim=1).values - sim_detached.min(dim=1).values).clamp(min=1e-3)
            scaled_m_sn = self.alpha_sn * sim_range
            scaled_m_hn = self.alpha_hn * sim_range

            # Soft negatives: label-negative samples above upper percentile
            tau_sn = torch.nanquantile(mu_sim, self.upper_perc, dim=1)
            soft_mask = mu_sim >= tau_sn.unsqueeze(1)

            # Hard negatives: label-negative samples below lower percentile
            tau_hn = torch.nanquantile(mu_sim, self.lower_perc, dim=1)
            neg_mask &= mu_sim < tau_hn.unsqueeze(1)

            # Guard empty rows
            has_valid = ~torch.isnan(mu_sim).all(dim=1)
            soft_mask &= has_valid[:, None]
            neg_mask &= has_valid[:, None]

        # -----------------------------
        # Hard-positive attraction (L_hp)
        # -----------------------------
        pos_inf = torch.finfo(sim.dtype).max
        hard_sim = sim.masked_fill(~hard_pos_mask, pos_inf)
        s_hp_min = hard_sim.min(dim=1).values
        has_hp = hard_pos_mask.any(dim=1)

        L_hp = torch.zeros_like(s_hp_min)
        valid_hp = has_hp
        L_hp[valid_hp] += (1.0 - s_hp_min[valid_hp]).pow(2)

        # -----------------------------
        # Soft-negative ranking loss (L_sn)
        # -----------------------------
        soft_sim = sim.masked_fill(~soft_mask, -pos_inf)
        s_sn_max = soft_sim.max(dim=1).values
        has_sn = soft_mask.any(dim=1)

        L_sn = torch.zeros_like(s_hp_min)
        valid_sn = has_hp & has_sn
        L_sn[valid_sn] += F.relu(scaled_m_sn[valid_sn] + s_sn_max[valid_sn] - s_hp_min[valid_sn])

        # -----------------------------
        # Hard-negative ranking loss (L_hn)
        # -----------------------------
        neg_sim = sim.masked_fill(~neg_mask, -pos_inf)
        s_hn_max = neg_sim.max(dim=1).values
        has_hn = neg_mask.any(dim=1)

        soft_min = soft_sim.masked_fill(~soft_mask, pos_inf).min(dim=1).values
        L_hn = torch.zeros_like(s_hp_min)
        valid_hn = has_hp & has_sn & has_hn
        L_hn[valid_hn] += F.relu(scaled_m_hn[valid_hn] + s_hn_max[valid_hn] - soft_min[valid_hn])

        # -----------------------------
        # Final MPRCL
        # -----------------------------
        loss = L_hp.mean() + self.lambda_sn * L_sn.mean() + self.lambda_hn * L_hn.mean()
        return loss

