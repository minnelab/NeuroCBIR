import torch
import torch.nn as nn
from monai.networks.nets.autoencoderkl import AutoencoderKL, Encoder
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

class ContrastiveModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_shape: Sequence[int],
        projector_dims: Sequence[int] = [128, 64, 32],
        include_ln: bool = True,
        final_dim: int = 32,
        device: str = "cpu"
    ):
        """
        Contrastive learning model that wraps an encoder and a projection MLP.

        Args:
            encoder: CNN encoder module
            input_shape: shape of input tensor excluding batch dim, e.g. [1, 20, 22, 26]
            projector_dims: list of layer dimensions for the FC projector
            include_bn: whether to include BatchNorm1d after FC layers
            final_dim: final projection dimension
            device: device for shape inference
        """
        super().__init__()
        self.encoder = encoder.to(device)

        # Global Average Pooling for 
        self.pool = nn.AdaptiveAvgPool3d(1)

        # Infer flattened encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).to(device)
            dummy_output = self.encoder(dummy_input)
            flat_dim = dummy_output.shape[1]

        # Build projection head
        layers = []
        input_dim = flat_dim
        for dim in projector_dims:
            layers.append(nn.Linear(input_dim, dim))
            if include_ln:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim

        if input_dim != final_dim:
            layers.append(nn.Linear(input_dim, final_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x, return_encoder_output=False):
        x = self.encoder(x)
        x_flat = torch.flatten(self.pool(x), start_dim=1)
        z = self.projector(x_flat)
        return (x, z) if return_encoder_output else z

class Q2EModel(nn.Module):
    def __init__(self, vae_encoder, cl_encoder):
        super().__init__()
        self.vae_encoder = vae_encoder
        self.cl_encoder = cl_encoder

    def forward(self, x):
        with torch.no_grad():
            z_mu, z_logvar = self.vae_encoder(x)
            out = self.cl_encoder(z_mu)
        return out.view(out.size(0), -1)  # Flatten except batch dimension

def load_vae_encoder(config, device):
    # Set up VAE
    vae_params = config["vae_params"]
    autoencoder = AutoencoderKL(**vae_params).to(device)

    # Load weights
    checkpoint = torch.load(config["vae_ckpt_path"], map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    logger.info("Loaded weights of VAE.")
    return autoencoder.encode

def load_cl_projector(config, device):
    def create_encoder(config, device):
        encoder_params = cl_params["encoder_params"]
        return Encoder(**encoder_params).to(device)
    cl_params = config["cl_params"]
    encoder = create_encoder(cl_params, device)
    model = ContrastiveModel(
        encoder=encoder,
        input_shape=cl_params["proj_params"]["input_shape"],
        projector_dims=cl_params["proj_params"]["projector_dims"],
        final_dim=cl_params["proj_params"]["final_dim"],
        device=device
    ).to(device)
    checkpoint = torch.load(config["cl_ckpt_path"], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Loaded weights of CL.")
    return model

def build_Q2E(config, device):
    vae_encoder = load_vae_encoder(config, device)   # returns autoencoder.encode
    cl_encoder = load_cl_projector(config, device)   # returns ContrastiveModel

    model = Q2EModel(vae_encoder, cl_encoder).to(device)
    model.eval()   # if you only need inference
    logger.info("Loaded Q2E module.")
    return model