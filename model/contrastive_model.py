import torch
import torch.nn as nn
from typing import Sequence

class ContrastiveModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_shape: Sequence[int],
        projector_dims: Sequence[int] = [128, 64, 32],
        include_bn: bool = True,
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

        # Infer flattened encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).to(device)
            dummy_output = self.encoder(dummy_input)
            flat_dim = dummy_output.view(1, -1).shape[1]

        # Build projection head
        layers = []
        input_dim = flat_dim
        for dim in projector_dims:
            layers.append(nn.Linear(input_dim, dim))
            if include_bn:
                layers.append(nn.LayerNorm(dim))  # ← Replace BatchNorm1d
            layers.append(nn.ReLU(inplace=True))
            input_dim = dim


        if input_dim != final_dim:
            layers.append(nn.Linear(input_dim, final_dim))

        self.projector = nn.Sequential(*layers)

    def forward(self, x, return_encoder_output=False):
        x = self.encoder(x)
        x_flat = torch.flatten(x, 1)
        z = self.projector(x_flat)
        return (x, z) if return_encoder_output else z
