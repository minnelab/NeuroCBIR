import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_config_from_path
import os
from data.multi_comp.cnn_pretrained_model.AD_pretrained_utilities import CNN_8CL_B, CNN

# Load config
config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))
config.update({"batch_size": 16})

# Load pretrained CNN
base_model = CNN(CNN_8CL_B(), return_conv=True)
w = torch.load(os.path.join(config["output_dir"], "AD_pretrained_weights.pt"), map_location="cpu")
base_model.load_state_dict(w)

# Wrapper model with resampling at the beginning
class ResampleWrapper(nn.Module):
    def __init__(self, model, target_shape=(73, 96, 96)):
        super().__init__()
        self.model = model
        self.target_shape = target_shape

    def forward(self, x):
        # Ensure input is float and shape [B, 1, D, H, W]
        x = x.float()
        x = F.interpolate(
            x, size=self.target_shape, mode="trilinear", align_corners=False
        )
        return self.model(x)

# Build wrapped model
model = ResampleWrapper(base_model)

# Add to config
config["pretrained_encoder"] = model
