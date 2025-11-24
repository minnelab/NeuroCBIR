from monai.networks.nets import resnet50
from dev.utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
        "batch_size": 4,

})

encoder = resnet50(n_input_channels=1, 
                     feed_forward=False,
                     shortcut_type="B",
                     bias_downsample=False,                
                     pretrained = True, 
                     progress = True)

# Wrapper model with resampling at the beginning
import torch
import torch.nn as nn
import torch.nn.functional as F

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

img_size = [64, 128, 128]
model = ResampleWrapper(encoder, target_shape=img_size)

config["pretrained_encoder"] = encoder

