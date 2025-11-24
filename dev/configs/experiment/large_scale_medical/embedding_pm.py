import torch
import torch.nn as nn
import torch.nn.functional as F
from dev.utils import load_config_from_path
import os
from monai.networks.nets import SwinUNETR
import urllib.request

def load_model(model, model_dict):
    # make sure you load our checkpoints
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    for k in current_model_dict.keys():
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            print(k)
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict, strict=True)
    return model



# Load config
config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))
config.update({"batch_size": 4})

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


class SwinUNETR_Features(SwinUNETR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # define 3D average pooling over the D,H,W dimensions
        self.global_avgpool = nn.AvgPool3d(kernel_size=(3, 3, 3))  # matches your feature map size

    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])

        hidden_states_out = self.swinViT(x_in, self.normalize)
        features = self.encoder10(hidden_states_out[4])  # shape: [B, 1536, 3, 3, 3]

        # global average pooling
        features = self.global_avgpool(features)  # shape: [B, 1536, 1, 1, 1]
        features = features.view(features.size(0), -1)  # flatten to [B, 1536]

        return features



# Define the model with same parameters used in training
img_size=(96, 96, 96)
base_model = SwinUNETR_Features(
    img_size=img_size,   # ROI size
    in_channels=1,           # for MRI
    out_channels=14,         # segmentation classes (can ignore if using as encoder)
    feature_size=96,         # 96 = Large (L), 48 = Base (B), 192 = Huge (H)
    use_checkpoint=True
)


# Target directory and file
weights_dir = os.path.join(config["output_dir"])
weights_file = "VoComni_L.pt"
weights_path = os.path.join(weights_dir, weights_file)

# Download the weights if they don't exist
url = "https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_L.pt?download=true"
if not os.path.exists(weights_path):
    print(f"Downloading VoComni_L weights to {weights_path}...")
    urllib.request.urlretrieve(url, weights_path)
    print("Download complete!")

# Load pretrained weights
pretrained_path = weights_path   # download from their release
model_dict = torch.load(pretrained_path, weights_only=False, map_location="cpu")

base_model = load_model(base_model, model_dict)

# Build wrapped model
model = ResampleWrapper(base_model, target_shape=img_size)

# Add to config
config["pretrained_encoder"] = model
