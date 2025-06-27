
if __name__ != "__main__":
    raise Exception("This script is only intended to be run as main!")

import torch
import torch.nn.functional as F
import os
from monai.networks.nets.autoencoderkl import AutoencoderKL
from monai.utils import set_determinism
from preprocessing.load_dataset import list_files_with_extension
import numpy as np
import os
from tqdm import tqdm

### Input data
# Path to dataset
load_ds_path = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/batched_adni/"
# Path to save embeddings
save_ds_path = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/batched_adni__emb/"
os.makedirs(save_ds_path, exist_ok=True) # create if not exist
# Files to load/save extension
extension = ".npz"
# Pretrained weights for the VAE
ckpt_path = "./data/pretrained_models/model_autoencoder.pt"
# Preparing image for using as input of the VAE
target_shape = [1, 160, 224, 160] # Desired shape: [1, 160, 224, 160]


### Loading VAE
set_determinism(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained networks
autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256),
    latent_channels=8,
    num_res_blocks=2,
    norm_num_groups=32,
    norm_eps=1e-06,
    attention_levels=(False, False, False),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    include_fc=False

)
autoencoder.to(device)

# Load pretrained model VAE
state_dict = torch.load(ckpt_path)
autoencoder.load_old_state_dict(state_dict)
print("VAE weights successfully loaded!")

### Loading the dataset
# Parched dataset file paths
file_paths, file_names = list_files_with_extension(load_ds_path, extension=extension)
file_names.sort()

print("Files to be processed: ")
[print(os.path.join(load_ds_path, file_path, file_name)) for file_path, file_name in zip(file_paths, file_names)]

# Initialization of lists
list_z_mu = []
list_sample_id = []

# Loop: Load the batched dataset
for file_path, file_name in zip(file_paths, file_names):
    file_to_load = os.path.join(load_ds_path, file_path, file_name)
    data = np.load(file_to_load)
    images = data['images']
    # segmentations = data['segmentations'] # unused
    sample_ids = data['ids']
    # Optionally: close file if it's a lazy loader (not strictly needed if you load all arrays)
    data.close()

    # Loop over batch
    for img, sample_id in tqdm(zip(images, sample_ids), desc=f"Processing {file_to_load}", total=len(images), ncols=180):
    
        # Preparing image for using as input of the VAE
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        current_shape = list(img.shape)
        img = torch.tensor(img)

        # Calculate padding: (left, right, top, bottom, front, back)
        pad_d = (0, target_shape[3] - current_shape[3])  # depth (no change)
        pad_h = (0, target_shape[2] - current_shape[2])  # height
        pad_w = (0, target_shape[1] - current_shape[1])  # width

        # Flatten padding list: reverse order for F.pad â†’ (D, H, W)
        padding = pad_d + pad_h + pad_w  # [depth, height, width]

        # Pad image
        padded_img = F.pad(img, padding).unsqueeze(0).to(device)

        # Inference through autoencoder
        autoencoder.eval()
        with torch.no_grad():
            z_mu, z_sigma = autoencoder.encode(padded_img)
        z_mu = z_mu.squeeze(0) # Remove batch dimension

        # Convert z_mu to numpy and float16
        z_mu = z_mu.cpu().numpy().astype(np.float16)

        # Append to the list
        list_z_mu.append(z_mu)
        list_sample_id.append(sample_id)

    # Save emb to save_ds_path + file_name
    filename = os.path.join(save_ds_path, file_name)
    np.savez_compressed(
                        filename,
                        mus=np.stack(list_z_mu),
                        ids=np.stack(list_sample_id),
                        )
    print(f"Saved final batch {filename} with {len(list_z_mu)} samples.")

    # Clear lists 
    list_z_mu.clear()
    list_sample_id.clear()

    


    
