import numpy as np
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math 
from torch.utils.data import Sampler
import random
import torchio as tio  # TorchIO is a popular library for 3D medical image augmentation
import warnings
from tqdm import tqdm
import json


def list_files_with_extension(directory, extension):
    """
    Returns a list of file paths with a specific extension in the specified directory
    and its subdirectories.

    Args:
        directory (str): Directory to start the search from.
        extension (str): File extension to filter the files.

    Returns:
        List[str]: List of file paths with the specified extension.
    """
    file_paths = []
    file_names = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(root.split(directory)[-1])
                file_names.append(file)
    return file_paths, file_names

def remove_nan(image):
    if np.isnan(image).any():
        print("Warning: NaN detected, replacing with 0")
        image = np.nan_to_num(image)  # Replace NaNs with 0
    return image

def normalize(image):
    """ Normalize an image while handling extreme values and NaNs. """
    image = np.nan_to_num(image)  # Replace NaNs with 0
    min_val, max_val = np.percentile(image, 1), np.percentile(image, 99)  # Use robust min/max
    image = np.clip(image, min_val, max_val)  # Remove extreme outliers
    if max_val - min_val == 0:  # Prevent division by zero
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val + 1e-8)

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def get_label(image_id, labels_df, column='disease_label'):
    try: 
        return labels_df.query(f"OASIS_session_label == '{image_id}'")[column].tolist()[0]
    except:
        print(f"Label {image_id} not found")
        return np.nan
    
class BrainMRIDataset(Dataset):
    def __init__(self, image_paths, ages, labels, transform=None, transform_age=None, cache=False, sparse_path=None, return_seg=False):
        """
        Parameters:
            image_paths (list or np.array): Paths to the MRI image files.
            ages (list or np.array): Corresponding ages.
            labels (list or np.array): Labels (0 or 1) for each sample.
            pe (bool): Whether to include positional encoding.
            transform (callable): Data augmentation transform to apply on the images.
                                  (Recommended: a TorchIO transform or a custom callable).
                                  'default': default aumentation operations.
            transform_age (float): np.random.uniform(-transform_age, +transform_age).
        """
        self.image_paths = np.array(image_paths)
        self.ages = np.array(ages)
        self.labels = np.array(labels)
        self.transform = transform
        self.transform_age = transform_age
        self.cache = cache
        self.sparse_path=sparse_path
        self.return_seg=return_seg
            
        if transform == "default":
            augmentation_transforms = tio.Compose([
                                                    tio.RandomAffine(
                                                        scales=(0.9, 1.1),        # Random scaling
                                                        degrees=15,               # Random rotation in degrees
                                                        translation=10            # Random translation in mm
                                                    ),
                                                    tio.RandomFlip(axes=('LR',)),  # Random left-right flip
                                                    tio.RandomNoise(mean=0.0, std=0.05),  # Add Gaussian noise
                                                    # tio.RandomBiasField(coefficients=0.5) # Simulate intensity inhomogeneities !too slow
                                                ])
            self.transform = augmentation_transforms

        # Preload images if cache is enabled
        if self.cache:
            self.cached_img = np.ones((len(self.image_paths), 176, 208, 160), dtype=np.uint8)
            self.cached_seg = np.ones((len(self.image_paths), 176, 208, 160), dtype=np.uint8)

            for i, path in enumerate(self.image_paths):
                print(f"Loading {i+1}/{len(self.image_paths)} {(i+1)/len(self.image_paths)*100:.1f}% ...", end='\r')
                img_seg = np.load(path, allow_pickle=True).item()
                
                image = img_seg['image']
                seg = img_seg['seg']

                assert image.shape == (176, 208, 160), f"Unexpected shape: {image.shape}"
                assert seg.shape == (176, 208, 160), f"Unexpected shape: {seg.shape}"

                self.cached_img[i] = image.copy()
                self.cached_seg[i] = seg.copy()

            print()

        elif self.sparse_path:
            with open(self.sparse_path, 'r') as f:
                self.common_bb = json.load(f)
            self.common_dim = np.array(self.common_bb['Hippocampus (lh)'][1]) - np.array(self.common_bb['Hippocampus (lh)'][0])
            self.dementia_subcortical_indices = {
                "Hippocampus (lh)": 17,
                "Hippocampus (rh)": 53,
                "Amygdala (lh)": 18,
                "Amygdala (rh)": 54,
                "Thalamus (lh)": 10,
                "Thalamus (rh)": 49,
                "Caudate (lh)": 11,
                "Caudate (rh)": 50,
                "Putamen (lh)": 12,
                "Putamen (rh)": 51,
            }


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Use cached image or load from disk
        if self.cache:
            in_im = self.cached_img[idx]
            in_im = in_im.astype(np.float32) / 255.0
            seg = self.cached_seg[idx]
            
        elif self.sparse_path:
            img_seg = np.load(self.image_paths[idx], allow_pickle=True).item()
            img = img_seg['image']
            seg = img_seg['seg']

            in_im = []
            for key, value in self.dementia_subcortical_indices.items():
                if key in self.common_bb and self.common_bb[key] is not None:
                    x0, x1 = self.common_bb[key][0][0], self.common_bb[key][1][0]
                    y0, y1 = self.common_bb[key][0][1], self.common_bb[key][1][1]
                    z0, z1 = self.common_bb[key][0][2], self.common_bb[key][1][2]

                    # Create a mask for the current subcortical region
                    region_mask = (seg == value)

                    # Apply BB and then mask (if you only want the region within the BB)
                    cropped_region = img[x0:x1, y0:y1, z0:z1]
                    cropped_mask = region_mask[x0:x1, y0:y1, z0:z1]
                    in_im.append(cropped_region * cropped_mask) # Element-wise multiplication
            in_im = np.array(in_im)
            in_im = in_im.astype(np.float32) / 255.0

        else:
            img_seg = np.load(self.image_paths[idx], allow_pickle=True).item()
            in_im = img_seg['image']
            in_im = in_im.astype(np.float32) / 255.0
            seg = img_seg['seg']


        out_im = in_im.copy()
        # in_im = np.array(nib.load(self.image_paths[idx]).get_fdata(), dtype=np.float32)
        
        # Apply data augmentation if a transform is provided
        if self.transform:
            
            # TorchIO expects a dictionary with 'image' as a key and a 4D tensor (C, H, W, D)
            # Here, we first convert the image to a tensor with a channel dimension.
            image_tensor = torch.tensor(in_im, dtype=torch.float32).unsqueeze(0)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
            subject = self.transform(subject)
            # Get the augmented image back as a numpy array, and remove the extra channel if necessary.
            out_im = subject.image.data.squeeze(0).numpy()
            

        # Convert image, age, and label to tensors.
        # Note: The image is converted again if no transform was applied.
        in_im = torch.tensor(in_im, dtype=torch.float32)  # Ensure a channel dimension
        seg = torch.tensor(seg, dtype=torch.float32)  # Ensure a channel dimension
        out_im = torch.tensor(out_im, dtype=torch.float32)  # Ensure a channel dimension
        if len(in_im.shape) < 4:
            in_im = in_im.unsqueeze(0)
            seg = seg.unsqueeze(0)
            out_im = out_im.unsqueeze(0)

        # Load and possibly augment age
        age = self.ages[idx]
        if self.transform_age:
            age = age + np.random.uniform(-self.transform_age, self.transform_age)

        # Load label
        label = self.labels[idx]

        age = torch.tensor(age, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)


        if self.return_seg:
            sample = {
                'input': out_im,
                'output': in_im,
                'age': age,
                'seg': seg,
                'label': label
            }
        else:
            sample = {
                'input': out_im,
                'output': in_im,
                'age': age,
                'label': label
            }
        return sample
    
    
class StratifiedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset (Dataset): The dataset to sample from. It is assumed that dataset.labels
                               contains the labels for each sample.
            batch_size (int): The batch size (must be even for perfectly balanced sampling).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        assert batch_size % 2 == 0, "Batch size must be even for balanced stratified sampling."
        self.samples_per_class = batch_size // 2

        # Create lists of indices for each class
        self.alzheimer_indices = [i for i, label in enumerate(dataset.labels) if label == 1]
        self.healthy_indices = [i for i, label in enumerate(dataset.labels) if label == 0]

        # Ensure both lists are the same length by oversampling the smaller class
        max_samples = max(len(self.alzheimer_indices), len(self.healthy_indices))
        self.alzheimer_indices = (self.alzheimer_indices * (max_samples // len(self.alzheimer_indices) + 1))[:max_samples]
        self.healthy_indices = (self.healthy_indices * (max_samples // len(self.healthy_indices) + 1))[:max_samples]

    def __iter__(self):
        # Shuffle indices for each class at the start of each epoch
        alz_indices = self.alzheimer_indices.copy()
        healthy_indices = self.healthy_indices.copy()
        random.shuffle(alz_indices)
        random.shuffle(healthy_indices)

        num_batches = len(alz_indices) // self.samples_per_class  # Now based on the largest class

        for i in range(num_batches):
            start = i * self.samples_per_class
            end = start + self.samples_per_class
            batch_alz = alz_indices[start:end]
            batch_healthy = healthy_indices[start:end]
            yield batch_alz + batch_healthy

    def __len__(self):
        return len(self.alzheimer_indices) // self.samples_per_class  # Now accounts for full dataset


class ToTensor:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # image = np.expand_dims(image, axis=0)  # Add channel dimension
        # mask = np.expand_dims(mask, axis=0)    # Add channel dimension
        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float()}