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
    def __init__(self, image_paths, ages, labels, pe=False, transform=None, transform_age=None):
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
        self.pe = pe
        self.transform = transform
        self.transform_age = transform_age

        if pe:
            # Example positional encoding; ensure that its dimensions match your image size.
            self.pos_en = positionalencoding2d(4, 128, 128).detach().cpu().numpy()
            
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
            

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the 3D MRI image from disk
        in_im = np.load(self.image_paths[idx])
        out_im = in_im.copy()
        # in_im = np.array(nib.load(self.image_paths[idx]).get_fdata(), dtype=np.float32)
        
        # Optionally add positional encoding as additional channels
        if self.pe:
            image = np.concatenate([image, self.pos_en], axis=0)
        
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
        in_im = torch.tensor(in_im, dtype=torch.float32).unsqueeze(0)  # Ensure a channel dimension
        out_im = torch.tensor(out_im, dtype=torch.float32).unsqueeze(0)  # Ensure a channel dimension

        # Load and possibly augment age
        age = self.ages[idx]
        if self.transform_age:
            age = age + np.random.uniform(-self.transform_age, self.transform_age)

        # Load label
        label = self.labels[idx]

        age = torch.tensor(age, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

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