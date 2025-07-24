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
from preprocessing.padding import pad_mri_to_shape
from collections import defaultdict
import math
import pandas as pd

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
    

class BrainSVFDataset(Dataset):
    def __init__(self, svf_paths, ages_dif, labels):
        """
        Parameters:
            image_paths (list or np.array): Paths to the MRI image files.
            ages (list or np.array): Corresponding ages.
            labels (list or np.array): Labels (0 or 1) for each sample.
        """
        self.svf_paths = np.array(svf_paths)
        self.ages_dif = np.array(ages_dif)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.svf_paths)

    def __getitem__(self, idx):

        svf = np.load(self.svf_paths[idx])
        svf = torch.tensor(svf, dtype=torch.float32).permute((-1, 1, 2, 0))

        age_dif = torch.tensor(self.ages_dif[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        return {'svf': svf,
                'age_dif': age_dif,
                'label': label}
    

class SubcorticalSVFDataset(Dataset):
    def __init__(self, svf_paths, ages_dif, labels, mri_svf_sessions, sparse_path):
        """
        Parameters:
            image_paths (list or np.array): Paths to the MRI image files.
            ages (list or np.array): Corresponding ages.
            labels (list or np.array): Labels (0 or 1) for each sample.
        """
        self.svf_paths = np.array(svf_paths)
        self.ages_dif = np.array(ages_dif)
        self.labels = np.array(labels)
        self.sparse_path = sparse_path
        self.mri_svf_sessions = mri_svf_sessions

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
        return len(self.svf_paths)

    def __getitem__(self, idx):

        whole_svf = np.load(self.svf_paths[idx])
        # whole_svf = torch.tensor(whole_svf, dtype=torch.float32).permute((-1, 1, 2, 0))

        seg_1 = pad_mri_to_shape(np.load(self.mri_svf_sessions[idx,0], allow_pickle=True).item()['seg'], target_shape=(96*2, 112*2, 80*2))[::2, ::2, ::2]
        seg_2 = pad_mri_to_shape(np.load(self.mri_svf_sessions[idx,1], allow_pickle=True).item()['seg'], target_shape=(96*2, 112*2, 80*2))[::2, ::2, ::2]

        #############
        # img_seg = np.load(self.image_paths[idx], allow_pickle=True).item()
        # img = img_seg['image']
        # seg = img_seg['seg']

        svf = []
        for key, value in self.dementia_subcortical_indices.items():
            if key in self.common_bb and self.common_bb[key] is not None:
                x0, x1 = self.common_bb[key][0][0]//2, self.common_bb[key][1][0]//2
                y0, y1 = self.common_bb[key][0][1]//2, self.common_bb[key][1][1]//2
                z0, z1 = self.common_bb[key][0][2]//2, self.common_bb[key][1][2]//2

                # Create a mask for the current subcortical region
                region_mask = (seg_1 == value) + (seg_2 == value)

                # Apply BB and then mask (if you only want the region within the BB)
                cropped_region = whole_svf[x0:x1, y0:y1, z0:z1]
                cropped_mask = region_mask[x0:x1, y0:y1, z0:z1]
                svf.append(cropped_region * cropped_mask[..., np.newaxis]) # Element-wise multiplication
                
        svf = np.array(svf)
        #############

        svf = torch.tensor(svf, dtype=torch.float32).permute((0, -1, 2, 3, 1))


        age_dif = torch.tensor(self.ages_dif[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        return {'svf': svf,
                'age_dif': age_dif,
                'label': label}
    
    
    
    
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



# >>> Batched data loader


class LookupNPZDataset(Dataset):
    def __init__(self, metadata, batch_file, use_segmentation=True):
        self.metadata = metadata[metadata["batch_file"] == batch_file].reset_index(drop=True)
        self.use_segmentation = use_segmentation
        self.batch_file = batch_file

        # Load data for this batch
        npz_data = np.load(self.batch_file)

        self.images = np.array(npz_data["images"])
        self.ids = np.array(npz_data["ids"])

        if self.use_segmentation and "segmentations" in npz_data:
            self.segmentations = np.array(npz_data["segmentations"])
        else:
            self.segmentations = None

        npz_data.close()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        index_in_batch = row["index_in_batch"]
        sample_id = row["GUID"]
        image = self.images[index_in_batch].copy().astype(float) / 255.0  # shape: [D, H, W]
        image = np.expand_dims(image, axis=0) # shape: [1, D, H, W]

        sample = {
            "id": sample_id,
            "image": torch.from_numpy(image)
        }

        if self.use_segmentation and self.segmentations is not None:
            seg = self.segmentations[index_in_batch] # shape: [D, H, W]
            seg = np.expand_dims(seg, axis=0) # shape: [1, D, H, W]
            sample["seg"] = torch.from_numpy(seg)

        return sample

# >>> BATCHED DATASET: WHOLE BRAIN <<<
class EmbBatchedDataset(Dataset):
    def __init__(self, metadata, batch_files, metadata_fields=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata with at least 'GUID' and 'batch_file'.
            batch_files (str or list): Path(s) to .npz files with 'mus' and 'GUID'.
            metadata_fields (list or None): Optional list of metadata fields to return.
        """
        if isinstance(batch_files, str):
            batch_files = [batch_files]

        self.metadata_fields = metadata_fields if metadata_fields else []

        # Filter metadata to only those from the batch files
        self.metadata = metadata[metadata["batch_file"].isin(batch_files)].reset_index(drop=True)
        valid_guids_set = set(self.metadata["GUID"])

        # Load all embeddings and GUIDs in bulk
        all_embs = []
        all_guids = []

        for batch_file in batch_files:
            npz_data = np.load(batch_file)
            embs = np.array(npz_data["mus"])
            guids = np.array(npz_data["GUID"])
            npz_data.close()

            # Keep only those in metadata
            keep_mask = [guid in valid_guids_set for guid in guids]
            filtered_embs = embs[keep_mask]
            filtered_guids = guids[keep_mask]

            all_embs.append(filtered_embs)
            all_guids.append(filtered_guids)

        self.embs = np.concatenate(all_embs, axis=0)
        self.guids = np.concatenate(all_guids, axis=0)

        # Build lookup table
        self.guid_to_metadata = self.metadata.set_index("GUID").to_dict(orient="index")

        # Optionally filter out NaNs in requested metadata fields
        if self.metadata_fields:
            keep_indices = []
            for i, guid in enumerate(self.guids):
                row = self.guid_to_metadata.get(guid, {})
                if all(pd.notna(row.get(field)) for field in self.metadata_fields):
                    keep_indices.append(i)

            self.embs = self.embs[keep_indices]
            self.guids = self.guids[keep_indices]

            # Also filter the metadata DataFrame to only keep these guids
            filtered_guids = set(self.guids)
            self.metadata = self.metadata[self.metadata["GUID"].isin(filtered_guids)].reset_index(drop=True)

            # Rebuild the guid_to_metadata lookup for consistency
            self.guid_to_metadata = self.metadata.set_index("GUID").to_dict(orient="index")

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        emb = self.embs[idx].copy().astype(float)
        guid = self.guids[idx]

        sample = {
            "guid": guid,
            "emb": torch.from_numpy(emb).type(torch.FloatTensor)
        }

        if self.metadata_fields:
            metadata_row = self.guid_to_metadata.get(guid, {})
            for field in self.metadata_fields:
                sample[field] = metadata_row.get(field, None)

        return sample


def encode_labels(label_list):
    """
    Converts a list of categorical labels (e.g., subject IDs) to numeric labels.

    Args:
        label_list (List[str]): List of string labels.

    Returns:
        numeric_labels (List[int]): List of numeric labels.
        label_to_index (Dict[str, int]): Mapping from original labels to numeric.
        index_to_label (Dict[int, str]): Reverse mapping (optional).
    """
    unique_labels = sorted(set(label_list))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    numeric_labels = [label_to_index[label] for label in label_list]
    
    return numeric_labels, label_to_index, index_to_label


def get_balanced_batch(dataset, batch_size=32, group_size=4, groups_per_batch=3, group_key="subject", device="cuda"):
    """
    Create a batch with group-structured samples, and fill remaining slots randomly (without duplicates).
    
    Args:
        dataset: A dataset with __getitem__ returning dicts with 'emb', 'guid', and optionally metadata fields.
        batch_size: Total number of samples in the batch.
        group_size: Number of samples per group.
        groups_per_batch: Number of groups to include.
        group_key: Key in dataset.metadata used to group samples (e.g., 'subject').
        device: Device to move tensors to.

    Returns:
        Dict with tensors for 'emb', 'guid', 'labels', and any extra metadata fields.
    """
    metadata = dataset.metadata
    subject_to_indices = defaultdict(list)

    for idx, row in metadata.iterrows():
        subject_to_indices[row[group_key]].append(idx)

    # Pick eligible subjects with enough samples
    eligible_subjects = [s for s, idxs in subject_to_indices.items() if len(idxs) >= group_size]
    if len(eligible_subjects) < groups_per_batch:
        raise ValueError(f"Not enough {group_key}s with ≥{group_size} samples.")

    random.shuffle(eligible_subjects)

    batch_indices = set()

    # Sample group-based subjects
    for subject in eligible_subjects[:groups_per_batch]:
        group_indices = random.sample(subject_to_indices[subject], group_size)
        batch_indices.update(group_indices)

    # Fill remaining slots with random (non-repeating) indices
    remaining_slots = batch_size - len(batch_indices)
    if remaining_slots > 0:
        all_indices = list(set(range(len(dataset))) - batch_indices)
        extra_indices = random.sample(all_indices, min(remaining_slots, len(all_indices)))
        batch_indices.update(extra_indices)

    batch_indices = list(batch_indices)

    # Initialize output
    output = {"emb": [], "guid": [], "labels": [], "subject": []}
    sample_keys = dataset[0].keys()
    extra_fields = [k for k in sample_keys if k not in ("emb", "guid")]

    for field in extra_fields:
        output[field] = []

    # Build batch
    for idx in batch_indices:
        sample = dataset[idx]
        output["emb"].append(sample["emb"].unsqueeze(0))
        output["guid"].append(sample["guid"])
        subject_label = dataset.metadata.loc[idx, group_key]
        output["subject"].append(subject_label)
        for field in extra_fields:
            output[field].append(sample[field])

    # Stack tensors
    output["emb"] = torch.cat(output["emb"], dim=0).to(device)
    # Encode labels from subject IDs
    output["label"], _, _ = encode_labels(output["subject"])
    output["label"] = torch.tensor(output["label"], dtype=torch.long, device=device)
    
    return output

class SequentialBatchIterator:
    def __init__(self, dataset, batch_size, device="cuda"):
        """
        Initializes the batch iterator.

        Args:
            dataset: Dataset with __getitem__ returning a dict.
            batch_size: Number of samples per batch.
            device: Device to move tensor fields to.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.current_index = 0
        self.total_samples = len(dataset)
        self.keys = list(dataset[0].keys())  # assumes consistent keys across dataset

    def __iter__(self):
        self.current_index = 0  # Reset for each iteration
        return self

    def __len__(self):
        return math.ceil(self.total_samples / self.batch_size)

    def __next__(self):
        if self.current_index >= self.total_samples:
            raise StopIteration

        start_idx = self.current_index
        end_idx = min(start_idx + self.batch_size, self.total_samples)

        output = {k: [] for k in self.keys}

        for idx in range(start_idx, end_idx):
            sample = self.dataset[idx]
            for k in self.keys:
                val = sample[k]
                if isinstance(val, torch.Tensor):
                    output[k].append(val.unsqueeze(0))  # keep batch dim
                else:
                    output[k].append(val)  # leave as list

        for k in self.keys:
            if output[k] and isinstance(output[k][0], torch.Tensor):
                output[k] = torch.cat(output[k], dim=0).to(self.device).float()

        self.current_index = end_idx
        return output

# >>> BATCHED DATASET: SUBCORTICAL/CORTICAL STRUCTURES <<< 
class SubCorBatDataset(Dataset):
    def __init__(self, metadata, batch_file, labels_bb_df, n_structs=10):
        self.metadata = metadata[metadata["batch_file"] == batch_file].reset_index(drop=True)
        self.batch_file = batch_file

        # Load data for this batch
        npz_data = np.load(self.batch_file)
        self.images = np.array(npz_data["images"])
        self.guids = np.array(npz_data["GUID"])
        self.segmentations = np.array(npz_data["segmentations"])
        npz_data.close()

        self.labels_bb_df = labels_bb_df.query("Use == 1").reset_index(drop=True)
        self.n_structs = n_structs

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        index_in_batch = row["index_in_batch"]
        guid = row["GUID"]

        brain = self.images[index_in_batch].copy().astype(np.float32) / 255.0  # [D, H, W]
        brain = np.expand_dims(brain, axis=0)  # [1, D, H, W]

        seg = self.segmentations[index_in_batch]  # [D, H, W]
        seg = np.expand_dims(seg, axis=0)  # [1, D, H, W]

        # Initialize sample container
        sample = {
            "GUID": [],
            "struct_name": [],
            "image": []
        }

        # Sample N random structures
        i_rows = np.random.choice(len(self.labels_bb_df), len(self.labels_bb_df), replace=False)

        for i_row in i_rows:
            struct_row = self.labels_bb_df.iloc[i_row]
            struct_name = struct_row["LabelName"]
            struct_map_id = struct_row["MapID"]

            # Bounding box
            x1, x2 = int(struct_row["min_x"]), int(struct_row["max_x"])
            y1, y2 = int(struct_row["min_y"]), int(struct_row["max_y"])
            z1, z2 = int(struct_row["min_z"]), int(struct_row["max_z"])

            # Ensure within volume bounds
            patch_brain = brain[:, x1:x2, y1:y2, z1:z2]
            patch_seg = (seg[:, x1:x2, y1:y2, z1:z2] == struct_map_id)

            # Check that patch_seg is not zeros
            if not np.any(patch_seg):
                continue

            # Apply mask
            struct = patch_brain * patch_seg

            # Convert to tensor and resize to (1, 64, 64, 64)
            struct_tensor = torch.from_numpy(struct).unsqueeze(0).half()  # [1, 1, D, H, W]
            resized_struct = F.interpolate(struct_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)

            
            sample["GUID"].append(guid)
            sample["struct_name"].append(struct_name)
            sample["image"].append(resized_struct)

            if len(sample["GUID"]) == self.n_structs:
                break

        # Stack
        sample["image"] = torch.cat(sample["image"], dim=0)  # [N, 1, 64, 64, 64]

        return sample
