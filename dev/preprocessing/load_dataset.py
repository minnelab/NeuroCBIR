import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import math 
import random
from collections import defaultdict
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


# >>> Batched data loader
class LookupNPZDataset(Dataset):
    def __init__(self, metadata, batch_file, use_segmentation=True):
        self.metadata = metadata[metadata["batch_file"] == batch_file].reset_index(drop=True)
        self.use_segmentation = use_segmentation
        self.batch_file = batch_file

        # Load data for this batch
        npz_data = np.load(self.batch_file, allow_pickle=True)

        self.images = np.array(npz_data["images"])
        self.guids = np.array(npz_data["GUID"])

        if self.use_segmentation and "segmentations" in npz_data:
            self.segmentations = np.array(npz_data["segmentations"])
        else:
            self.segmentations = None

        npz_data.close()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        guid = row["GUID"]
        index_in_batch = np.where(self.guids == guid)[0][0] # row["index_in_batch"]
        image = self.images[index_in_batch].copy().astype(float) / 255.0  # shape: [D, H, W]
        image = np.expand_dims(image, axis=0) # shape: [1, D, H, W]

        sample = {
            "GUID": guid,
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
            
        # Final metadata
        add_to_metadata_list = []
        for idx in range(self.__len__()):
                sample = self.__getitem__(idx)
                guid = sample["guid"]
                add_to_metadata_list.append({"GUID": guid, "idx": idx})
        add_to_metadata_df = pd.DataFrame(add_to_metadata_list)
        self.metadata = pd.merge(self.metadata, add_to_metadata_df, on="GUID", how="inner")
        self.metadata.set_index("idx", inplace=True)

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


def get_balanced_batch(dataset, batch_size=32, group_size=4, groups_per_batch=3, 
                       group_key="subject", device="cuda", subset_indices=None):
    """
    Create a batch with group-structured samples, and fill remaining slots randomly (without duplicates).
    
    Args:
        dataset: A dataset with __getitem__ returning dicts with 'emb', 'guid', and optionally metadata fields.
        batch_size: Total number of samples in the batch.
        group_size: Number of samples per group.
        groups_per_batch: Number of groups to include.
        group_key: Key in dataset.metadata used to group samples (e.g., 'subject').
        device: Device to move tensors to.
        subset_indices: Optional list of indices to restrict sampling to a subset of the dataset.

    Returns:
        Dict with tensors for 'emb', 'guid', 'labels', and any extra metadata fields.
    """

    if not isinstance(group_key, list):
        group_key = [group_key]
        
    if subset_indices is None:
        subset_indices = list(range(len(dataset)))

    # Prepare metadata
    metadata = dataset.metadata
    comb_name = '-'.join(group_key)

    # Avoid recomputing if already present
    if comb_name not in metadata.columns:
        metadata[comb_name] = metadata[group_key].agg('-'.join, axis=1)

    if not getattr(dataset, 'subject_to_indices', None):
        subject_to_indices = defaultdict(list)
        # for idx, row in metadata.iterrows():
        for idx in subset_indices:
            row = metadata.loc[idx]
            subject_to_indices[row[comb_name]].append(idx)
            dataset.subject_to_indices = subject_to_indices
    else:
        subject_to_indices = dataset.subject_to_indices

    # Pick eligible subjects with enough samples
    if not getattr(dataset, 'eligible_subjects', None):
        eligible_subjects = [s for s, idxs in subject_to_indices.items() if len(idxs) >= group_size]
        if len(eligible_subjects) < groups_per_batch:
            raise ValueError(f"Not enough {group_key}s for the requested batch size. Found eligible_subjects={len(eligible_subjects)}, need at least groups_per_batch={groups_per_batch}.")
        dataset.eligible_subjects = eligible_subjects
    else:
        eligible_subjects = dataset.eligible_subjects

    random.shuffle(eligible_subjects)

    batch_indices = set()

    # Sample group-based subjects
    for subject in eligible_subjects[:groups_per_batch]:
        group_indices = random.sample(subject_to_indices[subject], group_size)
        batch_indices.update(group_indices)

    # Fill remaining slots with random (non-repeating) indices
    remaining_slots = batch_size - len(batch_indices)
    if remaining_slots > 0:
        all_indices = list(set(subset_indices) - batch_indices)
        extra_indices = random.sample(all_indices, min(remaining_slots, len(all_indices)))
        batch_indices.update(extra_indices)

    batch_indices = list(batch_indices)
    
    # Initialize output
    output = {"emb": [], "guid": [], "label": [], "subject": []}
    sample_keys = dataset[0].keys()
    extra_fields = [k for k in sample_keys if k not in ("emb", "guid")]

    for field in extra_fields:
        output[field] = []

    # Build batch
    for idx in batch_indices:
        sample = dataset[idx]
        output["emb"].append(sample["emb"].unsqueeze(0))
        output["guid"].append(sample["guid"])
        subject_label = dataset.metadata.loc[idx, comb_name]
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
        '''
            n_structs: number of structs to include. If -1, all are included
        '''
        self.metadata = metadata[metadata["batch_file"] == batch_file].reset_index(drop=True)
        self.batch_file = batch_file

        # Load data for this batch
        npz_data = np.load(self.batch_file, allow_pickle=True)
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
        guid = row["GUID"]
        index_in_batch = np.where(self.guids == guid)[0][0] # row["index_in_batch"]
        
        brain = self.images[index_in_batch].copy().astype(np.float32) / 255.0  # [D, H, W]
        brain = np.expand_dims(brain, axis=0)  # [1, D, H, W]

        seg = self.segmentations[index_in_batch]  # [D, H, W]
        seg = np.expand_dims(seg, axis=0)  # [1, D, H, W]

        # Initialize sample container
        sample = {
            "GUID": [],
            "struct_name": [],
            "struct_map_id": [],
            "image": []
        }

        # Sample N random structures
        i_rows = np.random.choice(len(self.labels_bb_df), len(self.labels_bb_df), replace=False) if self.n_structs > 0 else range(len(self.labels_bb_df))

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

            # Apply mask
            struct = patch_brain * patch_seg

            # if self.n_structs > 0:
            if not np.any(patch_seg): # Check that patch_seg is not zeros
                continue
            if np.isnan(struct).any(): # NaN check
                continue

            # Convert to tensor and resize to (1, 64, 64, 64)
            struct_tensor = torch.from_numpy(struct).unsqueeze(0).half()  # [1, 1, D, H, W]
            resized_struct = F.interpolate(struct_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)

            
            sample["GUID"].append(guid)
            sample["struct_name"].append(struct_name)
            sample["struct_map_id"].append(struct_map_id)
            sample["image"].append(resized_struct)

            if len(sample["GUID"]) == self.n_structs:
                break

        if len(sample["image"]) == 0:
            raise ValueError(f"No valid structures found for index {idx}. GUID {guid}")

        # Stack
        sample["image"] = torch.cat(sample["image"], dim=0)  # [N, 1, 64, 64, 64]

        return sample

class RegionEmbBatchedDataset(Dataset):
    def __init__(self, metadata, batch_files, metadata_fields=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata with at least 'GUID' and 'batch_file'.
            batch_files (str or list): Path(s) to .npz files with 'mus', 'GUID', and 'LabelName'.
            metadata_fields (list or None): Optional list of metadata fields to return.
        """
        if isinstance(batch_files, str):
            batch_files = [batch_files]

        self.metadata_fields = metadata_fields if metadata_fields else []

        self.metadata = metadata[metadata["batch_file"].isin(batch_files)].reset_index(drop=True)
        valid_guids = set(self.metadata["GUID"])

        all_embs = []
        all_guids = []
        all_struct_names = []

        for batch_file in batch_files:
            npz_data = np.load(batch_file)
            embs = npz_data["mus"]                    # (N, 113, ...)
            guids = npz_data["GUID"]                 # (N,)
            struct_names = npz_data["struct_name"]     # (113,)
            npz_data.close()

            # Keep only the valid guid rows
            keep_mask = [guid in valid_guids for guid in guids]
            embs = embs[keep_mask]        # shape (N_keep, 113, ...)
            guids = guids[keep_mask]      # shape (N_keep,)
            struct_names = struct_names[keep_mask]      # shape (N_keep,)

            all_embs.append(embs)
            all_guids.append(guids)
            all_struct_names.append(struct_names)

        self.embs = np.concatenate(all_embs, axis=0)
        self.guids = np.concatenate(all_guids, axis=0)
        self.struct_names = np.concatenate(all_struct_names, axis=0)

        # Build metadata lookup
        self.guid_to_metadata = self.metadata.set_index("GUID").to_dict(orient="index")

        if self.metadata_fields:
            keep_indices = []
            for i, guid in enumerate(self.guids):
                row = self.guid_to_metadata.get(guid, {})
                if all(pd.notna(row.get(field)) for field in self.metadata_fields):
                    keep_indices.append(i)

            self.embs = self.embs[keep_indices]
            self.guids = self.guids[keep_indices]
            self.struct_names = self.struct_names[keep_indices]

            filtered_guids = set(self.guids)
            self.metadata = self.metadata[self.metadata["GUID"].isin(filtered_guids)].reset_index(drop=True)
            self.guid_to_metadata = self.metadata.set_index("GUID").to_dict(orient="index")

        # Final metadata
        add_to_metadata_list = []
        for idx in range(self.__len__()):
                sample = self.__getitem__(idx)
                guid = sample["guid"]
                struct_name = sample["struct_name"]
                add_to_metadata_list.append({"GUID": guid, "struct_name": struct_name, "idx": idx})
        add_to_metadata_df = pd.DataFrame(add_to_metadata_list)
        self.metadata = pd.merge(self.metadata, add_to_metadata_df, on="GUID", how="inner")
        self.metadata.set_index("idx", inplace=True)

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        emb = self.embs[idx].copy().astype(float)
        guid = self.guids[idx]
        struct_name = self.struct_names[idx]

        sample = {
            "guid": guid,
            "emb": torch.from_numpy(emb).float(),
            "struct_name": struct_name
        }

        if self.metadata_fields:
            metadata_row = self.guid_to_metadata.get(guid, {})
            for field in self.metadata_fields:
                sample[field] = metadata_row.get(field, None)

        return sample

class SingleStructDataset(Dataset):
    def __init__(self, metadata, batch_files, labels_bb_df, target_struct_name):
        '''
        Args:
            metadata: DataFrame with info about all samples.
            batch_file: Path to .npz file containing brain images and segmentations.
            labels_bb_df: DataFrame with bounding boxes and structure IDs.
            target_struct_name: Name of the subcortical structure to extract (must match LabelName column).
        '''
        if isinstance(batch_files, str):
            batch_files = [batch_files]

        self.metadata = metadata[metadata["batch_file"].isin(batch_files)].reset_index(drop=True)
        self.batch_files = batch_files
        self.target_struct_name = target_struct_name
        self.structs = []
        self.masks = []
        self.guids = []
        self.struct_map_id = None

        # Load data for this batch
        # npz_data = np.load(self.batch_file)
        # images = np.array(npz_data["images"])
        # segmentations = np.array(npz_data["segmentations"])
        # guids = np.array(npz_data["GUID"])
        # npz_data.close()



        for batch_file in batch_files:
            npz_data = np.load(batch_file)
            images = np.array(npz_data["images"])
            segmentations = np.array(npz_data["segmentations"])
            guids = np.array(npz_data["GUID"])
            npz_data.close()

            # Filter the label row for the selected structure
            struct_row_df = labels_bb_df.query(f"LabelName == '{self.target_struct_name}' and Use == 1").reset_index(drop=True)
            if len(struct_row_df) == 0:
                raise ValueError(f"Structure '{self.target_struct_name}' not found in labels_bb_df with Use == 1.")
            struct_row = struct_row_df.iloc[0]
            self.struct_map_id = struct_row["MapID"]

            # Bounding box
            x1, x2 = int(struct_row["min_x"]), int(struct_row["max_x"])
            y1, y2 = int(struct_row["min_y"]), int(struct_row["max_y"])
            z1, z2 = int(struct_row["min_z"]), int(struct_row["max_z"])

            # Preprocess all samples
            # for i in range(len(self.metadata)):
            for guid, image, seg in zip(guids, images, segmentations):
                # row = self.metadata.iloc[i]
                # index_in_batch = row["index_in_batch"]
                # guid = row["GUID"]

                brain = image.astype(np.float32) / 255.0  # Normalize
                brain = np.expand_dims(brain, axis=0)  # [1, D, H, W]
                seg = np.expand_dims(seg, axis=0)  # [1, D, H, W]

                # Crop and mask
                patch_brain = brain[:, x1:x2, y1:y2, z1:z2]
                patch_seg = (seg[:, x1:x2, y1:y2, z1:z2] == self.struct_map_id)

                if not np.any(patch_seg):
                    continue
                struct = patch_brain * patch_seg

                if np.isnan(struct).any():
                    continue

                # Resize to (1, 64, 64, 64)
                struct_tensor = torch.from_numpy(struct).unsqueeze(0).half()  # [1, 1, D, H, W]
                resized_struct = F.interpolate(struct_tensor, size=(64, 64, 64), mode='trilinear', align_corners=False)
                self.structs.append(resized_struct[0])  # [1, 64, 64, 64]
                self.masks.append(torch.from_numpy(patch_seg.astype(np.float32)))
                self.guids.append(guid)

        if len(self.structs) == 0:
            raise RuntimeError(f"No valid samples found for structure '{self.target_struct_name}'.")

    def __len__(self):
        return len(self.structs)

    def __getitem__(self, idx):
        return {
                    "GUID": self.guids[idx],
                    "struct_name": self.target_struct_name,
                    "struct_map_id": self.struct_map_id,
                    "image": self.structs[idx],           # [1, 64, 64, 64]
                    "mask": self.masks[idx]               # [1, 64, 64, 64]
                }


class SingleRegionEmbBatchedDataset(Dataset):
    def __init__(self, metadata, batch_files, target_struct_name, metadata_fields=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata with at least 'GUID' and 'batch_file'.
            batch_files (str or list): Path(s) to .npz files with 'mus', 'GUID', and 'LabelName'.
            metadata_fields (list or None): Optional list of metadata fields to return.
        """
        if isinstance(batch_files, str):
            batch_files = [batch_files]

        self.metadata_fields = metadata_fields if metadata_fields else []

        self.metadata = metadata[metadata["batch_file"].isin(batch_files)].reset_index(drop=True)
        valid_guids = set(self.metadata["GUID"])

        all_embs = []
        all_guids = []
        all_struct_names = []

        for batch_file in batch_files:
            npz_data = np.load(batch_file)
            embs = npz_data["mus"]                    # (N, 113, ...)
            guids = npz_data["GUID"]                 # (N,)
            struct_names = npz_data["struct_name"]     # (113,)
            npz_data.close()

            # Keep only the valid guid rows
            keep_mask = [(guid in valid_guids) and (struct_name == target_struct_name) for guid, struct_name in zip(guids, struct_names)]
            embs = embs[keep_mask]        # shape (N_keep, 113, ...)
            guids = guids[keep_mask]      # shape (N_keep,)
            struct_names = struct_names[keep_mask]      # shape (N_keep,)

            all_embs.append(embs)
            all_guids.append(guids)
            all_struct_names.append(struct_names)

        self.embs = np.concatenate(all_embs, axis=0)
        self.guids = np.concatenate(all_guids, axis=0)
        self.struct_names = np.concatenate(all_struct_names, axis=0)

        # Build metadata lookup
        self.guid_to_metadata = self.metadata.set_index("GUID").to_dict(orient="index")

        if self.metadata_fields:
            keep_indices = []
            for i, guid in enumerate(self.guids):
                row = self.guid_to_metadata.get(guid, {})
                if all(pd.notna(row.get(field)) for field in self.metadata_fields):
                    keep_indices.append(i)

            self.embs = self.embs[keep_indices]
            self.guids = self.guids[keep_indices]
            self.struct_names = self.struct_names[keep_indices]

            filtered_guids = set(self.guids)
            self.metadata = self.metadata[self.metadata["GUID"].isin(filtered_guids)].reset_index(drop=True)
            self.guid_to_metadata = self.metadata.set_index("GUID").to_dict(orient="index")

        # Final metadata
        add_to_metadata_list = []
        for idx in range(self.__len__()):
                sample = self.__getitem__(idx)
                guid = sample["guid"]
                struct_name = sample["struct_name"]
                add_to_metadata_list.append({"GUID": guid, "struct_name": struct_name, "idx": idx})
        add_to_metadata_df = pd.DataFrame(add_to_metadata_list)
        self.metadata = pd.merge(self.metadata, add_to_metadata_df, on="GUID", how="inner")
        self.metadata.set_index("idx", inplace=True)




    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        emb = self.embs[idx].copy().astype(float)
        guid = self.guids[idx]
        struct_name = self.struct_names[idx]

        sample = {
            "guid": guid,
            "emb": torch.from_numpy(emb).float(),
            "struct_name": struct_name
        }

        if self.metadata_fields:
            metadata_row = self.guid_to_metadata.get(guid, {})
            for field in self.metadata_fields:
                sample[field] = metadata_row.get(field, None)

        return sample
