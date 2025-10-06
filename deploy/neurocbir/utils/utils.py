
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import logging
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def normalize_to_unit_range(image):
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def crop_mri(image, target_shape=(160, 176, 208), start=[48, 38, 10]):
    end = np.minimum(np.array(start) + np.array(target_shape), image.shape)
    cropped = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    return cropped

def load_brain(img_path, crop_target_shape=(160, 176, 208), crop_start=[48, 38, 10]):
    """Load a nii.gz, .nii or mgz file given a relative path from metadata."""
    logger.info(f"Loading preprocessed image {img_path} ...")
    img = nib.load(img_path).get_fdata()
    logger.info(f"Normalizing intensity levels [0, 1] ...")
    img = normalize_to_unit_range(img)
    logger.info(f"Cropping image to {crop_target_shape} ...")
    img = crop_mri(img, target_shape=crop_target_shape, start=crop_start)
    return img

def load_region(brain_path, seg_path, labels_bb_df, region,
                crop_target_shape=(160, 176, 208), crop_start=[48, 38, 10],
                resize_target_shape=[64,64,64]):
    """
    Load brain region

    Args:
        brain_path (str): path to brain image file.
        seg_path (str): path to segmentation image file.
        labels_bb_df (pd.Dataframe): dataframe with the bounding boxes and map identifier per region.
        region (str): regions to be segmented.

    Returns:
        np.array: brain region.
    """

    # Load data
    logger.info(f"Loading preprocessed brain: {brain_path} ...")
    brain = nib.load(brain_path).get_fdata()
    logger.info(f"Loading segmentation: {seg_path} ...")
    seg = nib.load(seg_path).get_fdata()
    
    # Cropping brain and seg
    logger.info(f"Cropping ibrain and seg to {crop_target_shape} ...")
    brain = normalize_to_unit_range(brain)
    
    # Normalizing intensity levels for the whole brain
    logger.info(f"Normalizing intensity levels [0, 1] ...")
    brain = crop_mri(brain, target_shape=crop_target_shape, start=crop_start)
    seg = crop_mri(seg, target_shape=crop_target_shape, start=crop_start)

    # Filter the label row for the selected structure
    logger.info(f"Segmentation region {region}  ...")
    struct_row_df = labels_bb_df.query(f"LabelName == '{region}' and Use == 1").reset_index(drop=True)
    if len(struct_row_df) == 0:
        raise ValueError(f"Structure '{region}' not found in labels_bb_df with Use == 1.")
    struct_row = struct_row_df.iloc[0]
    region_map_id = struct_row["LabelID"]
    
    # Bounding box
    x1, x2 = int(struct_row["min_x"]), int(struct_row["max_x"])
    y1, y2 = int(struct_row["min_y"]), int(struct_row["max_y"])
    z1, z2 = int(struct_row["min_z"]), int(struct_row["max_z"])

    # Preprocess all samples
    patch_brain = brain[x1:x2, y1:y2, z1:z2]
    patch_seg = (seg[x1:x2, y1:y2, z1:z2] == region_map_id)

    struct = patch_brain * patch_seg
    logger.info(f"Region loaded: {region}. Initial shape: {struct.shape}.")
    
    # --- Resample to target_shape using trilinear interpolation ---
    zoom_factors = [t / s for t, s in zip(resize_target_shape, struct.shape)]
    struct_resampled = zoom(struct, zoom_factors, order=1)  # order=1 = trilinear
        
    logger.info(f"Region loaded: {region}. Final shape: {struct_resampled.shape}.")
    return struct_resampled

def retrieve_topk_for_query(query_features, 
                            dataset, 
                            top_k = 3, 
                            feature_column = "features", 
                            guid_column = "GUID"
                            ):
    """
    Retrieve the top-k most similar entries for a subset of queries, 
    using cosine similarity against the full dataset as the retrieval pool.

    Args:
        query_features (np.array): features of the query.
        dataset (pd.DataFrame): Full pool of entries with features and GUIDs.
        top_k (int): Number of top similar entries to retrieve.
        feature_column (str): Column containing the feature vectors.
        guid_column (str): Column with unique scan identifiers (e.g., 'GUID').

    Returns:
        pd.DataFrame: Retrieval results. One row per query, first column is the query GUID,
                      followed by the GUIDs of the top-k retrieved entries.
    """
    # Retrieval pool
    features_matrix = np.stack(dataset[feature_column].values)
    guids = dataset[guid_column].values

    # Compute similarities
    similarities = cosine_similarity(query_features, features_matrix)[0]
        
    # Get top-k
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    return guids[top_k_indices].tolist()