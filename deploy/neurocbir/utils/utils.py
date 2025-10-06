
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import yaml
import logging

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

def load_nifti(img_path, target_shape=(160, 176, 208), start=[48, 38, 10]):
    """Load a nii.gz, .nii or mgz file given a relative path from metadata."""
    print(f"Loading preprocessed image {img_path} ...")
    img = nib.load(img_path).get_fdata()
    print(f"Normalizing intensity levels in the range 0 to 1...")
    img = normalize_to_unit_range(img)
    print(f"Cropping image to {target_shape} ...")
    img = crop_mri(img, target_shape=target_shape, start=start)
    return img

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