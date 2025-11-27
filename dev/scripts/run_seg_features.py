# """
# This script extracts morphometry features from aparc+aseg segmentation masks for all subjects
# and saves them in a CSV file. First column: GUID, rest: features.
# """

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import logging
# from dev.preprocessing.bounding_box_utils import extract_region_features
from joblib import Parallel, delayed
import time
from scipy.ndimage import binary_erosion


def crop_mri(image, target_shape=(160, 176, 208), start=[48, 38, 10]):
    end = np.minimum(np.array(start) + np.array(target_shape), image.shape)
    cropped = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    return cropped

def fast_shape_pca(coords):
    # Downsample if needed for speed
    if len(coords) > 5000:
        coords = coords[np.random.choice(len(coords), 5000, replace=False)]

    # Fast PCA via SVD
    coords_centered = coords - coords.mean(axis=0)
    _, s, _ = np.linalg.svd(coords_centered, full_matrices=False)
    eigvals = s**2
    
    return eigvals

def extract_etiv(aseg_path):
    """
    Reads an aseg.stats file and extracts the eTIV/ICV value.
    Returns float or None if not found.
    """
    with open(aseg_path, "r") as f:
        for line in f:
            if ("eTIV" in line and "EstimatedTotalIntraCranialVol" in line) or \
               ("ICV" in line and "IntraCranialVol" in line):
                parts = [p.strip() for p in line.split(",")]
                try:
                    # The 4th field = numeric value
                    return float(parts[3])
                except:
                    return None
    return None

def extract_region_features(segmentation_mask, etiv, voxel_volume_mm3):
    """
    Compute shape features of a binary segmentation mask, accounting for voxel size.

    Args:
        segmentation_mask: np.ndarray, binary mask of the region
        etiv: float, total intracranial volume in mm^3
        voxel_volume_mm3: float, volume of a voxel in mm^3

    Returns:
        dict: volume, surface_area, compactness, elongation, flatness
    """
    coords = np.argwhere(segmentation_mask)
    if coords.size == 0:
        return {'volume':0,'surface_area':0,'compactness':0,'elongation':0,'flatness':0}
    
    # Volume (normalized by eTIV)
    start_time = time.time()
    vol_mm3 = segmentation_mask.sum() * voxel_volume_mm3  # absolute volume in mm^3
    vol = vol_mm3 / etiv  # normalized volume
       
    # Surface area approximation
    eroded = binary_erosion(segmentation_mask)
    surface_voxels = segmentation_mask.sum() - eroded.sum()
    surface_area_mm2 = surface_voxels * (voxel_volume_mm3 ** (2/3))  # approximate mm^2
    surface_area = surface_area_mm2 / etiv  # normalized
    
    logging.debug(f"Surface area calculation took {time.time() - start_time:.2f} seconds")
    
    # Compactness
    compactness = vol**2 / (surface_area**3 + 1e-6)
    
    # PCA for shape
    start_time = time.time()
    eigvals = fast_shape_pca(coords)  # assumes coords are in voxel space
    if len(eigvals) < 3:
        return {
            "volume": vol.astype(np.float32),
            "surface_area": surface_area.astype(np.float32),
            "compactness": compactness.astype(np.float32),
            "elongation": 0.0,
            "flatness": 0.0
        }
    
    elongation = eigvals[0] / (eigvals[2] + 1e-6)
    flatness = eigvals[1] / (eigvals[2] + 1e-6)
    
    logging.debug(f"PCA calculation took {time.time() - start_time:.2f} seconds")
    
    return {
        'volume': vol,
        'surface_area': surface_area.astype(np.float32),
        'compactness': compactness,
        'elongation': elongation,
        'flatness': flatness
    }

def process_roi(row, seg_data, etiv, voxel_volume_mm3):
    roi_name = row["LabelName"]
    roi_val = row["LabelID"]
    x1, x2 = int(row["min_x"]), int(row["max_x"])
    y1, y2 = int(row["min_y"]), int(row["max_y"])
    z1, z2 = int(row["min_z"]), int(row["max_z"])
    mask = (seg_data[x1:x2, y1:y2, z1:z2] == roi_val)
    feats = extract_region_features(mask, etiv=etiv, voxel_volume_mm3=voxel_volume_mm3)
    return roi_name, feats

def process_subject(row, data_path, labels_bb_df, voxel_volume_mm3):
    guid = row['GUID']
    seg_file = os.path.join(data_path, row['seg'])

    seg_img = nib.load(seg_file)
    seg_data = seg_img.get_fdata().astype(int)
    seg_data = crop_mri(seg_data)
    
    stats_path = row['seg'].split(guid)[0]
    aseg_stats_file = os.path.join(data_path, stats_path, guid, "stats", "aseg.stats")
    logging.debug(guid)
    etiv = int(extract_etiv(aseg_stats_file))
    logging.debug(f"ETIV for {guid}: {etiv}")

    # brain_vol = np.sum(seg_data > 0) * voxel_volume_mm3
    features_row = {"GUID": guid}

    for _, row in labels_bb_df.iterrows():
        roi_name = row["LabelName"]
        roi_val = row["LabelID"]
        x1, x2 = int(row["min_x"]), int(row["max_x"])
        y1, y2 = int(row["min_y"]), int(row["max_y"])
        z1, z2 = int(row["min_z"]), int(row["max_z"])
        mask = (seg_data[x1:x2, y1:y2, z1:z2] == roi_val).astype(bool)
        feats = extract_region_features(mask, etiv, voxel_volume_mm3)
        for k, v in feats.items():
            features_row[f"{roi_name}_{k}"] = v

    return features_row

# --- Main processing ---
def main(config):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_path = config["data_path"]
    segmentation_csv = pd.read_csv(os.path.join(data_path, config["metadata_file_name"]))  # Must contain GUID, file paths
    output_file = os.path.join(config["output_dir"], "morphometry_features.csv")
    labels_df = pd.read_csv(config["labels_path"])
    labels_df = labels_df.query("Use == 1").reset_index(drop=True)
    bb_df = pd.read_csv(config["bb_path"])
    labels_bb_df = pd.merge(labels_df, bb_df, on="LabelName", how="inner") # Merge on the 'GUID' column
    n_jobs = config.get("n_jobs", 4)
    voxel_volume_mm3 = config.get("voxel_volume_mm3", 1)
    checkpoint_every = config.get("checkpoint_every", 20)
    
    all_rows = []
    
    if config["parallel_by"] == "subject":
        iterator = list(segmentation_csv.iterrows())

        # Iterate in chunks for checkpointing
        for start in tqdm(range(0, len(iterator), checkpoint_every),
                        desc="Processing in batches", ncols=100):

            end = min(start + checkpoint_every, len(iterator))
            batch = iterator[start:end]

            # Parallel process this batch
            batch_rows = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(process_subject)(row, data_path, labels_bb_df, voxel_volume_mm3)
                for _, row in batch
            )

            all_rows.extend(batch_rows)  # optional if you want in-memory copy

            # --- Save only the batch ---
            df_batch = pd.DataFrame(batch_rows).round(6)
            if os.path.exists(output_file):
                df_batch.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df_batch.to_csv(output_file, mode='w', header=True, index=False)
            
    else:
        # Parallel by ROI
        for i, (idx, row) in tqdm(enumerate(segmentation_csv.iterrows()), total=len(segmentation_csv), desc="Processing subjects", ncols=150):
            guid = row['GUID']
            seg_file = os.path.join(data_path, row['seg'])  # aparc+aseg NIfTI path
            
            # Load segmentation
            start_time = time.time()
            seg_img = nib.load(seg_file)
            seg_data = seg_img.get_fdata().astype(int)
            seg_data = crop_mri(seg_data)
            logging.debug(f"Loaded segmentation for {guid} in {time.time() - start_time:.2f} seconds")
            
            # Estimate brain volume: sum of all non-zero voxels (approx)
            brain_vol = np.sum(seg_data > 0)
            
            features_row = {"GUID": guid}
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_roi)(row, seg_data, brain_vol)
                for _, row in labels_bb_df.iterrows()
            )
                
            # Combine results into features_row
            features_row = {"GUID": guid}  # or existing dict if already has GUID
            for roi_name, feats in results:
                for k, v in feats.items():
                    features_row[f"{roi_name}_{k}"] = v
            
            all_rows.append(features_row)
            
            # Save to CSV
            if i % checkpoint_every == 0 or i == len(segmentation_csv) - 1:
                df_features = pd.DataFrame(all_rows).round(6)
                df_features.to_csv(output_file, index=False)
                logging.debug(f"\nMorphometry features saved to {output_file}")
            
    # Save to CSV
    df_features = pd.DataFrame(all_rows).round(6)
    df_features.to_csv(output_file, index=False)
    logging.debug(f"\nMorphometry features saved to {output_file}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dev/configs/run_seg_features.py", help='Path to config file (.py) with paths and ROI definitions')
    args = parser.parse_args()
    
    from dev.utils import load_config_from_path
    config = load_config_from_path(args.config)
    
    main(config)
