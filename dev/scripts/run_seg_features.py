"""
This script extracts morphometry features from aparc+aseg segmentation masks for all subjects
and saves them in a CSV file. First column: GUID, rest: features.
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import logging
from dev.preprocessing.bounding_box_utils import extract_region_features
from joblib import Parallel, delayed

def process_roi(roi_name, roi_val, seg_data, brain_vol):
    mask = (seg_data == roi_val)
    feats = extract_region_features(mask, brain_vol)
    return roi_name, feats

# --- Main processing ---
def main(config):
    data_path = config["data_path"]
    segmentation_csv = pd.read_csv(os.path.join(data_path, config["metadata_file_name"]))  # Must contain GUID, file paths
    output_file = os.path.join(config["output_dir"], "morphometry_features.csv")
    labels_df = pd.read_csv(config["labels_path"])
    labels_df = labels_df.query("Use == 1").reset_index(drop=True)
    roi_labels = dict(zip(labels_df["LabelName"], labels_df["LabelID"]))
    n_jobs_roi = config.get("n_jobs_roi", 4)
    
    all_rows = []

    for i, (idx, row) in tqdm(enumerate(segmentation_csv.iterrows()), total=len(segmentation_csv), desc="Processing subjects"):
        guid = row['GUID']
        seg_file = os.path.join(data_path, row['seg'])  # aparc+aseg NIfTI path
        
        # Load segmentation
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata().astype(int)
        
        # Estimate brain volume: sum of all non-zero voxels (approx)
        brain_vol = np.sum(seg_data > 0)
        
        features_row = {"GUID": guid}
        
        results = Parallel(n_jobs=n_jobs_roi)(
            delayed(process_roi)(roi_name, roi_val, seg_data, brain_vol)
            for roi_name, roi_val in roi_labels.items()
        )
            
        # Combine results into features_row
        features_row = {"GUID": guid}  # or existing dict if already has GUID
        for roi_name, feats in results:
            for k, v in feats.items():
                features_row[f"{roi_name}_{k}"] = v
        
        # for roi_name, roi_val in roi_labels.items():
        #     mask = (seg_data == roi_val)
        #     feats = extract_region_features(mask, brain_vol)
        #     for k, v in feats.items():
        #         features_row[f"{roi_name}_{k}"] = v
        
        all_rows.append(features_row)
        
        # Save to CSV
        if i % 10 == 0 or i == len(segmentation_csv) - 1:
            df_features = pd.DataFrame(all_rows)
            df_features.to_csv(output_file, index=False)
            logging.info(f"✅ Morphometry features saved to {output_file}")
            
    # Save to CSV
    df_features = pd.DataFrame(all_rows)
    df_features.to_csv(output_file, index=False)
    logging.info(f"✅ Morphometry features saved to {output_file}")

        
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="dev/configs/run_seg_features.py", help='Path to config file (.py) with paths and ROI definitions')
    args = parser.parse_args()
    
    from dev.utils import load_config_from_path
    config = load_config_from_path(args.config)
    
    main(config)
