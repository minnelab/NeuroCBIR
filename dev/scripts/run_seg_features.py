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

# --- Main processing ---
def main(config):
    data_path = config["data_path"]
    segmentation_csv = pd.read_csv(os.path.join(data_path, config["metadata_file_name"]))  # Must contain GUID, file paths
    output_file = os.path.join(config["output_dir"], "morphometry_features.csv")
    
    all_rows = []

    for idx, row in tqdm(segmentation_csv.iterrows(), total=len(segmentation_csv), desc="Processing subjects"):
        guid = row['GUID']
        seg_file = os.path.join(data_path, row['seg'])  # aparc+aseg NIfTI path
        
        # Load segmentation
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata().astype(int)
        
        # Estimate brain volume: sum of all non-zero voxels (approx)
        brain_vol = np.sum(seg_data > 0)
        
        # Define ROI labels you want to extract features from
        roi_labels = config["roi_labels"]  # dict {roi_name: label_value}
        
        features_row = {"GUID": guid}
        
        for roi_name, roi_val in roi_labels.items():
            mask = (seg_data == roi_val)
            feats = extract_region_features(mask, brain_vol)
            for k, v in feats.items():
                features_row[f"{roi_name}_{k}"] = v
        
        all_rows.append(features_row)
    
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
