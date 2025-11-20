
import pandas as pd
import os
import numpy as np
from preprocessing.load_dataset import list_files_with_extension
from preprocessing.segmentation import get_common_bounding_box, extract_bounding_box
from utils import load_config_from_path
import argparse

def save_bounding_boxes_to_csv(bounding_boxes, common_bounding_boxes, common_extent, output_path):
    rows = []
    for label in bounding_boxes:
        bbox = bounding_boxes[label]
        common_bbox = common_bounding_boxes.get(label)

        if bbox is None:
            row = [label] + [None]*9 + list(common_extent) + [None]*6
        else:
            min_coords, max_coords = bbox
            extent = np.array(max_coords) - np.array(min_coords)
            if common_bbox is not None:
                min_common, max_common = common_bbox
            else:
                min_common = max_common = (None, None, None)
            row = [
                label,
                *min_coords,
                *max_coords,
                *extent,
                *common_extent,
                *min_common,
                *max_common
            ]
        rows.append(row)

    columns = [
        "LabelName", "min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
        "extent_x", "extent_y", "extent_z",
        "common_extent_x", "common_extent_y", "common_extent_z",
        "common_min_x", "common_min_y", "common_min_z",
        "common_max_x", "common_max_y", "common_max_z"
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Saved bounding boxes to: {output_path}")


def main(config):
    seed = config.get("random_state", 42)
    subcortical_indices = config["subcortical_indices"]
    input_shape = config["input_shape"]
    output_csv = config.get("output_csv", "bounding_boxes.csv")

    bounding_boxes = {}
    combined_segs = {key: np.zeros(input_shape, dtype=np.uint16) for key in subcortical_indices}

    file_paths, file_names = list_files_with_extension(config["load_path"], config["extension"])
    file_paths, file_names = file_paths[:config["n_batches"]], file_names[:config["n_batches"]]

    for file_path, file_name in zip(file_paths, file_names):
        file_to_load = os.path.join(config["load_path"], file_path, file_name)
        print(f"Processing {file_to_load}")
        data = np.load(file_to_load)
        segmentations = data["segmentations"]

        for key, value in subcortical_indices.items():
            for i in range(segmentations.shape[0]):
                print(f"Accumulating {key} ({i+1}/{len(segmentations)})...", end='\r')
                segmentation_mask = (segmentations[i] == value).astype(np.uint16)
                combined_segs[key] += segmentation_mask
    print()

    max_extent = np.array([0, 0, 0])
    for key in subcortical_indices:
        print(f"Extracting bounding box for: {key}")
        bbox = extract_bounding_box(combined_segs[key])
        bounding_boxes[key] = bbox
        if bbox:
            extent = np.array(bbox[1]) - np.array(bbox[0])
            max_extent = np.maximum(max_extent, extent)

    initial_common_extent = max_extent + 4  # Add margin of 2 voxels on each side
    common_extent = np.ceil(initial_common_extent / 16) * 16 # Round up to nearest multiple of 16
    common_extent = common_extent.astype(int)

    common_bounding_boxes = {}
    for key, value in subcortical_indices.items():
        original_bbox = bounding_boxes[key]
        common_bounding_boxes[key] = get_common_bounding_box(original_bbox, common_extent)

    print("Common bounding box extent (with padding, rounded):", common_extent)

    save_bounding_boxes_to_csv(bounding_boxes, common_bounding_boxes, common_extent, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)
    main(config)
