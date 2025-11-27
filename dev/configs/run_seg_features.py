config = {
    "logging_level": "INFO", # Options: DEBUG, INFO, WARNING, ERROR
    "parallel_by": "subject",  # Options: "roi" or "subject"
    "data_path": "/mnt/kth_cbh/fenda/Datasets/",
    "metadata_file_name": "/mnt/kth_cbh/fenda/Datasets/ADNI/metadata.csv",   # contains columns: GUID, segmentation_file
    "output_dir": "/mnt/kth_cbh/fenda/Datasets/ADNI",
    # "metadata_file_name": "/mnt/kth_cbh/fenda/Datasets/OASIS3/metadata.csv",   # contains columns: GUID, segmentation_file
    # "output_dir": "/mnt/kth_cbh/fenda/Datasets/OASIS3",
    "n_jobs": 10,
    "checkpoint_every": 20,
    "voxel_volume_mm3": 1,
    "labels_path": "deploy/data/labels.csv",
    "bb_path": "deploy/data/bounding_boxes.csv",
    
#     "roi_labels": 
#         # None,  # If None, will be loaded from labels_path        
#         {
#         "Left-Hippocampus": 17,
#         "Right-Hippocampus": 53,
#         "Left-Amygdala": 18,
#         "Right-Amygdala": 54,
#         "Left-Thalamus": 10,
#         "Right-Thalamus": 49,
#         "Left-Lateral-Ventricle": 4,
#         "Right-Lateral-Ventricle": 43,
#         "ctx-lh-fusiform": 106,
#         "ctx-rh-fusiform": 142,
#     # Add all 103 ROI labels here
# }
}
