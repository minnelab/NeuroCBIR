# config/shared_config.py

config = {
    "data_path": "dev/data_private/mock_dataset",
    "output_dir": "dev/data_private/results/region_brain/eval_cl32/",
    "dataset_index_file_name": "region_brain/dataset_index.csv",
    "metadata_file_name": "metadata.csv",
    "labels_path": "dev/data_private/mock_dataset/labels.csv",
    "bb_path": "dev/data_private/mock_dataset/bounding_boxes.csv",
    "proj_params": {
        "input_shape": [8, 8, 8, 8],
        "projector_dims": [128],
        "final_dim": 32 # Dimension of the final embedding
    },
    "encoder_params": {
        "spatial_dims": 3,
        "in_channels": 8,
        "channels": [64, 128, 256],
        "out_channels": 256,
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 8,
        "norm_eps": 1e-5,
        "attention_levels": [False, False, False],
        "with_nonlocal_attn": False,
        "include_fc": False,
    },
}
