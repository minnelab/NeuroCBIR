# config/shared_config.py

config = {
    "data_path": "dev/data_private/mock_dataset",
    "output_dir": "dev/data_private/results/whole_brain/eval_cl16/",
    "dataset_index_file_name": "whole_brain/dataset_index.csv",
    "metadata_file_name": "metadata.csv",
    "batch_size": 32,
    "proj_params": {
        "input_shape": [8, 20, 22, 26],
        "projector_dims": [128],
        "final_dim": 16 # Contrastive learning embedding dimension
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
