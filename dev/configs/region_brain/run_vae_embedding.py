# config file

config = {
    "random_state": 1234,
    "data_path": "data/mock_dataset/original/",
    "save_path": "data/mock_dataset/region_brain/",
    "metadata_file": "data/mock_dataset/metadata.csv", 
    "labels_path": "data/mock_dataset/labels.csv",
    "bb_path": "data/mock_dataset/bounding_boxes.csv",
    "ckpt_path": "data/mock_dataset/logs/vae_region_brain/checkpoint-epoch-0.pth",
    "use_old_state_dict": False,
    "n_structs": -1,
    "vae_params": {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 8,
        "channels": [64, 128, 128, 128],
        "num_res_blocks": 2,
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
        "attention_levels": [False, False, False, False],
        "with_decoder_nonlocal_attn": False,
        "with_encoder_nonlocal_attn": False,
    },
    "batch_size": 64,
}

