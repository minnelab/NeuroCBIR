# config file

config = {
    "random_state": 1234,
    "data_path": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/",
    "save_path": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/region_brain/",
    "labels_path": "data/labels.csv",
    "bb_path": "data/bounding_boxes.csv",
    "ckpt_path": "/cephyr/users/felixnie/Alvis/logs/20250725_110120/checkpoint-epoch-12.pth",
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

