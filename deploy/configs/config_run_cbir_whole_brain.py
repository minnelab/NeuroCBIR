config = {
    "seed": 42,
    "data_path": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/original/",
    "save_path": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/whole_brain/",
    "extension": ".npz",
    "metadata_file": "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/batched_datasets/combined_metadata.csv",
    "ckpt_path": "/cephyr/users/felixnie/Alvis/logs/20250703_094042/checkpoint-epoch-8.pth",
    "ckpt_key": "autoencoder_state_dict",
    "use_old_state_dict": False,
    "vae_params": {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [
            64,
            128,
            128,
            128
        ],
        "latent_channels": 8,
        "num_res_blocks": 2,
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
        "attention_levels": [
            False,
            False,
            False,
            False
        ],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "include_fc": False
    }
}