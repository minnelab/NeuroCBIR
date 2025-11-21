config = {
    "seed": 42,
    "device": "cpu",  # "cuda" or "cpu"
    "data_path": "data/mock_dataset/original/",
    "save_path": "data/mock_dataset/whole_brain/",
    "extension": ".npz",
    "metadata_file": "data/mock_dataset/metadata.csv",
    "ckpt_path": "data/mock_dataset/logs/vae_whole_brain/checkpoint-epoch-0.pth",
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