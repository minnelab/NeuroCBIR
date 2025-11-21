# config file

config = {
    "random_state": 1234,
    "device": "cpu",  # "cuda" or "cpu"
    "data_path": "data/mock_dataset/original/",
    "metadata_file": "data/mock_dataset/original/metadata.csv",
    "base_logging_path": "data/mock_dataset/logs/",
    "labels_path": "data/mock_dataset/labels.csv",
    "bb_path": "data/mock_dataset/bounding_boxes.csv",
    "resume_path": "", # e.g., "data/mock_dataset/logs/autoencoder/checkpoint_final.pth"
    "num_epochs": 10000,
    "max_batch_size": 8, # <-- must be divisible by "n_structs"
    "batch_size": 24, # <-- must be divisible by "max_batch_size"
    "n_structs": 4,
    "n_batches_per_file": 800,
    "lr": 1e-4,
    "adv_weight": 0.1,
    "perceptual_weight": 0.1,
    "kl_weight": 1e-7,
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
    "dis_params": {
        "spatial_dims": 3,
        "num_layers_d": 3,
        "channels": 32,
        "in_channels": 1,
        "out_channels": 1,
        "norm": "INSTANCE",
    },
}

assert config["batch_size"] % config["max_batch_size"] == 0
assert config["max_batch_size"] % config["n_structs"] == 0
