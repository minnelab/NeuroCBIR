# config/train_contrastive_config.py

from dev.utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "resume_path": "",
    "base_logging_path": "dev/data_private/mock_dataset/logs/",
    "partition": "train",
    "num_epochs": 100000,
    "batch_size": 512,
    "group_size": 3,
    "groups_per_batch": 160,
    "group_key": ["subject", "struct_name"],
    "n_file_loaded": 10,
    "n_batches_per_file": 1000,
    "lr": 0.0001,
    "loss_params": {
        "lambda_sn": 0.2,
        "lambda_hn": 0.1,
        "alpha_sn": 0.2,
        "alpha_hn": 0.2,
        "lower_perc": 0.05,
        "upper_perc": 0.95,
    }
})

