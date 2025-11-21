# config/train_contrastive_config.py

from dev.utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "resume_path": "",
    "base_logging_path": "dev/data_private/mock_dataset/logs/",
    "partition": "train",
    "num_epochs": 500,
    "batch_size": 32,
    "group_size": 3,
    "groups_per_batch": 4,
    "n_batches_per_file": 300,
    "lr": 0.0001,
})

