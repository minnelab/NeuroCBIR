# config/cl_embedding_config.py

from dev.utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "resume_path": "data/mock_dataset/logs/cl_region_brain/checkpoint.pth",
    "n_structs": -1,
    "batch_size": 256
})
