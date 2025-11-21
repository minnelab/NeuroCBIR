# config/cl_embedding_config.py

from utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "resume_path": "/cephyr/users/felixnie/Alvis/NeuroCBIR/data/results/whole_brain/eval_cl16/20250709_215303-checkpoint-epoch-195.pth",
})
