from dev.utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "embedding_file": "projected_embeddings.parquet",
    "project_filter": "",
    "class_column": "subject",
    "top_k_values": [
        1,
        5
    ],
    "bias_columns": [
        "partition",
        "project",
        "field_strength",
        "manufacturer",
        "model_name",
    ],
    "struct_names": None, # <-- Set to None to compute all struct names
    # "struct_names": [ # <-- Set to None to compute all struct names
    #     "Left-Hippocampus",
    #     "Left-Thalamus",
    #     "Left-Amygdala",
    #     "Left-Lateral-Ventricle",
    #     "Right-Hippocampus",
    #     "Right-Thalamus",
    #     "Right-Amygdala",
    #     "Right-Lateral-Ventricle",
    # ]
})

