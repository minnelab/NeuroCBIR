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
        "disease",
        "field_strength",
        "manufacturer",
        "model_name"
    ]
})

