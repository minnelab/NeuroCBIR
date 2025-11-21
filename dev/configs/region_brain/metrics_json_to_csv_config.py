from utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "json_path": "data/results/region_brain/eval_cl32/metrics.json",
    "csv_path": "data/results/region_brain/eval_cl32/metrics_table.csv",
    "struct_names": [ # <-- Set to None to compute all struct names
        "Left-Hippocampus",
        "Left-Thalamus",
        "Left-Amygdala",
        "Left-Lateral-Ventricle",
    ],
    "metrics": ["mAP@k", "success@k", "evaluated_queries"]
    
})

