from utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "json_path": "data/results/whole_brain/eval_cl16/metrics.json",
    "csv_path": "data/results/whole_brain/eval_cl16/metrics_table.csv",
    "metrics": ["mAP@k", "success@k", "evaluated_queries"]
    
})

