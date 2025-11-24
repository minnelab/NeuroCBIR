from utils import load_config_from_path
import os

config = load_config_from_path(os.path.join(os.path.dirname(__file__), "shared_config.py"))

config.update({
    "json_path": os.path.join(config["output_dir"], "metrics.json"),
    "csv_path": os.path.join(config["output_dir"], "metrics_table.csv"),
    "metrics": ["precision@k", "success@k", "evaluated_queries"]
    
})

