import json
import pandas as pd
from dev.utils import load_config_from_path
import argparse


def _fill_partition(row, bias_top_data, metric):
    """
    Fill train / test columns from any known partition schema.
    """
    partition_data = bias_top_data.get("partition", None)

    if isinstance(partition_data, dict):
        # {"train": {...}, "test": {...}}
        for part_name, part_vals in partition_data.items():
            if isinstance(part_vals, dict):
                row[part_name] = part_vals.get(metric)

    elif isinstance(partition_data, list):
        for p in partition_data:
            if isinstance(p, dict):
                # [{"partition": "train", ...}]
                row[p.get("partition")] = p.get(metric)
            elif isinstance(p, str):
                # ["train", "test"]
                row[p] = None


def _fill_project(row, bias_top_data, metric):
    """
    Fill ADNI / OASIS / AIBL / ... columns from any known project schema.
    """
    project_data = bias_top_data.get("project", None)

    if isinstance(project_data, dict):
        for proj_name, proj_vals in project_data.items():
            if not isinstance(proj_vals, dict):
                continue
            proj_name = proj_name.upper()
            if proj_name == "OASIS3":
                proj_name = "OASIS"
            row[proj_name] = proj_vals.get(metric)

    elif isinstance(project_data, list):
        for p in project_data:
            if isinstance(p, dict):
                proj_name = p.get("project", "").upper()
                if proj_name == "OASIS3":
                    proj_name = "OASIS"
                row[proj_name] = p.get(metric)
            elif isinstance(p, str):
                row[p.upper()] = None


def main(config):

    # Load JSON
    with open(config["json_path"], "r") as f:
        data = json.load(f)

    rows = []

    # Iterate over standard metrics
    for top, values in data.get("standard", {}).items():
        for metric in config["metrics"]:

            row = {
                "metric": metric,
                "top": top,
                "all": values.get(metric),

                # partition columns
                "train": None,
                "test": None,

                # project columns
                "ADNI": None,
                "OASIS": None,
                "AIBL": None,
                "MIRIAD": None,
                "SLIM": None,
            }

            bias_top_data = data.get("bias", {}).get(top, {})

            _fill_partition(row, bias_top_data, metric)
            _fill_project(row, bias_top_data, metric)

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Convert numeric columns from [0,1] → percentage
    if config.get("as_percentage", True):
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = (df[num_cols] * 100)
    
    # Round numeric columns to n_decimals decimal places
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(config.get("n_decimals", 4))

    # Save CSV
    df.to_csv(config["csv_path"], index=False)

    print(f"Saved CSV: {config['csv_path']}")
    print("Shape:", df.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config .py file"
    )
    args = parser.parse_args()

    config = load_config_from_path(args.config)
    main(config)
