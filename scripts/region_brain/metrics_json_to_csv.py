import json
import pandas as pd
from utils import load_config_from_path
import argparse

def main(config):
    # Load JSON file
    with open(config["json_path"], "r") as f:
        data = json.load(f)

    rows = []
    structures={}

    if config["struct_names"] is not None:
        for struct_name in config["struct_names"]:
            structures[struct_name] = data[struct_name]
    else:
        structures = data


    for structure, eval_data in structures.items():
        # Loop over standard metrics (all data)
        for top, values in eval_data.get("standard", {}).items():
            row = {
                "structure": structure,
                "top": top,
                "all": values["precision@k"],
                "train": None,
                "test": None,
                "ADNI": None,
                "OASIS": None,
                "AIBL": None,
                "MIRIAD": None,
                "SLIM": None,
            }

            # Add bias partition info (train/test)
            for p in eval_data.get("bias", {}).get(top, {}).get("partition", []):
                row[p["partition"]] = p["precision@k"]

            # Add bias project info
            for proj in eval_data.get("bias", {}).get(top, {}).get("project", []):
                proj_name = proj["project"].upper()
                if proj_name == "OASIS3":
                    proj_name = "OASIS"
                row[proj_name] = proj["precision@k"]

            rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Convert all numeric columns from 0–1 to percentage (0–100)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = (df[num_cols] * 100).round(1)

    # Save to CSV
    df.to_csv(config["csv_path"], index=False)

    print("Saved metrics_table.csv with shape:", df.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)