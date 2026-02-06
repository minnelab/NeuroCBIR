import json
import pandas as pd
from dev.utils import load_config_from_path
import argparse
from dev.scripts.whole_brain.metrics_json_to_csv import _fill_partition, _fill_project

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
            for metric in config["metrics"]:
                row = {
                    "metric": metric,
                    "top": top,
                    "structure": structure,
                    "all": values[metric],
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

                bias_top_data = eval_data.get("bias", {}).get(top, {})

                _fill_partition(row, bias_top_data, metric)
                _fill_project(row, bias_top_data, metric)

                rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Convert numeric columns from [0,1] → percentage
    if config.get("as_percentage", True):
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = (df[num_cols] * 100)
    
    # Round numeric columns to n_decimals decimal places
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].round(config.get("n_decimals", 4))

    # Save to CSV
    df.to_csv(config["csv_path"], index=False)

    print("Saved metrics_table.csv with shape:", df.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)