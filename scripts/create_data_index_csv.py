import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def create_lookup_table(root_dirs, output_csv, id_key):
    rows = []
    global_index = 0

    for dataset_name, root_dir in root_dirs.items():
        for batch_file in sorted(Path(root_dir).glob("*.npz")):
            with np.load(batch_file) as data:
                ids = np.unique(data[id_key])
                for i, sample_id in enumerate(ids):
                    rows.append({
                        "GUID": sample_id,
                        "dataset": dataset_name,
                        "batch_file": str(batch_file.resolve()),
                        "index_in_batch": i,
                        "index_global": global_index,
                    })
                    global_index += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Lookup table saved to {output_csv}")

def main(config):
    create_lookup_table(config["datasets"], config["output_csv"], config["id_key"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the JSON config file')
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)

