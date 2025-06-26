import numpy as np
import pandas as pd
from pathlib import Path

def create_lookup_table(root_dirs, output_csv="dataset_index.csv"):
    rows = []
    j = 0
    for dataset_name, root_dir in root_dirs.items():
        for batch_file in sorted(Path(root_dir).glob("*.npz")):
            with np.load(batch_file) as data:
                ids = data["ids"]
                for i, sample_id in enumerate(ids):
                    rows.append({
                        "id": sample_id,
                        "dataset": dataset_name,
                        "batch_file": str(batch_file.resolve()),
                        "index_in_batch": i,
                        "index_global": j,
                    })
                    j += 1

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Lookup table saved to {output_csv}")

if __name__ == "__main__":
    # Example usage
    create_lookup_table({
        "OASIS": "/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/batched_OASIS3",
        "ADNI": "/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/batched_adni"
    },
    output_csv="/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/dataset_index.csv")
