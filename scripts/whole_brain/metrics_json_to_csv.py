import json
import pandas as pd

# Load JSON file
with open("data/results/whole_brain/eval_cl32/metrics.json", "r") as f:
    data = json.load(f)

rows = []

# Loop over standard metrics (all data)
for top, values in data.get("standard", {}).items():
    row = {
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
    for p in data.get("bias", {}).get(top, {}).get("partition", []):
        row[p["partition"].capitalize()] = p["precision@k"]

    # Add bias project info
    for proj in data.get("bias", {}).get(top, {}).get("project", []):
        proj_name = proj["project"].upper()
        if proj_name == "OASIS3":
            proj_name = "OASIS"
        row[proj_name] = proj["precision@k"]

    rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv("data/results/whole_brain/eval_cl32/metrics_table.csv", index=False)

print("Saved metrics_table.csv with shape:", df.shape)
