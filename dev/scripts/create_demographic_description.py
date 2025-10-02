import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

# Paths
input_csv = "/mimer/NOBACKUP/groups/biomedicalimaging-kth/felixnie/combined_metadata.csv"  # Replace with actual filename
output_dir = "./data/demographic_description/"
os.makedirs(output_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(input_csv)
df = df.drop_duplicates()

# Demographic summary
summary = {}

# Age statistics
age_stats = df["age"].describe().to_dict()
summary["age"] = {
    "mean": round(age_stats["mean"], 2),
    "std": round(age_stats["std"], 2),
    "min": round(age_stats["min"], 2),
    "max": round(age_stats["max"], 2),
}

# Disease distribution
summary["disease_distribution"] = df["disease"].value_counts().to_dict()

# Partition distribution
summary["partition_distribution"] = df["partition"].value_counts().to_dict()

# Field strength
summary["field_strength_distribution"] = df["field_strength"].value_counts().to_dict()

# Manufacturer
summary["manufacturer_distribution"] = df["manufacturer"].fillna("Unknown").value_counts().to_dict()

# Model name
summary["model_name_distribution"] = df["model_name"].fillna("Unknown").value_counts().to_dict()

# Save summary as JSON
with open(os.path.join(output_dir, "demographic_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

# Plotting setup
sns.set(style="whitegrid")

# Age distribution plot
plt.figure(figsize=(6, 4))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution.pdf"))
plt.close()

# Disease distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="disease")
plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "disease_distribution.pdf"))
plt.close()

# Partition distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="partition")
plt.title("Partition Distribution")
plt.xlabel("Partition")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "partition_distribution.pdf"))
plt.close()

# Field strength distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="field_strength")
plt.title("Field Strength Distribution")
plt.xlabel("Field Strength (T)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "field_strength_distribution.pdf"))
plt.close()

# Manufacturer distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="manufacturer")
plt.title("Manufacturer Distribution")
plt.xlabel("Manufacturer")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "manufacturer_distribution.pdf"))
plt.close()

# Model name distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="model_name")
plt.title("Model Name Distribution")
plt.xlabel("Model Name")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_name_distribution.pdf"))
plt.close()

print("Demographic description saved in:", output_dir)
