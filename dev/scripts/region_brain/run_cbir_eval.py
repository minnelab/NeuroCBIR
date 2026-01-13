import os
import json
import numpy as np
import pandas as pd
import argparse
from dev.cbir.evaluation import get_topk_guid_retrievals, evaluate_guid_retrieval_map, evaluate_bias_by_column
from dev.utils import load_config_from_path

def main(config):

    # os.makedirs(config["output_dir"], exist_ok=True)

    # Load and filter metadata
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file_name"]))
    if config.get("project_filter"):
        clinical_ds = clinical_ds.query(f"project == '{config['project_filter']}'").reset_index(drop=True)

    # Load real features from parquet
    emb_path = os.path.join(config["output_dir"], config["embedding_file"])  # e.g., "outputs/embeddings.parquet"
    df_embs = pd.read_parquet(emb_path)

    # Ensure GUID is string and joinable
    df_embs["GUID"] = df_embs["GUID"].astype(str)
    df_embs["LabelName"] = df_embs["LabelName"].astype(str)
    clinical_ds["GUID"] = clinical_ds["GUID"].astype(str)

    # Merge on GUID
    dataset = pd.merge(clinical_ds, df_embs, on="GUID", how="inner")

    # Filter out rows    
    dataset = dataset.query("useable == 1").reset_index(drop=True)
    dataset = dataset.query("mislabel == 0").reset_index(drop=True)
    dataset['subject'].replace('', pd.NA, inplace=True)
    dataset = dataset.dropna(subset=['subject']).reset_index(drop=True)

    # Convert embedding columns into a single 'features' column of vectors
    embedding_cols = [col for col in df_embs.columns if not col in  ["GUID", "LabelName"]]
    dataset["features"] = dataset[embedding_cols].apply(lambda row: row.to_numpy(), axis=1)

    all_metrics = {}
    combined_retrievals = []

    struct_names = config["struct_names"]
    if struct_names is None:
        struct_names = dataset["LabelName"].unique()

    # Loading previous computed structures in the .json   
    metrics_path = os.path.join(config["output_dir"], "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r") as f:
            all_metrics = json.load(f)

    # Loading previuos computed structures in the .csv file
    cr_struct_names = []
    retrieval_path = os.path.join(config["output_dir"], "retrieval_all.csv")
    if os.path.isfile(retrieval_path):
        combined_retrievals.append(pd.read_csv(retrieval_path))
        cr_struct_names = list(combined_retrievals[0]["LabelName"].unique())

    for struct_name in struct_names:
        if (struct_name in all_metrics) and (struct_name in cr_struct_names):
            print(f"Skipping. Struct already computed in .json file: {struct_name}")
            continue

        print(f"Processing: {struct_name}")

        subset = dataset.query(f"LabelName == '{struct_name}'").reset_index(drop=True)

        # Compute retrieval
        print("Computing retrieved cases...")
        top_k_max = max(config["top_k_values"])
        retrieval_df = get_topk_guid_retrievals(subset, top_k=top_k_max)
        print("✓ Done: retrieved cases.")

        # Add LabelName column for tracking
        retrieval_df["LabelName"] = struct_name
        combined_retrievals.append(retrieval_df)

        # Check again if the struct eval metric was computed
        if (struct_name in all_metrics):
            print(f"Skipping. Struct already computed in .json file: {struct_name}")
            continue

        # Evaluate metrics and bias
        all_metrics[struct_name] = {"standard": {}, "bias": {}}

        print("Starting evaluation of retrieval metrics...")
        for k in config["top_k_values"]:
            print(f"\nEvaluating top-{k} metrics...")

            # Standard retrieval metric
            print("  → Computing standard retrieval metrics...")
            all_metrics[struct_name]["standard"][f"top_{k}"] = evaluate_guid_retrieval_map(
                retrieval_df, clinical_ds, top_k=k, class_column=config["class_column"]
            )
            print(f"    ✓ Done with top-{k} standard metrics.")

            # Bias metrics per specified column
            all_metrics[struct_name]["bias"][f"top_{k}"] = {}
            for col in config["bias_columns"]:
                print(f"  → Evaluating bias by column: '{col}'...")
                all_metrics[struct_name]["bias"][f"top_{k}"][col] = evaluate_bias_by_column(
                    retrieval_df, clinical_ds, top_k=k, class_column=config["class_column"], group_by_column=col
                )
                print(f"    ✓ Done with bias evaluation for column '{col}' at top-{k}.")

        print("\nAll evaluations complete.")

        # Save metrics to JSON
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=4)

        print(f"✅ Evaluation complete. Results saved to: {config['output_dir']}")
    
    # After the loop — save the combined retrievals
    pd.concat(combined_retrievals, ignore_index=True).to_csv(retrieval_path, index=False)

    # Save metrics to JSON
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"✅ Evaluation complete. Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)