import os
import json
import numpy as np
import pandas as pd
import argparse
import logging

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
    clinical_ds["GUID"] = clinical_ds["GUID"].astype(str)

    # Merge on GUID
    dataset = pd.merge(clinical_ds, df_embs, on="GUID", how="inner")
    
    # Filter out rows    
    dataset = dataset.query("useable == 1").reset_index(drop=True)
    dataset = dataset.query("mislabel == 0").reset_index(drop=True)
    dataset = dataset.query("valid_seg == 1").reset_index(drop=True)
    dataset['subject'].replace('', pd.NA, inplace=True)
    dataset = dataset.dropna(subset=['subject']).reset_index(drop=True)

    # Convert embedding columns into a single 'features' column of vectors
    embedding_cols = [col for col in df_embs.columns if col != "GUID"]
    dataset["features"] = dataset[embedding_cols].apply(lambda row: row.to_numpy(), axis=1)

    # Compute retrieval
    retrieval_path = os.path.join(config["output_dir"], "retrieval.csv")
    if not os.path.isfile(retrieval_path):
        logging.info("Computing retrieved cases...")
        top_k_max = max(config["top_k_values"])
        retrieval_df = get_topk_guid_retrievals(dataset, top_k=top_k_max)
        logging.info("✓ Done: retrieved cases.")

        # Save retrieval dataframe
        retrieval_df.to_csv(retrieval_path, index=False)
    else:
        logging.info("Loading retrieved cases...")
        retrieval_df = pd.read_csv(retrieval_path)

    # Evaluate metrics and bias
    all_metrics = {"standard": {}, "bias": {}}

    logging.info("Starting evaluation of retrieval metrics...")
    for k in config["top_k_values"]:
        logging.info(f"\nEvaluating top-{k} metrics...")

        # Standard retrieval metric
        logging.info("  → Computing standard retrieval metrics...")
        all_metrics["standard"][f"top_{k}"] = evaluate_guid_retrieval_map(
            retrieval_df, clinical_ds, top_k=k, class_column=config["class_column"]
        )
        logging.info(f"    ✓ Done with top-{k} standard metrics.")

        # Bias metrics per specified column
        all_metrics["bias"][f"top_{k}"] = {}
        for col in config["bias_columns"]:
            logging.info(f"  → Evaluating bias by column: '{col}'...")
            all_metrics["bias"][f"top_{k}"][col] = evaluate_bias_by_column(
                retrieval_df, clinical_ds, top_k=k, class_column=config["class_column"], group_by_column=col
            )
            logging.info(f"    ✓ Done with bias evaluation for column '{col}' at top-{k}.")
    logging.info("\nAll evaluations complete.")

    # Save metrics to JSON
    metrics_path = os.path.join(config["output_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    logging.info(f"✅ Evaluation complete. Results saved to: {config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)