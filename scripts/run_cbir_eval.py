import os
import json
import numpy as np
import pandas as pd
import argparse
from cbir.evaluation import get_topk_guid_retrievals, evaluate_guid_retrieval, evaluate_bias_by_column

def main(config):

    os.makedirs(config["output_dir"], exist_ok=True)

    # Load and filter metadata
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file"]))
    if config.get("project_filter"):
        clinical_ds = clinical_ds.query(f"project == '{config['project_filter']}'").reset_index(drop=True)

    # Create or load features
    n_feats = config["features_dim"]
    n_scans = len(clinical_ds)
    emb_ds = np.random.uniform(-1, 1, (n_scans, n_feats))

    dataset = clinical_ds.copy()
    dataset["features"] = [x for x in emb_ds]

    # Compute retrieval
    top_k_max = max(config["top_k_values"])
    retrieval_df = get_topk_guid_retrievals(dataset, top_k=top_k_max)

    # Save retrieval dataframe
    retrieval_path = os.path.join(config["output_dir"], "retrieval.csv")
    retrieval_df.to_csv(retrieval_path, index=False)

    # Evaluate metrics and bias
    all_metrics = {"standard": {}, "bias": {}}
    for k in config["top_k_values"]:
        all_metrics["standard"][f"top_{k}"] = evaluate_guid_retrieval(
            retrieval_df, clinical_ds, top_k=k, class_column=config["class_column"]
        )
        all_metrics["bias"][f"top_{k}"] = {}
        for col in config["bias_columns"]:
            all_metrics["bias"][f"top_{k}"][col] = evaluate_bias_by_column(
                retrieval_df, clinical_ds, top_k=k, class_column=config["class_column"], group_by_column=col
            )

    # Save metrics to JSON
    metrics_path = os.path.join(config["output_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"✅ Evaluation complete. Results saved to: {config['output_dir']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    main(config)
