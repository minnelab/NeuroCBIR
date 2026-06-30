import os
import json
import numpy as np
import pandas as pd
import argparse
import logging
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# from dev.cbir.evaluation import retrieve_topk_for_queries
from dev.utils import load_config_from_path

def spearman_per_query(df):

    if len(df) < 2:
        return np.nan
    return spearmanr(df["rank"], df["ms_ssim"] ).correlation

def retrieve_topk_for_queries(
    dataset: pd.DataFrame,
    queries: pd.DataFrame,
    top_k: int = 3,
    feature_column: str = "features",
    guid_column: str = "GUID"
) -> pd.DataFrame:
    """
    Retrieve the top-k most similar entries for a subset of queries, 
    using cosine similarity against the full dataset as the retrieval pool.

    Args:
        dataset (pd.DataFrame): Full pool of entries with features and GUIDs.
        queries (pd.DataFrame): Subset of rows from dataset to use as queries.
        top_k (int): Number of top similar entries to retrieve.
        feature_column (str): Column containing the feature vectors.
        guid_column (str): Column with unique scan identifiers (e.g., 'GUID').

    Returns:
        pd.DataFrame: Retrieval results. One row per query, first column is the query GUID,
                      followed by the GUIDs of the top-k retrieved entries.
    """
    # Retrieval pool
    features_matrix = np.stack(dataset[feature_column].values)
    guids = dataset[guid_column].values

    # Queries
    query_features = np.stack(queries[feature_column].values)
    query_guids = queries[guid_column].values

    retrievals = []
    n_col_to_rm = 0
    for i in tqdm(range(len(queries)), desc="Retrieving"):
        similarities = cosine_similarity(query_features[i].reshape(1, -1), features_matrix)[0]
        
        # Exclude self if query is in the dataset and same subject
        subject_mask = (dataset["subject"] == queries.iloc[i]["subject"]).values
        similarities[subject_mask] = -1  # Zero out similarities for same subject

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        row = [query_guids[i]] + guids[top_k_indices].tolist()
        retrievals.append(row)
        n_col_to_rm += np.sum(subject_mask)
        
    col_names = ["query"] + [f"top{i+1}" for i in range(top_k)]
    out = pd.DataFrame(retrievals, columns=col_names)
    
    # Remove last n_col_to_rm columns if they correspond to same-subject entries
    if n_col_to_rm > 0:
        out = out.iloc[:, :-n_col_to_rm]
    return out

def summarize_spearman_results(spearman_df):
        return {
            "mean": spearman_df["spearman_r"].mean(),
            "median": spearman_df["spearman_r"].median(),
            "std": spearman_df["spearman_r"].std(),
            "q1": spearman_df["spearman_r"].quantile(0.25),
            "q3": spearman_df["spearman_r"].quantile(0.75),
            "min": spearman_df["spearman_r"].min(),
            "max": spearman_df["spearman_r"].max(),
            "evaluated_queries": len(spearman_df),
        }

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

    # Load csv with precomputed MS-SSIM scores
    msssim_path = os.path.join(config["data_path"], config["msssim_file_name"])
    msssim_df = pd.read_csv(msssim_path)
    msssim_df["query_guid"] = msssim_df["query_guid"].astype(str)   
    msssim_df["reference_guid"] = msssim_df["reference_guid"].astype(str)   
    queries_guid_list = msssim_df["query_guid"].unique().tolist()
    dataset_queries = dataset[dataset["GUID"].isin(queries_guid_list)].reset_index(drop=True)
    logging.info(f"Number of queries per project: {dataset_queries['project'].value_counts().to_dict()}")
    logging.info(f"Number of queries per partition: {dataset_queries['partition'].value_counts().to_dict()}")

    # Compute retrieval
    logging.info("Computing retrieved cases...")
    retrieval_df = retrieve_topk_for_queries(dataset, dataset_queries, top_k=len(dataset))
    logging.info("✓ Done: retrieved cases.")
    
    retrieval_long = (
    retrieval_df
    .melt(
        id_vars="query",
        var_name="rank_col",
        value_name="reference_guid"
    )
    )

    # Extract numeric rank from "topK"
    retrieval_long["rank"] = (
        retrieval_long["rank_col"]
        .str.replace("top", "", regex=False)
        .astype(int)
    )

    retrieval_long = retrieval_long.drop(columns="rank_col")
    
    # Merge with MS-SSIM scores
    merged = (
    msssim_df
    .merge(
        retrieval_long,
        left_on=["query_guid", "reference_guid"],
        right_on=["query", "reference_guid"],
        how="inner"
    )
    )
    logging.debug(f"Merged df columns:\n{merged.columns}")
    logging.debug(f"Merged df head:\n{merged.head()}")
    
    # Compute spearman correlation per query
    spearman_df = (
    merged
    .groupby("query_guid")
    .apply(spearman_per_query)
    .reset_index(name="spearman_r")
    )
    spearman_df = pd.merge(spearman_df, dataset[["GUID"] + config["bias_columns"]], left_on="query_guid", right_on="GUID", how="left")
    logging.info(f"Computed spearman correlation for {len(spearman_df)} queries.")
    logging.debug(f"Spearman correlation head:\n{spearman_df.head()}")
    logging.debug(f"Spearman correlation describe:\n{spearman_df.describe()}")
    
    # Evaluate metrics and bias
    all_metrics = {"standard": {}, "bias": {}}
    logging.info("Starting evaluation of spearman correlation metrics...")
    # Standard retrieval metric
    logging.info("  → Computing standard spearman correlation metrics...")
    all_metrics["standard"]["spearman_r"] = summarize_spearman_results(spearman_df)
    
    # Bias metrics per specified column
    all_metrics["bias"]["spearman_r"] = {}
    for col in config["bias_columns"]:
        logging.info(f"  → Evaluating bias by column: '{col}'...")
        grouped = spearman_df.groupby(col)
        all_metrics["bias"]["spearman_r"][col] = {}
        for group_name, group_df in grouped:
            all_metrics["bias"]["spearman_r"][col][group_name] = summarize_spearman_results(group_df)
        logging.info(f"    ✓ Done with bias evaluation for column: '{col}'.")
    logging.info("\nAll evaluations complete.")
    
    # Save metrics to JSON
    metrics_path = os.path.join(config["output_dir"], "spearman_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)
    
    logging.info(f"Spearman correlation metrics saved to {metrics_path}")
    
    # Optional: Generate debug plots
    if logging.getLogger().level == logging.DEBUG:
    
        import matplotlib.pyplot as plt
        plt.set_loglevel("info") 

        plt.hist(spearman_df["spearman_r"].dropna(), bins=30)
        plt.xlabel("Spearman ρ")
        plt.ylabel("Number of queries")
        plt.title("Embedding vs MS-SSIM rank correlation")
        plt.savefig(os.path.join("./tmp/figures", "spearman_correlation_histogram.png"))

        # plot tendency df["ms_ssim"], -df["rank"]
        for _, row in tqdm(merged.groupby("query_guid")):
            plt.figure(figsize=(15,5))
            plt.scatter(row["rank"], row["ms_ssim"], s=0.5, alpha=0.5)
            plt.xlabel("Negative Rank")
            plt.ylabel("MS-SSIM")
            plt.title(f"Query GUID: {row['query_guid'].iloc[0]}")
            plt.savefig(os.path.join("./tmp/figures", f"spearman_scatter_{row['query_guid'].iloc[0]}.png"))
            plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # DEBUG or INFO
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config .py file')
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    main(config)