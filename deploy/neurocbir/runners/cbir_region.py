import argparse
import pandas as pd
from tabulate import tabulate
import torch
import logging
import json
import os

from neurocbir.utils import load_yaml, load_region, retrieve_topk_for_query
from neurocbir.models.q2e_model import build_Q2E  
import neurocbir

logger = logging.getLogger(__name__)

def main(
        img_path: str,
        seg_path: str,
        emb_dataset_path: str,
        labels_path: str,
        bb_path: str,
        region: str,
        vae_params: dict,
        vae_ckpt_path: str,
        cl_params: dict,
        cl_ckpt_path: str,
        o_path: str = None,
        top_k: int = 25,
        device: str = "cpu",
        **kwargs, # Capture unused config keys
        ):
    """
    Runs the Content-Based Image Retrieval (CBIR) pipeline for specific brain regions.

    Args:
        img_path (str): Path to the query brain image.
        seg_path (str): Path to the query segmentation image.
        emb_dataset_path (str): Path to the embedding dataset (Parquet).
        labels_path (str): Path to the CSV file with region labels.
        bb_path (str): Path to the CSV file with region bounding boxes.
        region (str): The specific brain region to query.
        vae_params (dict): Parameters for the VAE model.
        vae_ckpt_path (str): Path to the VAE checkpoint.
        cl_params (dict): Parameters for the contrastive learning model.
        cl_ckpt_path (str): Path to the contrastive learning checkpoint.
        o_path (str, optional): Directory to save the output JSON file. Defaults to None.
        top_k (int, optional): Number of top results to retrieve. Defaults to 25.
        device (str, optional): Device for computation (e.g., "cpu"). Defaults to "cpu".
        **kwargs: Catches any other unused arguments from the config.
    """
    logger.info("CBIR pipeline: region")
    
    # Setup
    q2e_module = build_Q2E(vae_params, vae_ckpt_path, cl_params, cl_ckpt_path, device)
    
    # Load dataset
    # Load real features from parquet
    logger.info("Loading embedding dataset.")
    embs_dataset = pd.read_parquet(emb_dataset_path)
    embs_dataset["LabelName"] = embs_dataset["LabelName"].astype(str)
    embs_dataset["GUID"] = embs_dataset["GUID"].astype(str) # Ensure GUID is string  
    # Convert embedding columns into a single 'features' column of vectors
    embedding_cols = [col for col in embs_dataset.columns if not col in  ["GUID", "LabelName"]]
    embs_dataset["features"] = embs_dataset[embedding_cols].apply(lambda row: row.to_numpy(), axis=1) 
    # Drop the old embedding columns
    embs_dataset = embs_dataset[["GUID", "LabelName", "features"]]
    # Select 
    embs_subset = embs_dataset.query(f"LabelName == '{region}'").reset_index(drop=True)
    
    # Load labels and bounding boxes for cortical/subcortical structures
    labels_df = pd.read_csv(labels_path)
    bb_df = pd.read_csv(bb_path)
    labels_bb_df = pd.merge(labels_df, bb_df, on="LabelName", how="inner") # Merge on the 'GUID' column
    
    # Load whole-brain image query
    i_q = load_region(img_path, seg_path, labels_bb_df, region)
    i_q = torch.from_numpy(i_q).float().unsqueeze(0).unsqueeze(0).to(device)

    # Get features of the query
    logger.info("Running Q2E on query. Returning features.")
    z_q = q2e_module(i_q)
    
    # Top-k retrieval
    logger.info(f"Computing similarities between query {list(z_q.shape)} and dataset {list(embs_dataset.features.values.shape)}.")
    top_k_retrieved = retrieve_topk_for_query(z_q, embs_subset, top_k=top_k)
    
    # Fancy print
    print("\n" + "="*50)
    print(f"Top-{top_k} Retrieval Results ".center(50, '='))
    print("="*50 + "\n")

    # Create table data
    table_data = []
    for rank, guid in enumerate(top_k_retrieved):
        table_data.append({"rank": rank, "guid": guid})

    # Print table
    headers = ["Rank", "GUID"]
    print(tabulate([[d["rank"], d["guid"]] for d in table_data[0:25]], headers=headers, tablefmt="fancy_grid"))
    print("\n" + "="*50 + "\n")
    print()
    print("In terminal only a max of the top-25 are shown. ") 
    
    # Save to JSON file
    if o_path:
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        output_file = os.path.join(o_path, f"{region}.json")
        with open(output_file, "w") as f:
            json.dump(table_data, f, indent=4)
        print(f"Saved in: {output_file}")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    package_dir = os.path.dirname(neurocbir.__file__)
    default_internal_config = os.path.join(package_dir, "configs/internal_config.yaml")
    default_user_config = os.path.join(package_dir, "configs/user_config.yaml")

      
    # Optional overrides
    parser.add_argument('--brain_path', help='Path to the preprocessed brain MRI image in .nii.gz, .nii or .mgz format.')
    parser.add_argument('--seg_path', help='Path to the segmentation file (aparc+aseg.nii) in .nii.gz, .nii or .mgz format.')
    parser.add_argument('--o_path', help='Path to the output file. If not specified then not saved.')
    parser.add_argument('--emb_dataset_path', help='Path to the embedding dataset in .parquet format.')
    parser.add_argument("--scope", choices=["whole_brain","region"])
    parser.add_argument("--region", help='Region name (see labels.csv). Example: --score "region" --region "Left-Hippocampus". Requiered if --scope region is selected.')
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--device", choices=["cpu","cuda"])
    parser.add_argument("--internal_config", default=default_internal_config, help="Path to the internal (read-only) config YAML.")
    parser.add_argument("--user_config", default=default_user_config, help="Path to the user config YAML.")
    args = parser.parse_args()

    # Load configuration file
    internal_config = load_yaml(args.internal_config)
    user_config = load_yaml(args.user_config)
    cli_config = vars(args)
    
    # Priority: default < internal config < user config < CLI arguments
    scope = cli_config.get("scope") or user_config.get("scope") or internal_config.get("scope")
    if scope not in ["region"]:
        raise Exception(f"--scope must be 'region'. Currently scope = {scope}") 
    
    # Load scope-specific configurations
    internal_config.update(internal_config.get(scope, {}))
    user_config.update(user_config.get(scope, {}))
    
    # Merge configurations with priority
    config = {}
    config.update(internal_config)
    config.update(user_config)
    
    # Override with CLI arguments if given
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value    
    
    # Enforce condition
    if config.get("scope") == "region" and not config.get("region"):
        raise Exception("--region is required when --scope=region")

    # Override with CLI arguments if given
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    main(**config)