import argparse
from .utils import load_config_from_path, load_nifti, retrieve_topk_for_query
from .model import build_Q2E
import pandas as pd
from tabulate import tabulate

def main(config):
    # Setup
    device = config["--device"]
    q2e_module = build_Q2E(config, device) # Load 
    
    # Load dataset
    # Load real features from parquet
    embs_dataset = pd.read_parquet(config["emb_dataset_path"])
    embs_dataset["GUID"] = embs_dataset["GUID"].astype(str) # Ensure GUID is string  
    # Convert embedding columns into a single 'features' column of vectors
    embedding_cols = [col for col in embs_dataset.columns if col != "GUID"]
    embs_dataset["features"] = embs_dataset[embedding_cols].apply(lambda row: row.to_numpy(), axis=1) 
    # Drop the old embedding columns
    embs_dataset = embs_dataset[["GUID", "features"]]

    # Load whole-brain image query
    i_q = load_nifti(config["--img_path"])
    
    # Get features of the query
    z_q = q2e_module(i_q)
    
    # Top-k retrieval
    top_k_retrieved = retrieve_topk_for_query(z_q, embs_dataset, top_k=config["--top_k"])
    
    # Fancy print
    print("\n" + "="*50)
    print(f"Top-{config["--top_k"]} Retrieval Results ".center(50, "="))
    print("="*50 + "\n")

    # Create table
    table_data = []
    for rank, item in enumerate(top_k_retrieved, 1):
        table_data.append([rank, item["GUID"]])

    headers = ["Rank", "GUID"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    print("\n" + "="*50 + "\n")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
      
    parser.add_argument('--img_path', required=True, help='Path to the MRI image in .nii.gz, .nii or .mgz format.')
    parser.add_argument("--scope", required=True, choices=["whole-brain","region"])
    parser.add_argument("--region", help='Region name (see labels.csv). Example: --score "region" --region "Left-Hippocampus". Requiered if --scope region is selected.')
    parser.add_argument("--top_k", default=25, choices=["whole-brain","region"])
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    parser.add_argument('--config', default="configs/config_run_cbir_whole_brain.py", help='Path to the config .py file') 
    args = parser.parse_args()
        
    # Enforce condition
    if args.scope == "region" and args.region is None:
        parser.error("--region is required when --scope=region")

    # Load configuration file
    config = load_config_from_path(args.config)
    
    # Add additional configuration parameters
    config.upload(args)
    
    main(config)