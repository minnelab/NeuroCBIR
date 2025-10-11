import argparse
import pandas as pd
from tabulate import tabulate
import torch
import logging
import os
import json

from neurocbir.utils import load_yaml, load_brain, retrieve_topk_for_query
from neurocbir.models.q2e_model import build_Q2E  
import neurocbir

logger = logging.getLogger(__name__)

def main(config):
    logger.info("CBIR pipeline: whole_brain")
    
    # Setup
    device = config["device"]
    q2e_module = build_Q2E(config, device) # Load 
    
    # Load dataset
    # Load real features from parquet
    logger.info("Loading embedding dataset.")
    embs_dataset = pd.read_parquet(config["emb_dataset_path"])
    embs_dataset["GUID"] = embs_dataset["GUID"].astype(str) # Ensure GUID is string  
    # Convert embedding columns into a single 'features' column of vectors
    embedding_cols = [col for col in embs_dataset.columns if col != "GUID"]
    embs_dataset["features"] = embs_dataset[embedding_cols].apply(lambda row: row.to_numpy(), axis=1) 
    # Drop the old embedding columns
    embs_dataset = embs_dataset[["GUID", "features"]]

    # Load whole-brain image query
    i_q = load_brain(config["img_path"])
    i_q = torch.from_numpy(i_q).float().unsqueeze(0).unsqueeze(0).to(device)

    # Get features of the query
    logger.info("Running Q2E on query. Returning features.")
    z_q = q2e_module(i_q)
    
    # Top-k retrieval
    logger.info(f"Computing similarities between query {list(z_q.shape)} and dataset {list(embs_dataset.features.values.shape)}.")
    top_k_retrieved = retrieve_topk_for_query(z_q, embs_dataset, top_k=config['top_k'])
    logger.info(f"Ranking and retrieving Top-{config['top_k']}.")
    
    # Fancy print
    print("\n" + "="*50)
    print(f"Top-{config['top_k']} Retrieval Results ".center(50, '='))
    print("="*50 + "\n")

    # Create table data
    table_data = []
    for rank, guid in enumerate(top_k_retrieved):
        table_data.append({"rank": rank, "guid": guid})

    # Print table
    headers = ["Rank", "GUID"]
    print(tabulate([[d["rank"], d["guid"]] for d in table_data[0:25]], headers=headers, tablefmt="fancy_grid"))
    print("\n" + "="*50 + "\n")
    print("In terminal only a max of the top-25 are shown. ") 
    
    # Save to JSON file
    o_path = config['o_path']
    if o_path:
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        output_file = os.path.join(o_path, "whole_brain.json")
        with open(output_file, "w") as f:
            json.dump(table_data, f, indent=4)
        print(f"Saved in: {output_file}")
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    package_dir = os.path.dirname(neurocbir.__file__)
    default_internal_config = os.path.join(package_dir, "configs/internal_config.yaml")
    default_user_config = os.path.join(package_dir, "configs/user_config.yaml")
      
    # Optional overrides
    parser.add_argument('--img_path', help='Path to the preprocessed brain MRI image in .nii.gz, .nii or .mgz format.')
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
    config = internal_config["common"]
    config.update(internal_config["whole_brain"])
    config.update(load_yaml(args.user_config))
        
    # Enforce condition
    if config.get("scope") != "whole_brain":
        raise Exception(f"--region must be 'whole_brain'")

    # Override with CLI arguments if given
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    main(config)