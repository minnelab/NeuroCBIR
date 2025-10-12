import argparse
import logging
import os
import json
from neurocbir.runners import cbir_region, cbir_whole_brain
from neurocbir.utils import load_yaml 
import neurocbir


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

def main():
    
    setup_logging()
    parser = argparse.ArgumentParser()
    
    package_dir = os.path.dirname(neurocbir.__file__)
    default_internal_config = os.path.join(package_dir, "configs/internal_config.yaml")
    default_user_config = os.path.join(package_dir, "configs/user_config.yaml")
      
    # Optional overrides
    parser.add_argument('--img_path', help='Path to the preprocessed brain MRI image in .nii.gz, .nii or .mgz format.')
    parser.add_argument('--seg_path', help='Path to the segmentation file (aparc+aseg.nii) in .nii.gz, .nii or .mgz format.')
    parser.add_argument('--o_path', help='Path to the output file. If not specified then not saved.')
    parser.add_argument('--emb_dataset_path', help='Path to the embedding dataset in .parquet format.')
    parser.add_argument("--scope", choices=["whole_brain","region"])
    parser.add_argument("--region", help='Region name (see labels.csv). Example: --scope "region" --region "Left-Hippocampus". Requiered if --scope region is selected.')
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--device", choices=["cpu","cuda"])
    parser.add_argument("--internal_config", default=default_internal_config, help="Path to the internal (read-only) config YAML.")
    parser.add_argument("--user_config", default=default_user_config, help="Path to the user config YAML.")
    args = parser.parse_args()

    # Load configuration file
    internal_config = load_yaml(args.internal_config)
    config = internal_config["common"]
    config.update(load_yaml(args.user_config))         
    
    # Override with CLI arguments if given
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
            
    # Load internal config
    if config["scope"] == "whole_brain":
        config.update(internal_config["whole_brain"])
    elif config["scope"] == "region":
        config.update(internal_config["region"])
        # Enforce condition
        if config.get("scope") == "region" and not config.get("region"):
            raise Exception("--region is required when --scope=region")
    else:
        raise Exception(f"--scope must be either 'whole_brain' or 'region'. Currently scope = {config['scope']}") 
            
    # 🔍 Display final configuration
    print("\n" + "="*70)
    print("NeuroCBIR Running Configuration")
    print("="*70)
    print(json.dumps(config, indent=4, sort_keys=True))
    print("="*70 + "\n")
    
    if config["scope"] == "whole_brain":
        cbir_whole_brain(**config)
    elif config["scope"] == "region":
        cbir_region(**config)
        
if __name__ == "__main__":
    main()
