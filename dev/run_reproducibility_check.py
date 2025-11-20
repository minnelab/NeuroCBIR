import datetime
from dev.preprocessing.prepare_mock_dataset import prepare_mock_dataset
import torch
import logging
import argparse
import os

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BatchedNPZDataset loader")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Prepare mock dataset
    # Original dataset
    prepare_mock_dataset("data/mock_dataset/original", image_shape=(32, 32, 32))
            
    # Test create_data_index script
    from dev.utils import load_config_from_path
    from dev.scripts.create_data_index_csv import main as create_data_index
    config_path = "dev/configs/data_index_config.py"
    config = load_config_from_path(config_path)
    create_data_index(config)
    

    # Test whole-brain autoencoder training script
    from dev.utils import load_config_from_path
    from dev.scripts.whole_brain.train_autoencoder import main as train_autoencoder 
    from datetime import datetime
    config_path = "dev/configs/whole_brain/train_autoencoder_config.py"
    config = load_config_from_path(config_path)
    config["num_epochs"] = 2  # Reduce epochs for testing
    config["batch_size"] = 2  # Reduce batch size for testing
    config["max_batch_size"] = 1  # Adjust max batch size accordingly
    # Set dynamic paths
    run_GUID = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    train_autoencoder(config)
    

        
