from dev.preprocessing.prepare_mock_dataset import prepare_mock_dataset
import logging
import argparse
import os

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BatchedNPZDataset loader")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--skip-wb-vae", action='store_true', help="Skip whole-brain VAE training test")
    parser.add_argument("--skip-region-vae", action='store_true', help="Skip region-brain VAE training test")
    parser.add_argument("--skip-wb-cl", action='store_true', help="Skip whole-brain contrastive model training test")
    parser.add_argument("--skip-region-cl", action='store_true', help="Skip region-brain contrastive model training test")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Mock dataset settings
    image_shape=(32, 32, 32)
    
    # Prepare mock dataset
    # Original dataset
    logging.info("Preparing mock dataset...")
    prepare_mock_dataset("dev/data_private/mock_dataset", image_shape=image_shape)
            
    # Test create_data_index script
    logging.info("Testing data index creation script...")
    from dev.utils import load_config_from_path
    from dev.scripts.create_data_index_csv import main as create_data_index
    config_path = "dev/configs/data_index_config.py"
    config = load_config_from_path(config_path)
    create_data_index(config)   
    
    # >>> Whole-brain <<<
    logging.info("Whole-brain tests...")

    # Test whole-brain autoencoder training script
    from dev.utils import load_config_from_path
    from dev.scripts.whole_brain.train_autoencoder import main as train_wb_autoencoder 
    from datetime import datetime
    config_path = "dev/configs/whole_brain/train_autoencoder.py"
    config = load_config_from_path(config_path)
    if args.skip_wb_vae:
        logging.info("Skipping whole-brain VAE training test as per argument.")
        config["num_epochs"] = 0  # Reduce epochs for testing
    else:
        logging.info("Testing whole-brain VAE training script...")
        config["num_epochs"] = 2  # Reduce epochs for testing
    config["batch_size"] = 2  # Reduce batch size for testing
    config["max_batch_size"] = 1  # Adjust max batch size accordingly
    # Set dynamic paths
    run_GUID = "vae_whole_brain" # datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    train_wb_autoencoder(config)
    
    # Test run_vae_embedding script
    logging.info("Testing whole-brain VAE embedding script...")
    from dev.utils import load_config_from_path
    from dev.scripts.whole_brain.run_vae_embedding import main as run_wb_vae_embedding
    config_path = "dev/configs/whole_brain/run_vae_embedding.py"
    config = load_config_from_path(config_path)
    config["ckpt_path"] = "dev/data_private/mock_dataset/logs/vae_whole_brain/checkpoint-epoch-0.pth"
    config["device"] = "cpu"  # Use CPU for testing
    run_wb_vae_embedding(config)
    
    # Test create_data_index script
    logging.info("Testing data index creation script whole-brain CL embeddings...")
    from dev.utils import load_config_from_path
    config = {
        "datasets": {
            "OASIS": "dev/data_private/mock_dataset/whole_brain/batched_OASIS3",
            "ADNI": "dev/data_private/mock_dataset/whole_brain/batched_adni",
        },
        "output_csv": "dev/data_private/mock_dataset/whole_brain/dataset_index.csv",
        "id_key": "GUID"
    }
    create_data_index(config) 
    
    # Test train_contrastive_model script
    logging.info("Testing whole-brain contrastive model training script...")
    from dev.utils import load_config_from_path
    from dev.scripts.whole_brain.train_contrastive_model import main as train_wb_contrastive_model
    config_path = "dev/configs/whole_brain/train_contrastive_model.py"
    config = load_config_from_path(config_path)
    config["n_batches_per_file"] = 1  # Reduce batches per file for testing
    config["batch_size"] = 16  # Reduce batch size for testing
    if args.skip_wb_cl:
        logging.info("Skipping whole-brain CL training test as per argument.")
        config["num_epochs"] = 0  # Reduce epochs for testing
    else:
        logging.info("Testing whole-brain CL training script...")
        config["num_epochs"] = 2  # Reduce epochs for testing
        # Set dynamic paths
    run_GUID = "cl_whole_brain" # datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    config["device"] = "cpu"  # Use CPU for testing
    train_wb_contrastive_model(config)
    
    # Test run_cl_embedding script
    logging.info("Testing whole-brain CL embedding script...")
    from dev.utils import load_config_from_path
    from dev.scripts.whole_brain.run_cl_embedding import main as run_wb_cl_embedding
    config_path = "dev/configs/whole_brain/run_cl_embedding.py"
    config = load_config_from_path(config_path)
    run_wb_cl_embedding(config) 
    
    # Test run_cbir_eval script
    logging.info("Testing whole-brain CBIR evaluation script...")
    from dev.utils import load_config_from_path
    from dev.scripts.whole_brain.run_cbir_eval import main as run_wb_cbir_eval
    config_path = "dev/configs/whole_brain/run_cbir_eval.py"
    config = load_config_from_path(config_path)
    config["device"] = "cpu"  # Use CPU for testing
    run_wb_cbir_eval(config)
    
    
    
    # >>> Region-brain <<<
    logging.info("Region-brain tests...")
    
    # Create bounding boxes
    logging.info("Testing bounding box creation script...")
    from dev.utils import load_config_from_path
    from dev.scripts.region_brain.create_bounding_boxes import main as create_bounding_boxes
    config_path = "dev/configs/create_bb_config.py"
    config = load_config_from_path(config_path)
    config["input_shape"] = image_shape
    config["subcortical_indices"] = {
        "Left-Hippocampus": 0,
        "Left-Cerebral-White-Matter": 1,}
    create_bounding_boxes(config)
    
    # Create fake labels.csv
    # This is provided as the real labels.csv is not in the repo
    logging.info("Creating fake labels.csv for testing...")
    import pandas as pd
    import os

    output_path = "dev/data_private/mock_dataset/labels.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([
        [0, 0, "Left-Hippocampus", 220, 216, 20, 0, 1],
        [1, 1, "Left-Cerebral-White-Matter", 245, 245, 245, 0, 1],
    ], columns=["LabelID", "MapID", "LabelName", "R", "G", "B", "A", "Use"])

    # Save as tab-separated to match your example
    df.to_csv(output_path, index=False)

    logging.info(f"Fake labels.csv created at: {output_path}")   

    # Test region-brain autoencoder training script
    from dev.utils import load_config_from_path
    from dev.scripts.region_brain.train_autoencoder import main as train_region_autoencoder 
    from datetime import datetime
    config_path = "dev/configs/region_brain/train_autoencoder.py"
    config = load_config_from_path(config_path)
    if args.skip_region_vae:
        logging.info("Skipping region-brain VAE training test as per argument.")
        config["num_epochs"] = 0  # Reduce epochs for testing
    else:
        logging.info("Testing region-brain VAE training script...")
        config["num_epochs"] = 2  # Reduce epochs for testing
    config["batch_size"] = 2  # Reduce batch size for testing
    config["max_batch_size"] = 1  # Adjust max batch size accordingly
    # Set dynamic paths
    run_GUID = "vae_region_brain" # datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    train_region_autoencoder(config)
    
    # Test run_vae_embedding script
    logging.info("Testing region-brain VAE embedding script...")
    from dev.utils import load_config_from_path
    from dev.scripts.region_brain.run_vae_embedding import main as run_region_vae_embedding
    config_path = "dev/configs/region_brain/run_vae_embedding.py"
    config = load_config_from_path(config_path)
    config["ckpt_path"] = "dev/data_private/mock_dataset/logs/vae_region_brain/checkpoint-epoch-0.pth"
    config["device"] = "cpu"  # Use
    config["batch_size"] = 16  # Reduce batch size for testing
    run_region_vae_embedding(config)
    
    # Test create_data_index script
    logging.info("Testing data index creation script region-brain CL embeddings...")
    from dev.utils import load_config_from_path
    config = {
        "datasets": {
            "OASIS": "dev/data_private/mock_dataset/region_brain/batched_OASIS3",
            "ADNI": "dev/data_private/mock_dataset/region_brain/batched_adni",
        },
        "output_csv": "dev/data_private/mock_dataset/region_brain/dataset_index.csv",
        "id_key": "GUID"
    }
    create_data_index(config) 
    
    # Test train_cl_embedding script
    logging.info("Testing region-brain contrastive model training script...")
    from dev.utils import load_config_from_path
    from dev.scripts.region_brain.train_contrastive_model import main as train_region_contrastive_model
    config_path = "dev/configs/region_brain/train_contrastive_model.py"
    config = load_config_from_path(config_path)
    if args.skip_region_cl:
        logging.info("Skipping region-brain CL training test as per argument.")
        config["num_epochs"] = 0  # Reduce epochs for testing
    else:
        logging.info("Testing region-brain CL training script...")
        config["num_epochs"] = 2  # Reduce epochs for testing
        # Set dynamic paths
    run_GUID = "cl_region_brain" # datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logging_path"] = os.path.join(config["base_logging_path"], run_GUID)
    config["device"] = "cpu"  # Use CPU for testing
    config["groups_per_batch"] = 2  # Reduce groups per batch for testing
    config["batch_size"] = 16  # Reduce batch size for testing
    config["group_size"] = 3  # Reduce group size for testing
    train_region_contrastive_model(config)
    
    # Test run_cl_embedding script
    logging.info("Testing region-brain CL embedding script...")
    from dev.utils import load_config_from_path
    from dev.scripts.region_brain.run_cl_embedding import main as run_region_cl_embedding
    config_path = "dev/configs/region_brain/run_cl_embedding.py"
    config = load_config_from_path(config_path)
    run_region_cl_embedding(config) 
    
    # Test run_cbir_eval script
    logging.info("Testing region-brain CBIR evaluation script...")
    from dev.utils import load_config_from_path
    from dev.scripts.region_brain.run_cbir_eval import main as run_region_cbir_eval
    config_path = "dev/configs/region_brain/run_cbir_eval.py"
    config = load_config_from_path(config_path)
    config["device"] = "cpu"  # Use CPU for testing
    run_region_cbir_eval(config)
    
    # >>> Multi comparison check <<<
    logging.info("Testing multi-comparison CBIR evaluation script...")
    from dev.utils import load_config_from_path
    from dev.scripts.run_embedding_pm import main as run_embedding_pm
    config_path = "dev/data/multi_comp/resnet10/config/embedding_pm.py"
    config = load_config_from_path(config_path)
    run_embedding_pm(config)
    config_path = "dev/data/multi_comp/resnet10/config/cbir_eval_config.py"
    config = load_config_from_path(config_path)

        
