from dev.preprocessing.prepare_mock_dataset import prepare_mock_dataset
import torch
import logging
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BatchedNPZDataset loader")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    if os.path.exists("data/mock_dataset") is False:
        prepare_mock_dataset()

    dataset = BatchedNPZDataset(
        root_dirs=["data/mock_dataset/batched_adni", "data/mock_dataset/batched_OASIS3"],
        metadata_csv="data/mock_dataset/metadata.csv",
        preload=True
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # dataset already iterates sequentially
        num_workers=0   # workers would break threading inside dataset
    )

    for img, meta, guid in loader:
        logging.info(f"Loaded image shape: {img.shape}, GUID: {guid}")
        
