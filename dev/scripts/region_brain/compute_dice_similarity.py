import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from preprocessing.load_dataset import SingleStructDataset
from utils import load_config_from_path
from monai.utils import set_determinism


def dice_score(mask1, mask2, eps=1e-6):
    intersection = (mask1 * mask2).sum().item()
    union = mask1.sum().item() + mask2.sum().item()
    return (2. * intersection + eps) / (union + eps)


def main(config):
    set_determinism(config["random_state"])
    os.makedirs(config["save_path"], exist_ok=True)

    # Load metadata
    index_ds = pd.read_csv(os.path.join(config["data_path"], "dataset_index.csv"))
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], "combined_metadata.csv"))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner")
    metadata['subject'].replace('', pd.NA, inplace=True)
    metadata = metadata.dropna(subset=['subject']).reset_index(drop=True)

    # Load label & bounding box info
    labels_df = pd.read_csv(config["labels_path"])
    bb_df = pd.read_csv(config["bb_path"])
    labels_bb_df = pd.merge(labels_df, bb_df, on="LabelName", how="inner")

    # Load dataset for a single subcortical structure
    batch_files = sorted(metadata["batch_file"].unique())
    dataset = SingleStructDataset(
        metadata=metadata,
        batch_files=batch_files,  # specify single batch file for simplicity
        labels_bb_df=labels_bb_df,
        target_struct_name=config["target_struct_name"]
    )

    print(f"Loaded {len(dataset)} samples for structure: {config['target_struct_name']}")

    # Precompute all masks
    guids = []
    struct_name = config["target_struct_name"]
    masks = []

    for sample in tqdm(dataset, desc="Preloading masks"):
        guids.append(sample["GUID"])
        masks.append(sample["mask"].float())  # shape: [1, 64, 64, 64]

    masks = torch.stack(masks)  # shape: [N, 1, 64, 64, 64]

    # Compute Dice similarity matrix
    N = len(masks)
    dice_matrix = torch.zeros((N, N))

    print("Computing Dice similarities...")
    for i in tqdm(range(N), desc="Dice matrix"):
        for j in range(i + 1, N):
            score = dice_score(masks[i], masks[j])
            dice_matrix[i, j] = score
            dice_matrix[j, i] = score  # symmetric

    # For each mask, find top-5 most similar other masks
    top_k = 5
    results = []

    for i in range(N):
        scores = dice_matrix[i].clone()
        scores[i] = -1  # ignore self
        topk_indices = torch.topk(scores, top_k).indices.tolist()
        result = {
            "GUID": guids[i],
            "struct_name": struct_name
        }
        for k, idx in enumerate(topk_indices, start=1):
            result[f"top{k}_guid"] = guids[idx]
        results.append(result)

    # Save to CSV
    df = pd.DataFrame(results)
    out_path = os.path.join(config["save_path"], f"dice_top5_{struct_name.replace(' ', '_')}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config .json")
    args = parser.parse_args()

    config = load_config_from_path(args.config)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
