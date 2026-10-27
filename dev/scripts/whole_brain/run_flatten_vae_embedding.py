import os
import argparse
import pandas as pd

from dev.preprocessing import EmbBatchedDataset
from dev.utils import load_config_from_path


def main(config):
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load metadata
    index_ds = pd.read_csv(os.path.join(config["data_path"], config["dataset_index_file_name"]))
    clinical_ds = pd.read_csv(os.path.join(config["data_path"], config["metadata_file_name"]))
    metadata = pd.merge(index_ds, clinical_ds, on="GUID", how="inner")
    metadata["subject"] = metadata["subject"].replace("", pd.NA)
    metadata = metadata.dropna( subset=["subject"] ).reset_index(drop=True)
    print(f"Loaded metadata: {len(metadata)} records.")

    # Filter by partition if specified
    partition = config.get("partition", None)

    if partition is not None:
        if partition.lower() not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid partition value: {partition}. "
                "Must be 'train', 'val', 'test', or not set."
            )

        metadata = metadata[
            metadata["partition"].str.lower() == partition.lower()
        ].reset_index(drop=True)

        print(
            f"Using partition: {partition} | "
            f"{len(metadata)} records selected."
        )
    else:
        print(
            f"Using all data | {len(metadata)} records selected."
        )

    # Load initial embeddings
    batch_files = sorted(metadata["batch_file"].unique())
    dataset = EmbBatchedDataset(
        metadata,
        batch_files=batch_files
    )
    raw_embs_np = dataset.embs
    all_guids = dataset.guids

    # Flatten embeddings to [N, D]
    raw_embs_np = raw_embs_np.reshape(raw_embs_np.shape[0], -1)

    print(f"Embedding matrix shape: {raw_embs_np.shape}")

    # Create DataFrame and preserve GUID as the first column
    df_embs = pd.DataFrame(raw_embs_np)
    df_embs.insert(0, "GUID", all_guids)
    df_embs.columns = df_embs.columns.astype(str)

    # Save in the same format as the original script
    output_path = os.path.join(config["output_dir"], "projected_embeddings.parquet")
    df_embs.to_parquet(output_path, index=False)

    print(f"Initial embeddings saved to {output_path}")
    print(f"Output shape: {df_embs.shape}")

    del dataset, metadata, raw_embs_np, df_embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config .py file"
    )

    args = parser.parse_args()
    config = load_config_from_path(args.config)

    main(config)
