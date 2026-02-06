import numpy as np
import pandas as pd
import random
from pathlib import Path
import logging

def prepare_mock_dataset(output_root="data/mock_dataset", image_shape=(1, 64, 64, 64)):
    """
    Prepares a mock dataset with random MRI volumes and metadata for testing.
    Parameters:
        output_root (str): Root directory to save the mock dataset.
        image_shape (tuple): Shape of the mock MRI volumes (C, H, W, D).
    """
    logger = logging.getLogger(__name__)
    
    # -----------------------------
    # Settings
    # -----------------------------
    output_root = Path(output_root) 
    output_data_path = output_root / "original"
    output_data_path.mkdir(parents=True, exist_ok=True)

    batch_specs = {
        "batched_adni": 2,   # 2 batches
        "batched_OASIS3": 1, # 1 batch
    }

    subjects_per_batch = 5
    n_samples_per_subject = 5
    metadata_rows = []

    # -----------------------------
    # Generate batches
    # -----------------------------
    for folder, num_batches in batch_specs.items():
        batch_dir = output_data_path / folder
        batch_dir.mkdir(parents=True, exist_ok=True)

        for b in range(num_batches):
            imgs = []
            segs = []
            guids = []

            for s in range(subjects_per_batch):
                
                for n in range(n_samples_per_subject):

                    guid = f"mock_{folder}_{b+1:04d}_{s+1:04d}_{n+1:04d}"
                    guids.append(guid)

                    # Random MRI volume
                    img = np.random.rand(*image_shape).astype(np.float32)

                    # Random segmentation mask
                    seg = np.random.randint(0, 2, size=image_shape).astype(np.uint8)

                    imgs.append(img)
                    segs.append(seg)

                    # Metadata row
                    metadata_rows.append({
                        "GUID": guid,
                        "project": guid.split("_")[0],
                        "subject": f"s{s+1:04d}",
                        "timepoint": f"t{n+1:04d}",
                        "scan_type": "T1",
                        "field_strength": 3,
                        "manufacturer": "SIEMENS",
                        "model_name": "MockModel",
                        "disease": random.choice(["CN", "MCI", "AD"]),
                        "age": random.randint(20, 90),
                        "partition": random.choice(["train", "val", "test"]),
                        "brain_qc": round(random.random(), 2),
                        "repet": 1,
                        "useable": 1,
                        "mislabel": 0,
                    })

                # Save NPZ batch
                save_path = batch_dir / f"batch_{b+1:04d}.npz"
                np.savez_compressed(
                    save_path,
                    images=np.stack(imgs),
                    segmentations=np.stack(segs),
                    GUID=np.array(guids, dtype=object)
                )

                print(f"Created: {save_path}")

    # -----------------------------
    # Write metadata.csv
    # -----------------------------
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = output_root / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    logger.info(f"\nMetadata saved to: {metadata_path}")
    logging.info("\nMock dataset generation complete!")


if __name__ == "__main__":
    prepare_mock_dataset()
    
