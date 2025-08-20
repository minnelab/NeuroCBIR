import os
import numpy as np
import pandas as pd

def fix_batch_file(npz_path):
    print(f"Fixing {npz_path}...")

    data = np.load(npz_path)
    mus = data["mus"]                  # Should be shape [N, D] or [N, ...]
    wrong_guids = data["GUID"]         # Shape [N * M], should be [N]
    label_names = data["LabelName"][0]    # Shape [1 * M]

    # Infer the correct number of samples
    n_samples = mus.shape[0]

    # Attempt to recover the correct GUIDs: take every M-th entry
    # assuming they were repeated M times
    repetition_factor = len(wrong_guids) // n_samples
    if repetition_factor * n_samples != len(wrong_guids):
        raise ValueError(f"GUID length {len(wrong_guids)} is not a multiple of mus length {n_samples}.")

    # Recover correct GUIDs
    fixed_guids = wrong_guids[::repetition_factor]

    # Optional: verify that they are repeating correctly
    for i in range(n_samples):
        assert all(wrong_guids[i * repetition_factor:(i + 1) * repetition_factor] == fixed_guids[i]), \
            f"GUID mismatch at index {i}"

    assert mus.shape[0] == len(fixed_guids), "mus len is not equal to guids len"

    # Save fixed data back (overwrite)
    np.savez_compressed(
        npz_path,
        mus=mus,
        GUID=fixed_guids,
        LabelName=label_names
    )
    print(f"✔ Saved fixed {npz_path}")

def fix_all_in_folder(folder_path):
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".npz"):
            fix_batch_file(os.path.join(folder_path, filename))

if __name__ == "__main__":
    base_folders = [
        "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/region_brain/batched_adni",
        "/mimer/NOBACKUP/groups/naiss2025-23-412/felixnie/region_brain/batched_OASIS3"
    ]

    for folder in base_folders:
        print(f"\n--- Processing folder: {folder} ---")
        fix_all_in_folder(folder)
