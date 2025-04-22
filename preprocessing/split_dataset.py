from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def stratified_patient_split(image_paths, labels, ages, ids, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits the dataset into training, validation, and test sets while ensuring:
    - Stratification based on labels (each partition maintains label distribution).
    - Grouping by patient IDs (each patient appears in only one set).

    Parameters:
    - image_paths: np.array of image file paths.
    - labels: np.array of labels (0 or 1).
    - ages: np.array of ages.
    - ids: np.array of patient IDs.
    - test_size: Proportion of data assigned to the test set.
    - val_size: Proportion of remaining data assigned to the validation set.
    - random_state: Seed for reproducibility.

    Returns:
    - train, val, test partitions as dictionaries containing image_paths, labels, ages, and ids.
    """
    image_paths, labels, ages, ids = map(np.array, [image_paths, labels, ages, ids])

    # Step 1: First split -> Training+Validation (1 - test_size) and Test (test_size)
    sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(sgkf.split(image_paths, labels, groups=ids))

    # Extract test set
    test_set = {
        "image_paths": image_paths[test_idx],
        "labels": labels[test_idx],
        "ages": ages[test_idx],
        "ids": ids[test_idx]
    }

    # Step 2: Second split -> Training (1 - val_size) and Validation (val_size) from Train+Validation set
    train_val_image_paths = image_paths[train_val_idx]
    train_val_labels = labels[train_val_idx]
    train_val_ages = ages[train_val_idx]
    train_val_ids = ids[train_val_idx]

    sgkf = StratifiedGroupKFold(n_splits=int(1/val_size), shuffle=True, random_state=random_state)
    train_idx_rel, val_idx_rel = next(sgkf.split(train_val_image_paths, train_val_labels, groups=train_val_ids))

    # Extract training and validation sets
    train_set = {
        "image_paths": train_val_image_paths[train_idx_rel],
        "labels": train_val_labels[train_idx_rel],
        "ages": train_val_ages[train_idx_rel],
        "ids": train_val_ids[train_idx_rel]
    }

    val_set = {
        "image_paths": train_val_image_paths[val_idx_rel],
        "labels": train_val_labels[val_idx_rel],
        "ages": train_val_ages[val_idx_rel],
        "ids": train_val_ids[val_idx_rel]
    }

    return train_set, val_set, test_set