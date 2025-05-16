from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def stratified_patient_split(test_size=0.15, val_size=0.15, random_state=42, label_key="labels", group_key="ids", **data_dict):
    """
    Splits dataset into train/val/test sets with stratification and patient grouping.

    Parameters:
    - test_size: Proportion of data for test split.
    - val_size: Proportion of train+val split for validation.
    - random_state: Seed for reproducibility.
    - label_key: Key in data_dict used for stratification.
    - group_key: Key in data_dict used for grouping patients.
    - **data_dict: Arbitrary number of named arrays/lists of equal length.

    Returns:
    - train_set, val_set, test_set: dictionaries with the same keys as input.
    """
    # Convert all to NumPy arrays and check lengths
    data_dict = {k: np.array(v) for k, v in data_dict.items()}
    lengths = [len(v) for v in data_dict.values()]
    assert len(set(lengths)) == 1, "All input arrays must be the same length."

    labels = data_dict[label_key]
    groups = data_dict[group_key]

    sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(sgkf.split(labels, labels, groups=groups))

    def subset(indices):
        return {k: v[indices] for k, v in data_dict.items()}

    test_set = subset(test_idx)
    train_val_set = subset(train_val_idx)

    # Second split: train and val
    train_val_labels = train_val_set[label_key]
    train_val_groups = train_val_set[group_key]

    sgkf = StratifiedGroupKFold(n_splits=int(1/val_size), shuffle=True, random_state=random_state)
    train_idx_rel, val_idx_rel = next(sgkf.split(train_val_labels, train_val_labels, groups=train_val_groups))

    # Use relative indices to index into train_val_set
    train_set = subset(train_val_idx[train_idx_rel])
    val_set = subset(train_val_idx[val_idx_rel])

    return train_set, val_set, test_set
