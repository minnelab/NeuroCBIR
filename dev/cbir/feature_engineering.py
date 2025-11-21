# import pandas as pd
# import numpy as np

# def prepare_dataset_with_combined_features(dataset: pd.DataFrame, subc_strs: list) -> pd.DataFrame:
#     """
#     Combines features across the selected subcortical structures into a single feature vector per record.
#     Assumes each subject-record_id pair has one row per subcortical structure.
#     """

#     # Filter the dataset for the selected subcortical structures
#     dataset_subc_str = dataset.copy()[dataset['subc_str'].isin(subc_strs[0:1])]
#     dataset_subc_str.subc_str = [subc_strs] * len(dataset_subc_str)
#     list_features_to_concat = []
#     for subc_str in subc_strs:
#         list_features_to_concat.append(np.stack(dataset.copy()[dataset['subc_str'].isin([subc_str])].features.values))
#     dataset_subc_str.features = list(np.concatenate(list_features_to_concat, axis=1))

#     return dataset_subc_str