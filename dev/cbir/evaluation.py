import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

def get_topk_guid_retrievals(dataset: pd.DataFrame, top_k: int = 3, feature_column: str = 'features', guid_column: str = 'GUID') -> pd.DataFrame:
    """
    For each row in the dataset, retrieve the top-k most similar (cosine) entries and return their GUIDs.

    Args:
        dataset (pd.DataFrame): DataFrame with features and GUIDs.
        top_k (int): Number of top similar entries to retrieve.
        feature_column (str): Name of the column containing the feature vectors.
        guid_column (str): Column name for unique scan identifiers (e.g., 'GUID').

    Returns:
        pd.DataFrame: DataFrame with one row per query, first column is the query GUID,
                      and columns 1...k are the GUIDs of the top-k retrieved entries.
    """
    features_matrix = np.stack(dataset[feature_column].values)
    guids = dataset[guid_column].values
    retrievals = []

    for i in tqdm(range(len(dataset))):
        similarities = cosine_similarity(features_matrix[i].reshape(1, -1), features_matrix)[0]
        similarities[i] = -1  # Exclude self
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        row = [guids[i]] + guids[top_k_indices].tolist()
        retrievals.append(row)

    col_names = ['query'] + [f'top{i+1}' for i in range(top_k)]
    return pd.DataFrame(retrievals, columns=col_names)

def retrieve_topk_for_queries(
    dataset: pd.DataFrame,
    queries: pd.DataFrame,
    top_k: int = 3,
    feature_column: str = "features",
    guid_column: str = "GUID",
    exclude_self: bool = True,
    exclude_same_subject: bool = True,
) -> pd.DataFrame:
    """
    Retrieve the top-k most similar entries for a subset of queries, 
    using cosine similarity against the full dataset as the retrieval pool.

    Args:
        dataset (pd.DataFrame): Full pool of entries with features and GUIDs.
        queries (pd.DataFrame): Subset of rows from dataset to use as queries.
        top_k (int): Number of top similar entries to retrieve.
        feature_column (str): Column containing the feature vectors.
        guid_column (str): Column with unique scan identifiers (e.g., 'GUID').

    Returns:
        pd.DataFrame: Retrieval results. One row per query, first column is the query GUID,
                      followed by the GUIDs of the top-k retrieved entries.
    """
    if top_k <= 0:
        top_k = len(dataset)
        
    # Retrieval pool
    features_matrix = np.stack(dataset[feature_column].values)
    guids = dataset[guid_column].values

    # Queries
    query_features = np.stack(queries[feature_column].values)
    query_guids = queries[guid_column].values

    retrievals = []
    n_col_to_rm = 0
    for i in tqdm(range(len(queries)), desc="Retrieving"):
        similarities = cosine_similarity(query_features[i].reshape(1, -1), features_matrix)[0]
        
        # Exclude self if query is in the dataset and same subject
        if exclude_self:
            self_mask = (dataset[guid_column] == query_guids[i]).values
            similarities[self_mask] = -1  # Zero out self-similarity   
            if np.sum(self_mask) > n_col_to_rm:
                n_col_to_rm = np.sum(self_mask) 
        if exclude_same_subject:
            subject_mask = (dataset["subject"] == queries.iloc[i]["subject"]).values
            similarities[subject_mask] = -1  # Zero out similarities for same subject
            if np.sum(subject_mask) > n_col_to_rm:
                n_col_to_rm = np.sum(subject_mask)

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        row = [query_guids[i]] + guids[top_k_indices].tolist()
        retrievals.append(row)
        
    col_names = ["query"] + [f"top{i+1}" for i in range(top_k)]
    out = pd.DataFrame(retrievals, columns=col_names)
    
    # Remove last n_col_to_rm columns if they correspond to same-subject entries
    if n_col_to_rm > 0:
        out = out.iloc[:, :-n_col_to_rm]
    return out

def evaluate_guid_retrieval_map(retrieval_df: pd.DataFrame, metadata_df: pd.DataFrame, top_k: int = 3, class_column: str = 'class_label') -> dict:
    """
    Compute retrieval metrics (mAP@k, success@k) from top-k GUID retrieval DataFrame.

    Args:
        retrieval_df (pd.DataFrame): Output from get_topk_guid_retrievals().
        metadata_df (pd.DataFrame): Original DataFrame that maps GUID to class.
        top_k (int): Number of top similar entries to retrieve.
        class_column (str): Column containing the class labels.

    Returns:
        dict: Retrieval evaluation metrics including mAP@k, success@k, and number of evaluated queries.
    """
    guid_to_class = metadata_df.set_index("GUID")[class_column].to_dict()

    ap_list = []
    hits_at_least_one = 0
    total_queries = 0

    for _, row in tqdm(retrieval_df.iterrows(), total=len(retrieval_df), ncols=150):
        query_guid = row['query']
        query_class = guid_to_class.get(query_guid, None)
        if query_class is None:
            continue

        # Skip queries if there are fewer than top_k relevant items (excluding the query itself)
        class_records = metadata_df[metadata_df[class_column] == query_class]
        if len(class_records) - 1 < top_k:
            continue

        retrieved_guids = row.values[1:top_k+1]
        retrieved_classes = [guid_to_class.get(g) for g in retrieved_guids]
        valid_classes = [cls for cls in retrieved_classes if cls is not None]

        if not valid_classes:
            continue

        # Compute AP for this query
        hits = 0
        precision_sum = 0.0
        for i, cls in enumerate(valid_classes, start=1):
            if cls == query_class:
                hits += 1
                precision_sum += hits / i
        if hits > 0:
            ap = precision_sum / min(len(class_records) - 1, top_k)  # normalize by number of possible relevant items
            hits_at_least_one += 1
        else:
            ap = 0.0
        ap_list.append(ap)
        total_queries += 1

    mAP_at_k = sum(ap_list) / total_queries if total_queries > 0 else 0.0
    success_at_k = hits_at_least_one / total_queries if total_queries > 0 else 0.0

    return {
        'mAP@k': mAP_at_k,
        'success@k': success_at_k,
        'evaluated_queries': total_queries
    }

def evaluate_bias_by_column(
    retrieval_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    top_k: int = 3,
    class_column: str = 'subject',
    group_by_column: str = 'manufacturer'
) -> list[dict]:
    """
    Evaluates retrieval metrics (precision@k, success@k) grouped by a metadata field (e.g., manufacturer, field_strength).

    Args:
        retrieval_df (pd.DataFrame): Retrieval results from get_topk_guid_retrievals().
        metadata_df (pd.DataFrame): Original metadata containing GUID, subject, and group-by column.
        top_k (int): Number of top similar entries to retrieve.
        class_column (str): The class label used for ground truth (e.g., 'subject').
        group_by_column (str): Metadata field to group by (e.g., 'manufacturer').

    Returns:
        list of dicts: One dict per group with retrieval metrics.
    """
    results = []
    merged = retrieval_df.merge(metadata_df[['GUID', group_by_column]], left_on='query', right_on='GUID')

    for group_val, group_df in merged.groupby(group_by_column):
        sub_retrieval_df = retrieval_df[retrieval_df['query'].isin(group_df['query'])]

        if len(sub_retrieval_df) < top_k:
            continue  # skip tiny groups

        metrics = evaluate_guid_retrieval_map(sub_retrieval_df, metadata_df, top_k=top_k, class_column=class_column)
        metrics[group_by_column] = group_val
        results.append(metrics)

    return results 


