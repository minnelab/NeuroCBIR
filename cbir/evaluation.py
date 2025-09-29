import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

def calculate_metrics(hits_at_k: int, hits_at_least_one: int, total_queries: int, total_retrieved: int) -> dict:
    """
    Computes precision@k and success@k.
    """
    precision_at_k = hits_at_k / total_retrieved if total_retrieved else 0
    success_at_k = hits_at_least_one / total_queries if total_queries else 0

    return {
        'precision@k': precision_at_k,
        'success@k': success_at_k,
        'num_evaluated': total_queries
    }

def evaluate_similarity_retrieval(dataset: pd.DataFrame, top_k: int = 3, class_column: str = 'class_label', evaluation_function=None) -> dict:
    """
    Computes precision@k and success@k using cosine similarity, but evaluating retrieval based on class match instead of subject_id.
    
    Args:
        dataset (pd.DataFrame): DataFrame with 'features' and class column (e.g., 'class_label').
        top_k (int): Number of top matches to consider.
        class_column (str): Name of the column representing the class.

    Returns:
        dict: Metrics (precision@k, success@k, number of evaluated queries).
    """
    if evaluation_function is None:
        evaluation_function = evaluation_function_classification

    hits_at_k = 0
    hits_at_least_one = 0
    total_queries = 0
    total_retrieved = 0

    features_matrix = np.stack(dataset['features'].values)

    for i in range(len(dataset)):
        query = dataset.iloc[i]
        query_class = query[class_column]
        features = query['features'].reshape(1, -1)

        # Check if there are enough records of this class
        class_records = dataset[dataset[class_column] == query_class]
        if len(class_records)-1 < top_k:
            continue  # Skip if not enough records of this class

        similarities = cosine_similarity(features, features_matrix)[0]
        similarities[i] = -1  # Exclude self

        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        top_k_classes = dataset.iloc[top_k_indices][class_column].values

        # Evaluation
        hits = evaluation_function(query_class, top_k_classes)
        if hits > 0:
            hits_at_least_one += 1
        hits_at_k += hits

        total_queries += 1
        total_retrieved += top_k

    return calculate_metrics(hits_at_k, hits_at_least_one, total_queries, total_retrieved)

def evaluation_function_classification(query_class, top_k_classes):
    hits = np.sum(top_k_classes == query_class)
    return hits

def evaluation_function_regression(query_class, top_k_classes):
    hits = np.sum(top_k_classes <= query_class)
    return hits

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

def evaluate_guid_retrieval(retrieval_df: pd.DataFrame, metadata_df: pd.DataFrame, top_k: int = 3, class_column: str = 'class_label') -> dict:
    """
    Compute retrieval metrics (precision@k, success@k) from top-k GUID retrieval DataFrame.

    Args:
        retrieval_df (pd.DataFrame): Output from get_topk_guid_retrievals().
        metadata_df (pd.DataFrame): Original DataFrame that maps GUID to class.
        top_k (int): Number of top similar entries to retrieve.
        class_column (str): Column containing the class labels.

    Returns:
        dict: Retrieval evaluation metrics.
    """
    guid_to_class = metadata_df.set_index("GUID")[class_column].to_dict()

    hits_at_k = 0
    hits_at_least_one = 0
    total_queries = 0

    for _, row in retrieval_df.iterrows():
        query_guid = row['query']
        query_class = guid_to_class.get(query_guid, None)

        if query_class is None:
            continue

        # Check if there are enough records of this class
        class_records = metadata_df[metadata_df[class_column] == query_class]
        if len(class_records)-1 < top_k:
            continue  # Skip if not enough records of this class

        retrieved_guids = row.values[1:top_k+1]
        retrieved_classes = [guid_to_class.get(g) for g in retrieved_guids]

        valid_classes = [cls for cls in retrieved_classes if cls is not None]

        if not valid_classes:
            continue

        hits = sum([1 for cls in valid_classes if cls == query_class])
        if hits > 0:
            hits_at_least_one += 1
        hits_at_k += hits
        total_queries += 1

    precision_at_k = hits_at_k / (total_queries * top_k) if total_queries > 0 else 0.0
    success_at_k = hits_at_least_one / total_queries if total_queries > 0 else 0.0

    return {
        'precision@k': precision_at_k,
        'success@k': success_at_k,
        'evaluated_queries': total_queries
    }

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


