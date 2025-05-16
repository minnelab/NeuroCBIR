import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def evaluate_similarity_retrieval(dataset: pd.DataFrame, top_k: int = 3, class_column: str = 'class_label') -> dict:
    """
    Computes precision@k and success@k using cosine similarity, but evaluating retrieval based on class match instead of subject_id.
    
    Args:
        dataset (pd.DataFrame): DataFrame with 'features' and class column (e.g., 'class_label').
        top_k (int): Number of top matches to consider.
        class_column (str): Name of the column representing the class.

    Returns:
        dict: Metrics (precision@k, success@k, number of evaluated queries).
    """
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
        if len(class_records) < top_k:
            continue  # Skip if not enough records of this class

        similarities = cosine_similarity(features, features_matrix)[0]
        similarities[i] = -1  # Exclude self

        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        top_k_classes = dataset.iloc[top_k_indices][class_column].values

        # Evaluation
        hits = np.sum(top_k_classes == query_class)
        if hits > 0:
            hits_at_least_one += 1
        hits_at_k += hits

        total_queries += 1
        total_retrieved += top_k

    return calculate_metrics(hits_at_k, hits_at_least_one, total_queries, total_retrieved)