import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns


def compute_umap_embeddings(features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, n_components: int = 2, random_state: int = 42):
    """
    Computes UMAP embeddings from feature data.

    Args:
        features (np.ndarray): Array of shape (N_samples, N_features).
        n_neighbors (int): Size of local neighborhood (affects manifold approximation).
        min_dist (float): Minimum distance between points in the low-dimensional space.
        n_components (int): Dimension of the reduced space (2 or 3).
        random_state (int): Seed for reproducibility.

    Returns:
        np.ndarray: UMAP-transformed embeddings (N_samples, n_components).
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(features)
    return embedding

def plot_projection(embedding: np.ndarray, labels: np.ndarray = None, title: str = "UMAP Embedding", figsize=(8, 6), cmap='tab10'):
    plt.figure(figsize=figsize)
    if labels is not None:
        # Use seaborn to handle label coloring
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette=cmap, s=10, linewidth=0)
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    else:
        # Plot without coloring
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10, c='gray')
    
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()