"""
clustering.py - Article clustering using HDBSCAN

This module provides functions for clustering articles based on their embeddings
using the HDBSCAN algorithm.

Exported functions:
- cluster_articles(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3) -> Tuple[np.ndarray, List[int]]
  Performs HDBSCAN clustering on article embeddings
- preprocess_embeddings(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray
  Reduces dimensionality of embeddings using UMAP
- evaluate_clustering(labels: np.ndarray) -> Dict[str, Any]
  Evaluates clustering quality
- calculate_centroids(X: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, Any]]
  Calculates centroids for each cluster and returns cluster data
- assign_articles_to_clusters(labels: np.ndarray, article_ids: List[int]) -> Dict[int, List[int]]
  Creates a mapping of cluster IDs to article IDs

Related files:
- src/steps/step2/core.py: Uses these functions for clustering
- src/steps/step2/hotness.py: Uses clustering results for hotness calculation
- src/steps/step2/interpretation.py: Interprets clusters based on clustering results
"""

from sklearn.decomposition import PCA
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import hdbscan
from collections import defaultdict, Counter
import time

# Import dimensionality reduction libraries
UMAP_AVAILABLE = False
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    logging.warning(
        "UMAP not available; will use PCA for dimensionality reduction")


# Configure logging
logger = logging.getLogger(__name__)


def preprocess_embeddings(
    embeddings: np.ndarray,
    n_components: int = 50,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce dimensionality of embeddings for faster clustering.

    Args:
        embeddings: Article embedding vectors
        n_components: Number of dimensions to reduce to
        random_state: Random seed for reproducibility

    Returns:
        Reduced dimensionality embeddings
    """
    start_time = time.time()

    # Ensure embeddings are numpy array with appropriate dtype
    embeddings = np.array(embeddings, dtype=np.float32)

    # Cap n_components at embedding dimension - 1
    n_components = min(n_components, embeddings.shape[1] - 1)

    # Use UMAP if available (better quality)
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=n_components,
            metric='cosine',
            n_neighbors=30,
            min_dist=0.1,
            random_state=random_state
        )
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(
            f"UMAP dimensionality reduction completed in {time.time() - start_time:.2f}s")
    else:
        # Fall back to PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        logger.info(
            f"PCA dimensionality reduction completed in {time.time() - start_time:.2f}s")

    return reduced_embeddings


def cluster_articles(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = 'euclidean',
    cluster_selection_epsilon: float = 0.5,
    alpha: float = 1.0,
    use_dimensionality_reduction: bool = True
) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster articles using HDBSCAN based on their embeddings.

    Args:
        embeddings: Article embedding vectors
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples in neighborhood for core points
        metric: Distance metric to use
        cluster_selection_epsilon: Controls whether points are assigned to clusters
        alpha: Affects cluster boundary expansion
        use_dimensionality_reduction: Whether to reduce dimensions before clustering

    Returns:
        Tuple containing (cluster labels, cluster IDs)
    """
    start_time = time.time()

    # Preprocess embeddings if needed
    if use_dimensionality_reduction and embeddings.shape[1] > 50:
        embeddings = preprocess_embeddings(embeddings)

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        prediction_data=True
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    # Get unique cluster IDs (excluding noise which is -1)
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])

    # Log clustering results
    n_clusters = len(unique_clusters)
    n_noise = np.sum(cluster_labels == -1)
    logger.info(
        f"HDBSCAN clustering completed in {time.time() - start_time:.2f}s")
    logger.info(f"Found {n_clusters} clusters with {n_noise} noise points " +
                f"({n_noise/len(cluster_labels):.1%} of all articles)")

    return cluster_labels, unique_clusters


def evaluate_clustering(labels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate clustering quality and provide statistics.

    Args:
        labels: Cluster labels from clustering algorithm

    Returns:
        Dictionary of clustering quality metrics
    """
    # Count articles per cluster
    cluster_counts = Counter(labels)

    # Remove noise (-1) from statistics
    if -1 in cluster_counts:
        noise_count = cluster_counts.pop(-1)
    else:
        noise_count = 0

    # Calculate statistics
    cluster_sizes = list(cluster_counts.values())

    if not cluster_sizes:
        return {
            "num_clusters": 0,
            "noise_points": noise_count,
            "noise_percentage": 100.0 if labels.size > 0 else 0.0,
            "avg_cluster_size": 0,
            "median_cluster_size": 0,
            "min_cluster_size": 0,
            "max_cluster_size": 0
        }

    stats = {
        "num_clusters": len(cluster_counts),
        "noise_points": noise_count,
        "noise_percentage": 100 * noise_count / len(labels) if len(labels) > 0 else 0,
        "avg_cluster_size": np.mean(cluster_sizes),
        "median_cluster_size": np.median(cluster_sizes),
        "min_cluster_size": min(cluster_sizes),
        "max_cluster_size": max(cluster_sizes),
        "cluster_size_distribution": dict(sorted(cluster_counts.items())),
    }

    return stats


def calculate_centroids(X: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Calculate centroids for each cluster.

    Args:
        X: Array of embeddings
        labels: Cluster labels from HDBSCAN

    Returns:
        Dictionary mapping cluster labels to cluster data (centroid, count, indices)
    """
    from sklearn.preprocessing import normalize

    # Initialize dictionary to store cluster data
    cluster_data = {}

    # Group indices by cluster label
    for i, label in enumerate(labels):
        if label >= 0:  # Skip noise points (-1)
            if label not in cluster_data:
                cluster_data[label] = {
                    "indices": [],
                    "centroid": None,
                    "count": 0
                }
            cluster_data[label]["indices"].append(i)

    # Calculate centroid for each cluster
    for label, data in cluster_data.items():
        indices = data["indices"]
        cluster_embeddings = X[indices]

        # Calculate mean embedding vector
        centroid = np.mean(cluster_embeddings, axis=0)

        # Normalize centroid to unit length
        normalized_centroid = normalize(centroid.reshape(1, -1), norm='l2')[0]

        # Store data
        data["centroid"] = normalized_centroid.tolist()
        data["count"] = len(indices)

    return cluster_data


def assign_articles_to_clusters(
    labels: np.ndarray,
    article_ids: List[int]
) -> Dict[int, List[int]]:
    """
    Create a mapping of cluster IDs to article IDs.

    Args:
        labels: Cluster labels from clustering algorithm
        article_ids: List of article IDs corresponding to the labels

    Returns:
        Dictionary mapping cluster IDs to lists of article IDs
    """
    if len(labels) != len(article_ids):
        raise ValueError(
            f"Length mismatch: {len(labels)} labels vs {len(article_ids)} article IDs")

    cluster_to_articles = defaultdict(list)

    for idx, cluster_id in enumerate(labels):
        # Skip noise points (cluster_id = -1)
        if cluster_id >= 0:
            cluster_to_articles[int(cluster_id)].append(article_ids[idx])

    return dict(cluster_to_articles)
