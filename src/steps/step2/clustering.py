"""
clustering.py - Article clustering using HDBSCAN

This module provides functions for clustering articles based on their embeddings
using the HDBSCAN algorithm.

Exported functions:
- cluster_articles(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3) -> Tuple[np.ndarray, List[int]]
  Performs HDBSCAN clustering on article embeddings
- evaluate_clustering(labels: np.ndarray) -> Dict[str, Any]
  Evaluates clustering quality
- calculate_centroids(labels: np.ndarray, X: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]
  Calculates centroids for each cluster and organizes data
- assign_articles_to_clusters(labels: np.ndarray, article_ids: List[int]) -> Dict[int, List[int]]
  Creates a mapping of cluster IDs to article IDs

Related files:
- src/steps/step2/core.py: Uses these functions for clustering
- src/steps/step2/hotness.py: Uses clustering results for hotness calculation
- src/steps/step2/interpretation.py: Interprets clusters based on clustering results
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import time

# Handle optional dependencies with try/except blocks
try:
    import numpy as np
except ImportError:
    logging.error("numpy is required but not available")
    raise

try:
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning(
        "sklearn not available; some functionality will be limited")
    normalize = None
    SKLEARN_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    logging.warning("hdbscan not available; clustering will not be possible")
    hdbscan = None
    HDBSCAN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


def cluster_articles(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = 'cosine',
    cluster_selection_epsilon: float = 0.5,
    alpha: float = 1.0
) -> Tuple[np.ndarray, List[int]]:
    """
    Cluster articles using HDBSCAN based on their embeddings.

    Args:
        embeddings: Article embedding vectors
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples in neighborhood for core points
        metric: Distance metric to use (using cosine for high-dimensional data)
        cluster_selection_epsilon: Controls whether points are assigned to clusters
        alpha: Affects cluster boundary expansion

    Returns:
        Tuple containing (cluster labels, cluster IDs)
    """
    start_time = time.time()

    logger.info(
        f"Clustering {embeddings.shape[0]} articles with {embeddings.shape[1]} dimensions")

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        prediction_data=True,
        core_dist_n_jobs=-1  # Use all available cores
    )

    # Note: HDBSCAN does not support GPU acceleration
    # All computation is CPU-based regardless of hardware
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


def calculate_centroids(labels: np.ndarray, X: np.ndarray) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """
    Calculate centroids for each cluster and organize data.

    Args:
        labels: Cluster labels from clustering algorithm 
        X: Array of embeddings

    Returns:
        Tuple containing:
        - Dictionary mapping cluster labels to lists of indices
        - Dictionary mapping cluster labels to centroid vectors
    """
    # Initialize dictionaries to store cluster data and centroids
    cluster_data = {}  # Maps labels to indices
    centroids = {}     # Maps labels to centroid vectors

    # Group indices by cluster label
    for i, label in enumerate(labels):
        if label >= 0:  # Skip noise points (-1)
            # Convert numpy.int64 to regular Python int for dictionary key
            label_key = int(label)
            if label_key not in cluster_data:
                cluster_data[label_key] = []
            cluster_data[label_key].append(i)

    # Calculate centroid for each cluster
    for label, indices in cluster_data.items():
        # Get embeddings for this cluster
        cluster_embeddings = X[indices]

        # Calculate mean embedding vector
        centroid = np.mean(cluster_embeddings, axis=0)

        # Normalize centroid to unit length
        normalized_centroid = normalize(centroid.reshape(1, -1), norm='l2')[0]

        # Store normalized centroid as list
        centroids[label] = normalized_centroid.tolist()

    return cluster_data, centroids


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
