"""
database.py - Database operations for clustering

This module contains functions for interacting with the database
to store clustering results and update article cluster assignments.

Exported functions:
- insert_clusters(reader_client: ReaderDBClient, centroids: Dict[int, List[float]], 
                 cluster_hotness_map: Dict[int, bool], cluster_hotness_scores: Dict[int, float], cluster_article_counts: Dict[int, int]) -> Dict[int, int]
  Inserts clusters into the database with centroid, hotness status, hotness score, and article count
- batch_update_article_cluster_assignments(reader_client: ReaderDBClient, 
                                          article_ids: List[int], labels: List[int], cluster_db_map: Dict[int, int]) -> Dict[str, int]
  Updates cluster assignments for multiple articles in batch

Related files:
- src/steps/step2/core.py: Uses these functions to store clustering results
- src/steps/step2/hotness.py: Provides hotness data for cluster storage
- src/database/reader_db_client.py: Database client used for operations
"""

import logging
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from src.database.reader_db_client import ReaderDBClient

# Configure logging
logger = logging.getLogger(__name__)


def insert_clusters(
    reader_client: ReaderDBClient,
    centroids: Dict[int, List[float]],
    cluster_hotness_map: Dict[int, bool],
    cluster_hotness_scores: Dict[int, float],
    cluster_article_counts: Dict[int, int]
) -> Dict[int, int]:
    """
    Insert clusters into the database with centroid, hotness status, hotness score, and article count.

    Args:
        reader_client: Initialized ReaderDBClient
        centroids: Dictionary mapping cluster labels to centroid vectors
        cluster_hotness_map: Dictionary mapping cluster labels to is_hot status
        cluster_hotness_scores: Dictionary mapping cluster labels to numerical hotness scores
        cluster_article_counts: Dictionary mapping cluster labels to article counts

    Returns:
        Dictionary mapping original cluster labels (from HDBSCAN) to database cluster IDs
    """
    cluster_db_map = {}

    for label, centroid in centroids.items():
        # Convert numpy types to native Python types if needed
        label_key = int(label) if hasattr(label, 'item') else label

        # Get the data for this cluster
        is_hot = cluster_hotness_map.get(label_key, False)
        hotness_score = cluster_hotness_scores.get(label_key)
        article_count = cluster_article_counts.get(label_key, 0)

        # Ensure centroid is a list of Python floats if it's a numpy array
        if isinstance(centroid, np.ndarray):
            centroid_list = centroid.tolist()
        else:
            centroid_list = centroid  # Assume it's already a list

        # Insert cluster directly using the updated client method
        cluster_id = reader_client.insert_cluster(
            centroid=centroid_list,
            is_hot=is_hot,
            article_count=article_count,
            hotness_score=hotness_score,
            # Metadata can be added here if needed, e.g., name/description
            # metadata={'name': f'ClusterLabel_{label_key}'}
        )

        if cluster_id:
            cluster_db_map[label_key] = cluster_id
            logger.debug(
                f"Inserted cluster {label_key} as DB ID {cluster_id} (hot: {is_hot}, score: {hotness_score:.4f}, count: {article_count})")
        else:
            logger.warning(f"Failed to insert cluster for label {label_key}")

    logger.info(
        f"Attempted insertion for {len(centroids)} clusters, successfully inserted {len(cluster_db_map)}")

    # Log how many hot clusters were inserted
    hot_count = sum(
        1 for label in cluster_db_map if cluster_hotness_map.get(label, False))
    logger.info(f"{hot_count} clusters marked as 'hot'")

    return cluster_db_map


def batch_update_article_cluster_assignments(
    reader_client: ReaderDBClient,
    article_ids: List[int],
    labels: List[int],
    cluster_db_map: Dict[int, int],
    cluster_hotness_map: Dict[int, bool] = None
) -> Dict[str, int]:
    """
    Update cluster assignments for articles based on cluster labels and set article hotness flags.

    Args:
        reader_client: Initialized ReaderDBClient
        article_ids: List of article database IDs
        labels: List of cluster labels corresponding to article_ids
        cluster_db_map: Dictionary mapping cluster labels to database cluster IDs
        cluster_hotness_map: Dictionary mapping cluster labels to hotness status

    Returns:
        Dictionary with success and failure counts
    """
    # Check if arrays are empty using length instead of boolean evaluation
    if len(article_ids) == 0 or len(labels) == 0:
        return {"success": 0, "failure": 0}

    # Default cluster_hotness_map if not provided
    if cluster_hotness_map is None:
        cluster_hotness_map = {}

    # Prepare data in the format required by batch_update_article_clusters
    # Format: List of tuples (article_id, cluster_id, is_hot)
    assignments = []
    for article_id, label in zip(article_ids, labels):
        label_int = int(label) if hasattr(label, 'item') else label

        if label_int >= 0:
            cluster_id = cluster_db_map.get(label_int)
            is_hot = cluster_hotness_map.get(label_int, False)
        else:
            cluster_id = None  # Use None for SQL NULL
            is_hot = False

        assignments.append((article_id, cluster_id, is_hot))

    try:
        # Use the ReaderDBClient's method to handle the update
        result = reader_client.batch_update_article_clusters(assignments)

        # Convert the result to the expected format if needed
        success_count = result.get("success", 0)
        failure_count = result.get("failure", 0)

        logger.info(
            f"Updated {success_count} article cluster assignments (failed: {failure_count})")
        return {"success": success_count, "failure": failure_count}

    except Exception as e:
        logger.error(
            f"Failed to update cluster assignments: {e}", exc_info=True)
        return {"success": 0, "failure": len(assignments)}
