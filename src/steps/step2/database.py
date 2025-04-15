"""
database.py - Database operations for clustering

This module contains functions for interacting with the database
to store clustering results and update article cluster assignments.

Exported functions:
- insert_clusters(reader_client: ReaderDBClient, centroids: Dict[int, List[float]], 
                 cluster_hotness_map: Dict[int, bool]) -> Dict[int, int]
  Inserts clusters into the database with hotness values
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

from src.database.reader_db_client import ReaderDBClient

# Configure logging
logger = logging.getLogger(__name__)


def insert_clusters(
    reader_client: ReaderDBClient,
    centroids: Dict[int, List[float]],
    cluster_hotness_map: Dict[int, bool]
) -> Dict[int, int]:
    """
    Insert clusters into the database with hotness value from calculated map.

    Args:
        reader_client: Initialized ReaderDBClient
        centroids: Dictionary mapping cluster labels to centroid vectors
        cluster_hotness_map: Dictionary mapping cluster labels to is_hot status

    Returns:
        Dictionary mapping cluster labels to database cluster IDs
    """
    cluster_db_map = {}

    for label, centroid in centroids.items():
        # Convert numpy types to native Python types if needed
        label_key = int(label) if hasattr(label, 'item') else label

        # Get the is_hot value for this cluster from the hotness map
        # or default to False if not in map
        is_hot = cluster_hotness_map.get(label_key, False)

        # Insert cluster into database
        cluster_id = reader_client.insert_cluster(
            centroid=centroid,
            is_hot=is_hot
        )

        if cluster_id:
            cluster_db_map[label_key] = cluster_id
            logger.debug(
                f"Inserted cluster {label_key} as DB ID {cluster_id} (hot: {is_hot})")

    logger.info(f"Inserted {len(cluster_db_map)} clusters into database")

    # Log how many hot clusters
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

    # Prepare data for update
    update_data = []
    for article_id, label in zip(article_ids, labels):
        label_int = int(label) if hasattr(label, 'item') else label

        if label_int >= 0:
            cluster_id = cluster_db_map.get(label_int)
            is_hot = cluster_hotness_map.get(label_int, False)
        else:
            cluster_id = None  # Use None for SQL NULL
            is_hot = False

        update_data.append((article_id, cluster_id, is_hot))

    conn = None
    try:
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        # Prepare lists for parameterized query
        batch_article_ids = [d[0] for d in update_data]
        batch_cluster_ids = [d[1] for d in update_data]
        batch_is_hot = [d[2] for d in update_data]

        # Use parameterized query with unnest
        query = """
        UPDATE articles
        SET cluster_id = data.cluster_id,
            is_hot = data.is_hot
        FROM unnest(%s::int[], %s::int[], %s::boolean[]) AS data(article_id, cluster_id, is_hot)
        WHERE articles.id = data.article_id;
        """

        cursor.execute(query, (batch_article_ids,
                       batch_cluster_ids, batch_is_hot))
        updated_rows = cursor.rowcount  # Get number of rows updated
        conn.commit()

        success_count = updated_rows
        failure_count = len(update_data) - updated_rows

        logger.info(
            f"Updated {success_count} article cluster assignments (failed: {failure_count})")
        return {"success": success_count, "failure": failure_count}

    except Exception as e:
        logger.error(
            f"Failed to update cluster assignments: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return {"success": 0, "failure": len(update_data)}
    finally:
        if conn:
            reader_client.release_connection(conn)
