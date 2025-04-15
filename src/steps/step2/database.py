"""
database.py - Database operations for clustering

This module contains functions for interacting with the database
to store clustering results and update article cluster assignments.

Exported functions:
- insert_clusters(reader_client: ReaderDBClient, cluster_data: Dict[int, Dict[str, Any]], 
                 cluster_hotness_map: Dict[int, bool]) -> Dict[int, int]
  Inserts clusters into the database with hotness values
- batch_update_article_cluster_assignments(reader_client: ReaderDBClient, 
                                          assignments: List[Tuple[int, Optional[int]]]) -> Tuple[int, int]
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
    cluster_data: Dict[int, Dict[str, Any]],
    cluster_hotness_map: Dict[int, bool]
) -> Dict[int, int]:
    """
    Insert clusters into the database with hotness value from calculated map.

    Args:
        reader_client: Initialized ReaderDBClient
        cluster_data: Dictionary of cluster data from calculate_centroids
        cluster_hotness_map: Dictionary mapping HDBSCAN label to is_hot status

    Returns:
        Dictionary mapping HDBSCAN labels to database cluster IDs
    """
    hdbscan_to_db_id_map = {}

    for label, data in cluster_data.items():
        centroid = data["centroid"]
        article_count = data["count"]

        # Get the is_hot value for this cluster from the hotness map
        # or use default based on article count if not in map
        is_hot = cluster_hotness_map.get(
            label,
            article_count >= int(os.getenv("HOT_CLUSTER_THRESHOLD", "20"))
        )

        # Use existing insert_cluster function, which returns cluster ID
        cluster_id = reader_client.insert_cluster(
            centroid=centroid,
            is_hot=is_hot
        )

        if cluster_id:
            hdbscan_to_db_id_map[label] = cluster_id
            logger.debug(
                f"Inserted cluster {label} with {article_count} articles as DB ID {cluster_id} (hot: {is_hot})")

    logger.info(f"Inserted {len(hdbscan_to_db_id_map)} clusters into database")

    # Log how many hot clusters
    hot_count = sum(
        1 for label in hdbscan_to_db_id_map if cluster_hotness_map.get(label, False))
    logger.info(f"{hot_count} clusters marked as 'hot'")

    return hdbscan_to_db_id_map


def batch_update_article_cluster_assignments(
    reader_client: ReaderDBClient,
    assignments: List[Tuple[int, Optional[int]]]
) -> Tuple[int, int]:
    """
    Update cluster assignments for multiple articles in batch.

    Args:
        reader_client: Initialized ReaderDBClient
        assignments: List of tuples (article_id, cluster_id)

    Returns:
        Tuple of (success_count, failure_count)
    """
    if not assignments:
        return 0, 0

    try:
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        # Split assignments into batches
        batch_size = 1000
        success_count = 0
        failure_count = 0

        for i in range(0, len(assignments), batch_size):
            batch = assignments[i:i+batch_size]

            try:
                # Convert batch to SQL-friendly format and handle NULL properly
                values = []
                for article_id, cluster_id in batch:
                    if cluster_id is None:
                        values.append(f"({article_id}, NULL)")
                    else:
                        values.append(f"({article_id}, {cluster_id})")

                values_str = ", ".join(values)

                # SQL query using temporary table for efficient update
                query = f"""
                UPDATE articles
                SET cluster_id = temp.cluster_id
                FROM (VALUES {values_str}) AS temp(article_id, cluster_id)
                WHERE articles.id = temp.article_id
                """

                cursor.execute(query)
                success_count += len(batch)
            except Exception as e:
                logger.error(f"Error updating batch {i//batch_size}: {e}")
                failure_count += len(batch)
                # Continue with next batch rather than failing completely

        conn.commit()
        cursor.close()
        reader_client.release_connection(conn)

        logger.info(
            f"Updated {success_count} article cluster assignments (failed: {failure_count})")
        return success_count, failure_count

    except Exception as e:
        logger.error(
            f"Failed to update cluster assignments: {e}", exc_info=True)
        return 0, len(assignments)
