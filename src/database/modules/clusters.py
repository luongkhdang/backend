"""
clusters.py - Cluster-related database operations

This module provides functions for managing article clusters in the database,
including creating, updating, and retrieving clusters and their articles.

Exported functions:
- create_cluster(conn, centroid: Optional[List[float]] = None, is_hot: bool = False,
                   article_count: int = 0, metadata: Optional[Dict[str, Any]] = None,
                   hotness_score: Optional[float] = None) -> Optional[int]
  Creates a new cluster
- get_all_clusters(conn) -> List[Dict[str, Any]]
  Gets all clusters from database
- get_cluster_by_id(conn, cluster_id: int) -> Optional[Dict[str, Any]]
  Retrieves a cluster by ID
- update_cluster(conn, cluster_id: int, centroid: Optional[List[float]] = None,
                   is_hot: Optional[bool] = None, article_count: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None, hotness_score: Optional[float] = None) -> bool
  Updates a cluster's metadata
- delete_cluster(conn, cluster_id: int) -> bool
  Deletes a cluster
- get_articles_by_cluster(conn, cluster_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets articles belonging to a specific cluster
- get_hot_clusters(conn, min_articles: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets clusters that contain hot articles
- get_recent_clusters(conn, days: int = 7, min_articles: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets clusters with articles published within the last specified days
- get_largest_clusters(conn, limit: int = 10) -> List[Dict[str, Any]]
  Gets the largest clusters by article count
- delete_clusters_from_today(conn) -> int
  Deletes all clusters created on the current day (Pacific Time)

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for cluster operations
- Articles in modules/articles.py
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Configure logging
logger = logging.getLogger(__name__)


def create_cluster(conn, centroid: Optional[List[float]] = None, is_hot: bool = False,
                   article_count: int = 0, metadata: Optional[Dict[str, Any]] = None,
                   hotness_score: Optional[float] = None) -> Optional[int]:
    """
    Create a new cluster in the database using the defined schema columns.

    Args:
        conn: Database connection
        centroid: Optional centroid vector of the cluster
        is_hot: Whether the cluster is hot
        article_count: Initial article count (defaults to 0)
        metadata: Optional JSONB metadata
        hotness_score: Optional initial hotness score

    Returns:
        int or None: Cluster ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()

        # Insert the cluster, relying on DB default for created_at
        cursor.execute("""
            INSERT INTO clusters (
                centroid, is_hot, article_count, metadata, hotness_score
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            centroid,
            is_hot,
            article_count,
            json.dumps(metadata) if metadata else None,
            hotness_score
        ))

        result = cursor.fetchone()
        conn.commit()
        cursor.close()

        if result:
            cluster_id = result[0]
            logger.info(
                f"Created cluster {cluster_id} with metadata: {metadata}")
            return cluster_id
        else:
            logger.warning("Cluster creation did not return an ID.")
            return None

    except Exception as e:
        logger.error(f"Error creating cluster: {e}", exc_info=True)
        conn.rollback()
        return None


def get_all_clusters(conn) -> List[Dict[str, Any]]:
    """
    Get all clusters from the database.

    Args:
        conn: Database connection

    Returns:
        List[Dict[str, Any]]: List of all clusters
    """
    try:
        cursor = conn.cursor()

        # Select columns based on schema, remove updated_at
        cursor.execute("""
            SELECT c.id, c.centroid, c.is_hot, c.article_count, c.metadata, c.created_at, c.hotness_score,
                   (SELECT COUNT(*) FROM articles WHERE cluster_id = c.id) as current_article_count
            FROM clusters c
            ORDER BY c.created_at DESC, c.id DESC
        """)

        clusters = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            cluster_data = dict(zip(columns, row))
            # Ensure metadata is a dict, default to empty if null/invalid
            raw_metadata = cluster_data.get('metadata')
            if isinstance(raw_metadata, str):
                try:
                    cluster_data['metadata'] = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON metadata for cluster {cluster_data['id']}, using empty dict.")
                    cluster_data['metadata'] = {}
            elif not isinstance(raw_metadata, dict):
                cluster_data['metadata'] = {}

            clusters.append(cluster_data)

        cursor.close()
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving all clusters: {e}", exc_info=True)
        return []


def get_cluster_by_id(conn, cluster_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a cluster by its ID.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster

    Returns:
        Dict[str, Any] or None: The cluster data or None if not found
    """
    try:
        cursor = conn.cursor()

        # Select columns based on schema, remove updated_at
        cursor.execute("""
            SELECT c.id, c.centroid, c.is_hot, c.article_count, c.metadata, c.created_at, c.hotness_score,
                   (SELECT COUNT(*) FROM articles WHERE cluster_id = c.id) as current_article_count
            FROM clusters c
            WHERE c.id = %s
        """, (cluster_id,))

        row = cursor.fetchone()
        cursor.close()

        if row:
            columns = [desc[0] for desc in cursor.description]
            cluster_data = dict(zip(columns, row))
            # Ensure metadata is a dict, default to empty if null/invalid
            raw_metadata = cluster_data.get('metadata')
            if isinstance(raw_metadata, str):
                try:
                    cluster_data['metadata'] = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON metadata for cluster {cluster_data['id']}, using empty dict.")
                    cluster_data['metadata'] = {}
            elif not isinstance(raw_metadata, dict):
                cluster_data['metadata'] = {}

            return cluster_data
        else:
            return None

    except Exception as e:
        logger.error(
            f"Error retrieving cluster {cluster_id}: {e}", exc_info=True)
        return None


def update_cluster(conn, cluster_id: int, centroid: Optional[List[float]] = None,
                   is_hot: Optional[bool] = None, article_count: Optional[int] = None,
                   metadata: Optional[Dict[str, Any]] = None, hotness_score: Optional[float] = None) -> bool:
    """
    Update a cluster's information.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster to update
        centroid: Optional new centroid vector
        is_hot: Optional new hotness status
        article_count: Optional new article count
        metadata: Optional new metadata (will fully replace existing metadata if provided)
        hotness_score: Optional new hotness score

    Returns:
        bool: True if successful, False otherwise
    """
    if all(v is None for v in [centroid, is_hot, article_count, metadata, hotness_score]):
        logger.info(f"No updates provided for cluster {cluster_id}. Skipping.")
        return True  # No update needed is considered success

    try:
        cursor = conn.cursor()

        # Build the update parts dynamically
        update_parts = []
        params = []

        if centroid is not None:
            update_parts.append("centroid = %s")
            params.append(centroid)
        if is_hot is not None:
            update_parts.append("is_hot = %s")
            params.append(is_hot)
        if article_count is not None:
            update_parts.append("article_count = %s")
            params.append(article_count)
        if metadata is not None:
            update_parts.append("metadata = %s")
            params.append(json.dumps(metadata))
        if hotness_score is not None:
            update_parts.append("hotness_score = %s")
            params.append(hotness_score)

        set_clause = ", ".join(update_parts)
        params.append(cluster_id)  # Add cluster_id for WHERE clause

        # Execute update
        cursor.execute(f"""
            UPDATE clusters
            SET {set_clause}
            WHERE id = %s
            RETURNING id;
        """, tuple(params))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        if result:
            logger.info(f"Successfully updated cluster {cluster_id}.")
        else:
            logger.warning(
                f"Cluster {cluster_id} not found or no update occurred.")

        return result

    except Exception as e:
        logger.error(
            f"Error updating cluster {cluster_id}: {e}", exc_info=True)
        conn.rollback()
        return False


def delete_cluster(conn, cluster_id: int) -> bool:
    """
    Delete a cluster.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster to delete

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # First, remove the cluster_id from all articles in this cluster
        cursor.execute("""
            UPDATE articles
            SET cluster_id = NULL
            WHERE cluster_id = %s
        """, (cluster_id,))

        # Then delete the cluster
        cursor.execute("""
            DELETE FROM clusters
            WHERE id = %s
            RETURNING id;
        """, (cluster_id,))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        if result:
            logger.info(f"Deleted cluster {cluster_id}")
        else:
            logger.warning(f"Cluster {cluster_id} not found for deletion")

        return result

    except Exception as e:
        logger.error(f"Error deleting cluster {cluster_id}: {e}")
        conn.rollback()
        return False


def get_articles_by_cluster(conn, cluster_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get articles belonging to a specific cluster.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster
        limit: Optional limit on the number of articles to return

    Returns:
        List[Dict[str, Any]]: List of articles in the cluster (with columns id, scraper_id, title, pub_date, domain, is_hot)
    """
    try:
        cursor = conn.cursor()

        # Select columns that exist in the articles table schema
        query = """
            SELECT id, scraper_id, title, pub_date, domain, is_hot 
            FROM articles
            WHERE cluster_id = %s
            ORDER BY pub_date DESC
        """

        params = [cluster_id]
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        cursor.execute(query, tuple(params))

        articles = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            articles.append(dict(zip(columns, row)))

        cursor.close()

        logger.debug(
            f"Retrieved {len(articles)} articles for cluster {cluster_id}")
        return articles

    except Exception as e:
        logger.error(
            f"Error retrieving articles for cluster {cluster_id}: {e}", exc_info=True)
        return []


def get_hot_clusters(conn, min_articles: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get clusters that contain hot articles.

    Args:
        conn: Database connection
        min_articles: Minimum number of articles in the cluster
        limit: Optional limit on the number of clusters to return

    Returns:
        List[Dict[str, Any]]: List of hot clusters with id, metadata, article_count, hot_article_count
    """
    try:
        cursor = conn.cursor()

        # Get clusters with hot articles and article counts
        # Read metadata instead of non-existent name/description columns
        query = """
            SELECT c.id, c.metadata, 
                   COUNT(a.id) as article_count,
                   SUM(CASE WHEN a.is_hot THEN 1 ELSE 0 END) as hot_article_count
            FROM clusters c
            JOIN articles a ON a.cluster_id = c.id
            GROUP BY c.id, c.metadata
            HAVING COUNT(a.id) >= %s AND SUM(CASE WHEN a.is_hot THEN 1 ELSE 0 END) > 0
            ORDER BY hot_article_count DESC, article_count DESC
        """

        params = [min_articles]
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        cursor.execute(query, tuple(params))

        clusters = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            cluster_data = dict(zip(columns, row))
            # Process metadata (similar to get_all_clusters)
            raw_metadata = cluster_data.get('metadata')
            if isinstance(raw_metadata, str):
                try:
                    cluster_data['metadata'] = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    cluster_data['metadata'] = {}
            elif not isinstance(raw_metadata, dict):
                cluster_data['metadata'] = {}
            clusters.append(cluster_data)

        cursor.close()

        logger.info(f"Retrieved {len(clusters)} hot clusters")
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving hot clusters: {e}", exc_info=True)
        return []


def get_recent_clusters(conn, days: int = 7, min_articles: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get clusters with articles published within the last specified days.

    Args:
        conn: Database connection
        days: Number of days to look back
        min_articles: Minimum number of articles in the cluster
        limit: Optional limit on the number of clusters to return

    Returns:
        List[Dict[str, Any]]: List of recent clusters with id, metadata, article_count, latest_pub_date
    """
    try:
        cursor = conn.cursor()

        # Get clusters with recent articles
        # Read metadata instead of non-existent name/description columns
        query = """
            SELECT c.id, c.metadata, 
                   COUNT(a.id) as article_count,
                   MAX(a.pub_date) as latest_pub_date
            FROM clusters c
            JOIN articles a ON a.cluster_id = c.id
            WHERE a.pub_date >= NOW() - INTERVAL '%s days'
            GROUP BY c.id, c.metadata
            HAVING COUNT(a.id) >= %s
            ORDER BY latest_pub_date DESC, article_count DESC
        """

        params = [days, min_articles]
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        cursor.execute(query, tuple(params))

        clusters = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            cluster_data = dict(zip(columns, row))
            # Process metadata (similar to get_all_clusters)
            raw_metadata = cluster_data.get('metadata')
            if isinstance(raw_metadata, str):
                try:
                    cluster_data['metadata'] = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    cluster_data['metadata'] = {}
            elif not isinstance(raw_metadata, dict):
                cluster_data['metadata'] = {}
            clusters.append(cluster_data)

        cursor.close()

        logger.info(
            f"Retrieved {len(clusters)} recent clusters from the last {days} days")
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving recent clusters: {e}", exc_info=True)
        return []


def get_largest_clusters(conn, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get the largest clusters by article count.

    Args:
        conn: Database connection
        limit: Number of clusters to return

    Returns:
        List[Dict[str, Any]]: List of the largest clusters with id, metadata, article_count, oldest_pub_date, newest_pub_date
    """
    try:
        cursor = conn.cursor()

        # Read metadata instead of non-existent name/description columns
        cursor.execute("""
            SELECT c.id, c.metadata, 
                   COUNT(a.id) as article_count,
                   MIN(a.pub_date) as oldest_pub_date,
                   MAX(a.pub_date) as newest_pub_date
            FROM clusters c
            JOIN articles a ON a.cluster_id = c.id
            GROUP BY c.id, c.metadata
            ORDER BY article_count DESC
            LIMIT %s
        """, (limit,))

        clusters = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            cluster_data = dict(zip(columns, row))
            # Process metadata (similar to get_all_clusters)
            raw_metadata = cluster_data.get('metadata')
            if isinstance(raw_metadata, str):
                try:
                    cluster_data['metadata'] = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    cluster_data['metadata'] = {}
            elif not isinstance(raw_metadata, dict):
                cluster_data['metadata'] = {}
            clusters.append(cluster_data)

        cursor.close()

        logger.info(f"Retrieved {len(clusters)} largest clusters")
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving largest clusters: {e}", exc_info=True)
        return []


def delete_clusters_from_today(conn) -> int:
    """
    Delete all clusters created on the current day (Pacific Time).

    This ensures we only maintain one set of clusters per day.

    Args:
        conn: Database connection

    Returns:
        int: Number of clusters deleted
    """
    try:
        cursor = conn.cursor()

        # Calculate the current date in Pacific Time
        pacific_tz = ZoneInfo('America/Los_Angeles')
        today_pacific = datetime.now(pacific_tz).date()

        # First, update all articles to remove cluster_id reference for clusters created today
        cursor.execute("""
            UPDATE articles
            SET cluster_id = NULL
            WHERE cluster_id IN (
                SELECT id FROM clusters 
                WHERE created_at = %s
            )
        """, (today_pacific,))

        # Then delete the clusters created today
        cursor.execute("""
            DELETE FROM clusters
            WHERE created_at = %s
            RETURNING id;
        """, (today_pacific,))

        deleted_rows = cursor.rowcount
        conn.commit()
        cursor.close()

        logger.info(
            f"Deleted {deleted_rows} clusters created today in Pacific Time")
        return deleted_rows

    except Exception as e:
        logger.error(f"Error deleting today's clusters: {e}")
        conn.rollback()
        return 0
