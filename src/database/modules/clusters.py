"""
clusters.py - Cluster-related database operations

This module provides functions for managing article clusters in the database,
including creating, updating, and retrieving clusters and their articles.

Exported functions:
- create_cluster(conn, name: str, description: Optional[str] = None) -> Optional[int]
  Creates a new cluster
- get_all_clusters(conn) -> List[Dict[str, Any]]
  Gets all clusters from database
- get_cluster_by_id(conn, cluster_id: int) -> Optional[Dict[str, Any]]
  Retrieves a cluster by ID
- update_cluster(conn, cluster_id: int, name: str = None, description: Optional[str] = None,
                centroid: Optional[List[float]] = None, is_hot: Optional[bool] = None) -> bool
  Updates a cluster's metadata
- delete_cluster(conn, cluster_id: int) -> bool
  Deletes a cluster
- get_articles_by_cluster(conn, cluster_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets articles belonging to a specific cluster
- update_cluster_metrics(conn, cluster_id: int, metrics: Dict[str, Any]) -> bool
  Updates metrics for a cluster

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for cluster operations
- Articles in modules/articles.py
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def create_cluster(conn, name: str = None, description: Optional[str] = None,
                   centroid: Optional[List[float]] = None, is_hot: bool = False) -> Optional[int]:
    """
    Create a new cluster with created_at timestamp in Pacific Time (day-month-year format).

    Args:
        conn: Database connection
        name: Name of the cluster (will be stored in metadata)
        description: Optional description of the cluster (will be stored in metadata)
        centroid: Optional centroid vector of the cluster
        is_hot: Whether the cluster is hot

    Returns:
        int or None: ID of the created cluster, or None if failed
    """
    try:
        cursor = conn.cursor()

        # Store name and description in metadata since the actual schema doesn't have these columns
        metadata = {}
        if name:
            metadata['name'] = name
        if description:
            metadata['description'] = description

        # Add current date in Pacific Time with day-month-year format to metadata
        # This is for display purposes - the actual created_at field is used for queries
        cursor.execute(
            "SELECT TO_CHAR(NOW() AT TIME ZONE 'America/Los_Angeles', 'DD-MM-YYYY')")
        pacific_date = cursor.fetchone()[0]
        metadata['pacific_date'] = pacific_date

        # Use default value for article_count
        article_count = 0

        # Create the cluster with the fields that actually exist in the schema
        # Use Pacific Time for created_at timestamp
        cursor.execute("""
            INSERT INTO clusters (
                centroid, is_hot, article_count, metadata, created_at
            )
            VALUES (
                %s, %s, %s, %s, 
                (NOW() AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')
            )
            RETURNING id;
        """, (
            centroid,
            is_hot,
            article_count,
            json.dumps(metadata) if metadata else None
        ))

        result = cursor.fetchone()
        conn.commit()
        cursor.close()

        if result:
            cluster_id = result[0]
            logger.info(
                f"Created cluster {cluster_id} with Pacific Time date: {pacific_date}")
            return cluster_id

        return None

    except Exception as e:
        logger.error(f"Error creating cluster: {e}")
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

        cursor.execute("""
            SELECT c.id, c.centroid, c.is_hot, c.article_count, c.metadata, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM articles WHERE cluster_id = c.id) as actual_article_count
            FROM clusters c
            ORDER BY c.id
        """)

        clusters = []
        for row in cursor.fetchall():
            metadata = row[4] if row[4] else {}

            # Extract name and description from metadata if they exist
            name = metadata.get('name', f"Cluster-{row[0]}")
            description = metadata.get('description', '')

            clusters.append({
                'id': row[0],
                'centroid': row[1],
                'is_hot': row[2],
                'article_count': row[3],
                'name': name,
                'description': description,
                'metadata': metadata,
                'created_at': row[5],
                'updated_at': row[6],
                'actual_article_count': row[7]
            })

        cursor.close()

        return clusters

    except Exception as e:
        logger.error(f"Error retrieving all clusters: {e}")
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

        cursor.execute("""
            SELECT c.id, c.centroid, c.is_hot, c.article_count, c.metadata, c.created_at, c.updated_at,
                   (SELECT COUNT(*) FROM articles WHERE cluster_id = c.id) as actual_article_count
            FROM clusters c
            WHERE c.id = %s
        """, (cluster_id,))

        row = cursor.fetchone()
        cursor.close()

        if row:
            metadata = row[4] if row[4] else {}

            # Extract name and description from metadata if they exist
            name = metadata.get('name', f"Cluster-{row[0]}")
            description = metadata.get('description', '')

            return {
                'id': row[0],
                'centroid': row[1],
                'is_hot': row[2],
                'article_count': row[3],
                'name': name,
                'description': description,
                'metadata': metadata,
                'created_at': row[5],
                'updated_at': row[6],
                'actual_article_count': row[7]
            }

        return None

    except Exception as e:
        logger.error(f"Error retrieving cluster {cluster_id}: {e}")
        return None


def update_cluster(conn, cluster_id: int, name: str = None, description: Optional[str] = None,
                   centroid: Optional[List[float]] = None, is_hot: Optional[bool] = None) -> bool:
    """
    Update a cluster's information.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster to update
        name: New name for the cluster (stored in metadata)
        description: Optional new description for the cluster (stored in metadata)
        centroid: Optional new centroid vector
        is_hot: Optional new hotness status

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # First get existing metadata to update
        cursor.execute(
            "SELECT metadata FROM clusters WHERE id = %s", (cluster_id,))
        result = cursor.fetchone()

        if not result:
            logger.warning(f"Cluster {cluster_id} not found for update")
            return False

        # Get existing metadata or create empty dict
        metadata = result[0] if result[0] else {}

        # Update metadata with new values if provided
        if name is not None:
            metadata['name'] = name
        if description is not None:
            metadata['description'] = description

        # Build the update parts dynamically
        update_parts = []
        params = []

        if centroid is not None:
            update_parts.append("centroid = %s")
            params.append(centroid)

        if is_hot is not None:
            update_parts.append("is_hot = %s")
            params.append(is_hot)

        # Always update metadata
        update_parts.append("metadata = %s")
        params.append(json.dumps(metadata))

        # Add the cluster_id as the last parameter
        params.append(cluster_id)

        # Construct and execute the update query
        if update_parts:
            set_clause = ", ".join(update_parts)
            if set_clause:
                query = f"""
                    UPDATE clusters
                    SET {set_clause}
                    WHERE id = %s
                    RETURNING id;
                """

                cursor.execute(query, params)
                result = cursor.fetchone() is not None
                conn.commit()

                if result:
                    logger.info(f"Updated cluster {cluster_id}")
                else:
                    logger.warning(
                        f"Cluster {cluster_id} update did not return ID")
                    result = False
            else:
                result = True

            cursor.close()
            return result
        else:
            cursor.close()
            return True

    except Exception as e:
        logger.error(f"Error updating cluster {cluster_id}: {e}")
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
        List[Dict[str, Any]]: List of articles in the cluster
    """
    try:
        cursor = conn.cursor()

        query = """
            SELECT id, scraper_id, title, url, pub_date, source, is_hot
            FROM articles
            WHERE cluster_id = %s
            ORDER BY pub_date DESC
        """

        if limit:
            query += " LIMIT %s"
            cursor.execute(query, (cluster_id, limit))
        else:
            cursor.execute(query, (cluster_id,))

        articles = []
        for row in cursor.fetchall():
            articles.append({
                'id': row[0],
                'scraper_id': row[1],
                'title': row[2],
                'url': row[3],
                'pub_date': row[4],
                'source': row[5],
                'is_hot': row[6]
            })

        cursor.close()

        logger.info(
            f"Retrieved {len(articles)} articles for cluster {cluster_id}")
        return articles

    except Exception as e:
        logger.error(
            f"Error retrieving articles for cluster {cluster_id}: {e}")
        return []


def update_cluster_metrics(conn, cluster_id: int, metrics: Dict[str, Any]) -> bool:
    """
    Update metrics and metadata for a cluster.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster
        metrics: Dictionary of metrics to store in the metadata field

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # First, get current metadata
        cursor.execute("""
            SELECT metadata
            FROM clusters
            WHERE id = %s
        """, (cluster_id,))

        row = cursor.fetchone()
        if not row:
            logger.warning(
                f"Cluster {cluster_id} not found for metadata update")
            return False

        # Update metadata with new metrics
        current_metadata = row[0] if row[0] else {}

        # If current metadata is a string, convert to dict
        if isinstance(current_metadata, str):
            try:
                current_metadata = json.loads(current_metadata)
            except:
                current_metadata = {}

        # Update with new metrics
        current_metadata.update({
            'metrics': metrics,
            'last_metrics_update': datetime.now().isoformat()
        })

        # Update the cluster
        cursor.execute("""
            UPDATE clusters
            SET metadata = %s,
                updated_at = NOW()
            WHERE id = %s
            RETURNING id;
        """, (
            json.dumps(current_metadata),
            cluster_id
        ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        if result:
            logger.info(f"Updated metrics for cluster {cluster_id}")

        return result

    except Exception as e:
        logger.error(f"Error updating metrics for cluster {cluster_id}: {e}")
        conn.rollback()
        return False


def get_hot_clusters(conn, min_articles: int = 3, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get clusters that contain hot articles.

    Args:
        conn: Database connection
        min_articles: Minimum number of articles in the cluster
        limit: Optional limit on the number of clusters to return

    Returns:
        List[Dict[str, Any]]: List of hot clusters
    """
    try:
        cursor = conn.cursor()

        # Get clusters with hot articles and article counts
        query = """
            SELECT c.id, c.name, c.description, 
                   COUNT(a.id) as article_count,
                   SUM(CASE WHEN a.is_hot THEN 1 ELSE 0 END) as hot_article_count
            FROM clusters c
            JOIN articles a ON a.cluster_id = c.id
            GROUP BY c.id, c.name, c.description
            HAVING COUNT(a.id) >= %s AND SUM(CASE WHEN a.is_hot THEN 1 ELSE 0 END) > 0
            ORDER BY hot_article_count DESC, article_count DESC
        """

        if limit:
            query += " LIMIT %s"
            cursor.execute(query, (min_articles, limit))
        else:
            cursor.execute(query, (min_articles,))

        clusters = []
        for row in cursor.fetchall():
            clusters.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'article_count': row[3],
                'hot_article_count': row[4]
            })

        cursor.close()

        logger.info(f"Retrieved {len(clusters)} hot clusters")
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving hot clusters: {e}")
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
        List[Dict[str, Any]]: List of recent clusters
    """
    try:
        cursor = conn.cursor()

        # Get clusters with recent articles
        query = """
            SELECT c.id, c.name, c.description, 
                   COUNT(a.id) as article_count,
                   MAX(a.pub_date) as latest_pub_date
            FROM clusters c
            JOIN articles a ON a.cluster_id = c.id
            WHERE a.pub_date >= NOW() - INTERVAL '%s days'
            GROUP BY c.id, c.name, c.description
            HAVING COUNT(a.id) >= %s
            ORDER BY latest_pub_date DESC, article_count DESC
        """

        if limit:
            query += " LIMIT %s"
            cursor.execute(query, (days, min_articles, limit))
        else:
            cursor.execute(query, (days, min_articles))

        clusters = []
        for row in cursor.fetchall():
            clusters.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'article_count': row[3],
                'latest_pub_date': row[4]
            })

        cursor.close()

        logger.info(
            f"Retrieved {len(clusters)} recent clusters from the last {days} days")
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving recent clusters: {e}")
        return []


def get_largest_clusters(conn, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get the largest clusters by article count.

    Args:
        conn: Database connection
        limit: Number of clusters to return

    Returns:
        List[Dict[str, Any]]: List of the largest clusters
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT c.id, c.name, c.description, 
                   COUNT(a.id) as article_count,
                   MIN(a.pub_date) as oldest_pub_date,
                   MAX(a.pub_date) as newest_pub_date
            FROM clusters c
            JOIN articles a ON a.cluster_id = c.id
            GROUP BY c.id, c.name, c.description
            ORDER BY article_count DESC
            LIMIT %s
        """, (limit,))

        clusters = []
        for row in cursor.fetchall():
            clusters.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'article_count': row[3],
                'oldest_pub_date': row[4],
                'newest_pub_date': row[5]
            })

        cursor.close()

        logger.info(f"Retrieved {len(clusters)} largest clusters")
        return clusters

    except Exception as e:
        logger.error(f"Error retrieving largest clusters: {e}")
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

        # First, update all articles to remove cluster_id reference
        cursor.execute("""
            UPDATE articles
            SET cluster_id = NULL
            WHERE cluster_id IN (
                SELECT id FROM clusters 
                WHERE DATE(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles') = 
                      DATE(NOW() AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')
            )
        """)

        # Then delete the clusters created today in Pacific time
        cursor.execute("""
            DELETE FROM clusters
            WHERE DATE(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles') = 
                  DATE(NOW() AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')
            RETURNING id;
        """)

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
