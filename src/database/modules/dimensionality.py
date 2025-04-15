"""
dimensionality.py - Dimensionality reduction data storage and retrieval

This module provides functions for storing and retrieving dimensionality reduction 
coordinates (UMAP, t-SNE, PCA) for visualizing article embeddings.

Exported functions:
- store_coordinates(conn, article_id: int, coordinates: Dict[str, Any], method: str) -> bool
- batch_store_coordinates(conn, coordinates_data: List[Dict[str, Any]]) -> Dict[str, int]
- get_coordinates(conn, article_id: int, method: Optional[str] = None) -> List[Dict[str, Any]]
- get_all_coordinates(conn, method: str, limit: Optional[int] = None) -> List[Dict[str, Any]]
- delete_coordinates(conn, article_id: int, method: Optional[str] = None) -> bool
- update_coordinates_config(conn, config: Dict[str, Any], method: str) -> bool
- get_coordinates_config(conn, method: str) -> Optional[Dict[str, Any]]

Related modules:
- Connection management from connection.py
- Used by clustering visualization tools
- Works with embeddings in modules/embeddings.py
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values

from src.database.modules.connection import get_connection, release_connection

# Configure logging
logger = logging.getLogger(__name__)


def store_coordinates(conn, article_id: int, coordinates: Dict[str, Any], method: str) -> bool:
    """
    Store dimensionality reduction coordinates for an article.

    Args:
        conn: Database connection
        article_id: ID of the article
        coordinates: Dictionary containing the coordinates and metadata
        method: Dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Check if coordinates already exist for this article and method
        cursor.execute("""
            SELECT id FROM article_coordinates 
            WHERE article_id = %s AND method = %s
        """, (article_id, method))

        existing = cursor.fetchone()

        if existing:
            # Update existing coordinates
            cursor.execute("""
                UPDATE article_coordinates
                SET coordinates = %s,
                    metadata = %s,
                    updated_at = NOW()
                WHERE article_id = %s AND method = %s
                RETURNING id
            """, (
                json.dumps(coordinates.get('coords', [])),
                json.dumps(coordinates.get('metadata', {})),
                article_id,
                method
            ))
        else:
            # Insert new coordinates
            cursor.execute("""
                INSERT INTO article_coordinates (
                    article_id, coordinates, method, metadata, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                RETURNING id
            """, (
                article_id,
                json.dumps(coordinates.get('coords', [])),
                method,
                json.dumps(coordinates.get('metadata', {}))
            ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        if result:
            logger.info(
                f"Stored {method} coordinates for article {article_id}")
        else:
            logger.warning(
                f"Failed to store {method} coordinates for article {article_id}")

        return result

    except Exception as e:
        logger.error(
            f"Error storing {method} coordinates for article {article_id}: {e}")
        conn.rollback()
        return False


def batch_store_coordinates(conn, coordinates_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Store multiple sets of coordinates in batches.

    Args:
        conn: Database connection
        coordinates_data: List of dictionaries containing article_id, coordinates, method
            Each dict should have: article_id, coords, method, and optional metadata

    Returns:
        Dict[str, int]: Dictionary with counts of successful and failed operations
    """
    success_count = 0
    error_count = 0

    try:
        cursor = conn.cursor()

        for data in coordinates_data:
            article_id = data.get('article_id')
            method = data.get('method')
            coords = data.get('coords', [])
            metadata = data.get('metadata', {})

            if not article_id or not method or not coords:
                error_count += 1
                logger.warning(
                    f"Skipping coordinates batch item with missing required fields")
                continue

            try:
                # Check if coordinates already exist
                cursor.execute("""
                    SELECT id FROM article_coordinates 
                    WHERE article_id = %s AND method = %s
                """, (article_id, method))

                existing = cursor.fetchone()

                if existing:
                    # Update existing coordinates
                    cursor.execute("""
                        UPDATE article_coordinates
                        SET coordinates = %s,
                            metadata = %s,
                            updated_at = NOW()
                        WHERE article_id = %s AND method = %s
                    """, (
                        json.dumps(coords),
                        json.dumps(metadata),
                        article_id,
                        method
                    ))
                else:
                    # Insert new coordinates
                    cursor.execute("""
                        INSERT INTO article_coordinates (
                            article_id, coordinates, method, metadata, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, NOW(), NOW())
                    """, (
                        article_id,
                        json.dumps(coords),
                        method,
                        json.dumps(metadata)
                    ))

                success_count += 1

            except Exception as e:
                error_count += 1
                logger.error(
                    f"Error in batch processing coordinates for article {article_id}: {e}")
                # Continue with the next item despite error

        conn.commit()
        cursor.close()

        logger.info(
            f"Batch stored coordinates: {success_count} successful, {error_count} failed")
        return {
            'success_count': success_count,
            'error_count': error_count
        }

    except Exception as e:
        logger.error(f"Error in batch storing coordinates: {e}")
        conn.rollback()
        return {
            'success_count': success_count,
            'error_count': error_count
        }


def get_coordinates(conn, article_id: int, method: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get dimensionality reduction coordinates for an article.

    Args:
        conn: Database connection
        article_id: ID of the article
        method: Optional method filter (e.g., 'umap', 'tsne', 'pca')

    Returns:
        List[Dict[str, Any]]: List of coordinate data for the article
    """
    try:
        cursor = conn.cursor()

        if method:
            cursor.execute("""
                SELECT article_id, coordinates, method, metadata, created_at, updated_at
                FROM article_coordinates
                WHERE article_id = %s AND method = %s
            """, (article_id, method))
        else:
            cursor.execute("""
                SELECT article_id, coordinates, method, metadata, created_at, updated_at
                FROM article_coordinates
                WHERE article_id = %s
            """, (article_id,))

        coordinates_list = []
        for row in cursor.fetchall():
            coords = row[1]
            metadata = row[3]

            # Convert from JSON if stored as string
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except:
                    coords = []

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            coordinates_list.append({
                'article_id': row[0],
                'coords': coords,
                'method': row[2],
                'metadata': metadata,
                'created_at': row[4],
                'updated_at': row[5]
            })

        cursor.close()

        if method:
            logger.info(
                f"Retrieved {len(coordinates_list)} {method} coordinates for article {article_id}")
        else:
            logger.info(
                f"Retrieved {len(coordinates_list)} coordinate sets for article {article_id}")

        return coordinates_list

    except Exception as e:
        if method:
            logger.error(
                f"Error retrieving {method} coordinates for article {article_id}: {e}")
        else:
            logger.error(
                f"Error retrieving coordinates for article {article_id}: {e}")
        return []


def get_all_coordinates(conn, method: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all coordinates for a specific dimensionality reduction method.

    Args:
        conn: Database connection
        method: Dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')
        limit: Optional limit on the number of results

    Returns:
        List[Dict[str, Any]]: List of all coordinate data for the method
    """
    try:
        cursor = conn.cursor()

        query = """
            SELECT ac.article_id, ac.coordinates, ac.metadata, 
                   a.title, a.pub_date, a.cluster_id
            FROM article_coordinates ac
            JOIN articles a ON ac.article_id = a.id
            WHERE ac.method = %s
            ORDER BY a.pub_date DESC
        """

        if limit:
            query += " LIMIT %s"
            cursor.execute(query, (method, limit))
        else:
            cursor.execute(query, (method,))

        coordinates_list = []
        for row in cursor.fetchall():
            coords = row[1]
            metadata = row[2]

            # Convert from JSON if stored as string
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except:
                    coords = []

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            coordinates_list.append({
                'article_id': row[0],
                'coords': coords,
                'metadata': metadata,
                'title': row[3],
                'pub_date': row[4],
                'cluster_id': row[5]
            })

        cursor.close()

        logger.info(f"Retrieved {len(coordinates_list)} {method} coordinates")
        return coordinates_list

    except Exception as e:
        logger.error(f"Error retrieving {method} coordinates: {e}")
        return []


def get_coordinates_by_cluster(conn, cluster_id: int, method: str) -> List[Dict[str, Any]]:
    """
    Get coordinates for all articles in a specific cluster.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster
        method: Dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')

    Returns:
        List[Dict[str, Any]]: List of coordinate data for articles in the cluster
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ac.article_id, ac.coordinates, ac.metadata, 
                   a.title, a.pub_date, a.is_hot
            FROM article_coordinates ac
            JOIN articles a ON ac.article_id = a.id
            WHERE ac.method = %s AND a.cluster_id = %s
        """, (method, cluster_id))

        coordinates_list = []
        for row in cursor.fetchall():
            coords = row[1]
            metadata = row[2]

            # Convert from JSON if stored as string
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except:
                    coords = []

            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            coordinates_list.append({
                'article_id': row[0],
                'coords': coords,
                'metadata': metadata,
                'title': row[3],
                'pub_date': row[4],
                'is_hot': row[5]
            })

        cursor.close()

        logger.info(
            f"Retrieved {len(coordinates_list)} {method} coordinates for cluster {cluster_id}")
        return coordinates_list

    except Exception as e:
        logger.error(
            f"Error retrieving {method} coordinates for cluster {cluster_id}: {e}")
        return []


def delete_coordinates(conn, article_id: int, method: Optional[str] = None) -> bool:
    """
    Delete coordinates for an article.

    Args:
        conn: Database connection
        article_id: ID of the article
        method: Optional method filter (e.g., 'umap', 'tsne', 'pca')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        if method:
            cursor.execute("""
                DELETE FROM article_coordinates
                WHERE article_id = %s AND method = %s
                RETURNING id
            """, (article_id, method))
        else:
            cursor.execute("""
                DELETE FROM article_coordinates
                WHERE article_id = %s
                RETURNING id
            """, (article_id,))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        if result:
            if method:
                logger.info(
                    f"Deleted {method} coordinates for article {article_id}")
            else:
                logger.info(
                    f"Deleted all coordinates for article {article_id}")
        else:
            if method:
                logger.warning(
                    f"No {method} coordinates found for article {article_id}")
            else:
                logger.warning(
                    f"No coordinates found for article {article_id}")

        return result

    except Exception as e:
        if method:
            logger.error(
                f"Error deleting {method} coordinates for article {article_id}: {e}")
        else:
            logger.error(
                f"Error deleting coordinates for article {article_id}: {e}")
        conn.rollback()
        return False


def update_coordinates_config(conn, config: Dict[str, Any], method: str) -> bool:
    """
    Update configuration metadata for a dimensionality reduction method.
    This stores global configuration for the method in a special record.

    Args:
        conn: Database connection
        config: Configuration dictionary with parameters and metadata
        method: Dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Use special article_id -1 to store global configuration
        cursor.execute("""
            SELECT id FROM dim_reduction_config
            WHERE method = %s
        """, (method,))

        existing = cursor.fetchone()

        if existing:
            # Update existing config
            cursor.execute("""
                UPDATE dim_reduction_config
                SET config = %s,
                    updated_at = NOW()
                WHERE method = %s
                RETURNING id
            """, (
                json.dumps(config),
                method
            ))
        else:
            # Insert new config
            cursor.execute("""
                INSERT INTO dim_reduction_config (
                    method, config, created_at, updated_at
                )
                VALUES (%s, %s, NOW(), NOW())
                RETURNING id
            """, (
                method,
                json.dumps(config)
            ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        if result:
            logger.info(f"Updated configuration for {method}")
        else:
            logger.warning(f"Failed to update configuration for {method}")

        return result

    except Exception as e:
        logger.error(f"Error updating configuration for {method}: {e}")
        conn.rollback()
        return False


def get_coordinates_config(conn, method: str) -> Optional[Dict[str, Any]]:
    """
    Get the configuration for a dimensionality reduction method.

    Args:
        conn: Database connection
        method: Dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')

    Returns:
        Dict[str, Any] or None: The configuration data or None if not found
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT config, created_at, updated_at
            FROM dim_reduction_config
            WHERE method = %s
        """, (method,))

        row = cursor.fetchone()
        cursor.close()

        if row:
            config = row[0]

            # Convert from JSON if stored as string
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except:
                    config = {}

            return {
                'method': method,
                'config': config,
                'created_at': row[1],
                'updated_at': row[2]
            }

        return None

    except Exception as e:
        logger.error(f"Error retrieving configuration for {method}: {e}")
        return None


def get_all_coordinates_configs(conn) -> List[Dict[str, Any]]:
    """
    Get all available dimensionality reduction configurations.

    Args:
        conn: Database connection

    Returns:
        List[Dict[str, Any]]: List of all configuration data
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT method, config, created_at, updated_at
            FROM dim_reduction_config
            ORDER BY method
        """)

        configs = []
        for row in cursor.fetchall():
            config = row[1]

            # Convert from JSON if stored as string
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                except:
                    config = {}

            configs.append({
                'method': row[0],
                'config': config,
                'created_at': row[2],
                'updated_at': row[3]
            })

        cursor.close()

        logger.info(
            f"Retrieved {len(configs)} dimensionality reduction configurations")
        return configs

    except Exception as e:
        logger.error(
            f"Error retrieving dimensionality reduction configurations: {e}")
        return []
