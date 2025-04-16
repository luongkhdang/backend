"""
essays.py - Essay-related database operations

This module provides functions for managing essays in the database,
including inserting, linking, and retrieving essays.

Exported functions:
- insert_essay(conn, essay: Dict[str, Any]) -> Optional[int]
  Inserts an essay into the database
- link_essay_entity(conn, essay_id: int, entity_id: int) -> bool
  Links an essay to an entity
- get_essay_by_id(conn, essay_id: int) -> Optional[Dict[str, Any]]
  Gets an essay by its ID
- get_essays_by_cluster(conn, cluster_id: int) -> List[Dict[str, Any]]
  Gets essays associated with a cluster

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for essay operations
- Entities in modules/entities.py
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def insert_essay(conn, essay: Dict[str, Any]) -> Optional[int]:
    """
    Insert an essay into the database.

    Args:
        conn: Database connection
        essay: The essay data to insert (expects 'type', 'title', 'content', 
               optionally 'article_id', 'layer_depth', 'cluster_id', 'tags')

    Returns:
        int or None: The new essay ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()

        # Extract essay data using correct schema fields
        essay_type = essay.get('type', 'unknown')  # Required
        title = essay.get('title', '')  # Required, default empty
        content = essay.get('content', '')  # Required, default empty
        article_id = essay.get('article_id')  # Optional
        layer_depth = essay.get('layer_depth')  # Optional
        cluster_id = essay.get('cluster_id')  # Optional
        tags = essay.get('tags')  # Optional (list of strings)

        # Insert the essay using correct columns
        cursor.execute("""
            INSERT INTO essays (
                type, article_id, title, content, layer_depth, cluster_id, tags 
                -- created_at uses DEFAULT CURRENT_TIMESTAMP
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """, (
            essay_type,
            article_id,
            title,
            content,
            layer_depth,
            cluster_id,
            tags  # Pass list directly, psycopg2 handles TEXT[]
        ))

        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(
            f"Successfully inserted essay {new_id} of type '{essay_type}'.")
        return new_id

    except Exception as e:
        logger.error(f"Error inserting essay: {e}", exc_info=True)
        conn.rollback()
        return None


def link_essay_entity(conn, essay_id: int, entity_id: int) -> bool:
    """
    Link an essay to an entity.

    Args:
        conn: Database connection
        essay_id: ID of the essay
        entity_id: ID of the entity

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Insert the link, do nothing on conflict. No RETURNING clause.
        cursor.execute("""
            INSERT INTO essay_entities (
                essay_id, entity_id
            )
            VALUES (%s, %s)
            ON CONFLICT (essay_id, entity_id) DO NOTHING;
        """, (
            essay_id,
            entity_id
        ))

        # Assume success if no exception is raised.
        # cursor.rowcount might be 0 if conflict occurred, but that's not an error here.
        conn.commit()
        cursor.close()
        logger.debug(
            f"Attempted to link essay {essay_id} to entity {entity_id}.")
        return True

    except Exception as e:
        logger.error(
            f"Error linking essay {essay_id} to entity {entity_id}: {e}", exc_info=True)
        conn.rollback()
        return False


def get_essay_by_id(conn, essay_id: int) -> Optional[Dict[str, Any]]:
    """
    Get an essay by its ID.

    Args:
        conn: Database connection
        essay_id: ID of the essay

    Returns:
        Dict[str, Any] or None: The essay data, or None if not found
    """
    try:
        cursor = conn.cursor()

        # Get the essay using correct schema columns
        cursor.execute("""
            SELECT id, type, article_id, title, content, layer_depth, cluster_id, created_at, tags
            FROM essays
            WHERE id = %s
        """, (essay_id,))

        row = cursor.fetchone()

        if not row:
            cursor.close()
            return None

        # Convert row to dictionary using column names
        columns = [desc[0] for desc in cursor.description]
        essay_data = dict(zip(columns, row))
        cursor.close()

        return essay_data

    except Exception as e:
        logger.error(f"Error retrieving essay {essay_id}: {e}", exc_info=True)
        return None


def get_essays_by_cluster(conn, cluster_id: int) -> List[Dict[str, Any]]:
    """
    Get all essays associated with a cluster.

    Args:
        conn: Database connection
        cluster_id: ID of the cluster

    Returns:
        List[Dict[str, Any]]: List of essays for the cluster
    """
    try:
        cursor = conn.cursor()

        # Get essays for the cluster using correct schema columns
        cursor.execute("""
            SELECT id, type, article_id, title, content, layer_depth, cluster_id, created_at, tags
            FROM essays
            WHERE cluster_id = %s
            ORDER BY created_at DESC
        """, (cluster_id,))

        essays = []
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            essays.append(dict(zip(columns, row)))

        cursor.close()
        return essays

    except Exception as e:
        logger.error(
            f"Error retrieving essays for cluster {cluster_id}: {e}", exc_info=True)
        return []
