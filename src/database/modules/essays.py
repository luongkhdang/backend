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
        essay: The essay data to insert

    Returns:
        int or None: The new essay ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()

        # Extract essay data with defaults
        title = essay.get('title', '')
        content = essay.get('content', '')
        cluster_id = essay.get('cluster_id')
        author = essay.get('author', 'system')
        essay_type = essay.get('type', 'summary')
        metadata = essay.get('metadata', {})

        # Insert the essay
        cursor.execute("""
            INSERT INTO essays (
                title, content, cluster_id, author, type, metadata, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            RETURNING id;
        """, (
            title,
            content,
            cluster_id,
            author,
            essay_type,
            json.dumps(metadata) if metadata else None
        ))

        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        return new_id

    except Exception as e:
        logger.error(f"Error inserting essay: {e}")
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

        # Insert the link
        cursor.execute("""
            INSERT INTO essay_entities (
                essay_id, entity_id
            )
            VALUES (%s, %s)
            ON CONFLICT (essay_id, entity_id) DO NOTHING
            RETURNING id;
        """, (
            essay_id,
            entity_id
        ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        return result

    except Exception as e:
        logger.error(
            f"Error linking essay {essay_id} to entity {entity_id}: {e}")
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

        # Get the essay
        cursor.execute("""
            SELECT id, title, content, cluster_id, author, type, metadata, created_at
            FROM essays
            WHERE id = %s
        """, (essay_id,))

        row = cursor.fetchone()
        cursor.close()

        if not row:
            return None

        # Convert row to dictionary
        return {
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'cluster_id': row[3],
            'author': row[4],
            'type': row[5],
            'metadata': row[6] if row[6] else {},
            'created_at': row[7]
        }

    except Exception as e:
        logger.error(f"Error retrieving essay {essay_id}: {e}")
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

        # Get essays for the cluster
        cursor.execute("""
            SELECT id, title, content, author, type, metadata, created_at
            FROM essays
            WHERE cluster_id = %s
            ORDER BY created_at DESC
        """, (cluster_id,))

        essays = []
        for row in cursor.fetchall():
            essays.append({
                'id': row[0],
                'title': row[1],
                'content': row[2],
                'author': row[3],
                'type': row[4],
                'metadata': row[5] if row[5] else {},
                'created_at': row[6]
            })

        cursor.close()
        return essays

    except Exception as e:
        logger.error(f"Error retrieving essays for cluster {cluster_id}: {e}")
        return []
