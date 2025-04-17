"""
entity_snippets.py - Entity snippets database operations

This module provides functions for managing entity snippets in the database,
including storing, retrieving, and managing text snippets related to entities.

Exported functions:
- store_entity_snippet(conn, entity_id: int, article_id: int, snippet: str, is_influential: bool = False) -> bool
  Stores a text snippet associated with an entity and article
- get_entity_snippets(conn, entity_id: int, limit: int = 30) -> List[Dict[str, Any]]
  Retrieves snippets for a specific entity
- get_article_entity_snippets(conn, article_id: int, entity_id: int) -> List[Dict[str, Any]]
  Retrieves snippets for a specific entity in a specific article
- get_article_snippets(conn, article_id: int, limit_per_entity: int = 5) -> Dict[int, List[Dict[str, Any]]]
  Retrieves snippets for all entities in a specific article
- delete_entity_snippets(conn, entity_id: int) -> int
  Deletes all snippets for a specific entity
- delete_article_entity_snippets(conn, article_id: int, entity_id: int) -> int
  Deletes snippets for a specific entity in a specific article

Related modules:
- Connection management from connection.py
- Entities in modules/entities.py
- Articles in modules/articles.py
- Used by ReaderDBClient for entity snippet operations
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import random

# Import specific exception for deadlock detection if using psycopg2
try:
    # Standard psycopg2 error
    from psycopg2.errors import DeadlockDetected
except ImportError:
    # Fallback for other DB drivers or if psycopg2 isn't directly used here
    # Note: This might require adjusting the exception type in the except block
    DeadlockDetected = None

# Configure logging
logger = logging.getLogger(__name__)


def store_entity_snippet(conn, entity_id: int, article_id: int, snippet: str, is_influential: bool = False, max_retries: int = 3) -> bool:
    """
    Store a text snippet associated with an entity and article, with deadlock retry logic.

    Args:
        conn: Database connection
        entity_id: ID of the entity
        article_id: ID of the article
        snippet: Text snippet containing or related to the entity
        is_influential: Whether this snippet is considered influential for the entity
        max_retries: Maximum number of retry attempts for deadlocks (default: 3)

    Returns:
        bool: True if successful, False otherwise
    """
    # Validate snippet outside the loop
    if not snippet or len(snippet.strip()) == 0:
        logger.warning(
            f"Empty snippet for entity {entity_id} in article {article_id}")
        return False

    attempt = 0
    while attempt < max_retries:
        try:
            cursor = conn.cursor()
            # Insert the snippet
            cursor.execute("""
                INSERT INTO entity_snippets (
                    entity_id, article_id, snippet, is_influential, created_at
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                entity_id,
                article_id,
                snippet,
                is_influential,
                datetime.now()
            ))

            result = cursor.fetchone() is not None
            conn.commit()  # Commit successful transaction
            cursor.close()
            return result  # Success, exit loop and function

        except DeadlockDetected as deadlock_err:
            conn.rollback()  # Rollback the failed transaction
            attempt += 1
            if attempt >= max_retries:
                logger.error(
                    f"Deadlock detected storing snippet for entity {entity_id}, article {article_id} after {max_retries} attempts. Giving up. Error: {deadlock_err}")
                return False  # Give up after max retries
            else:
                wait_time = (2 ** attempt) * 0.1 + random.uniform(0.05, 0.1)
                logger.warning(
                    f"Deadlock detected storing snippet for entity {entity_id}, article {article_id}. Attempt {attempt}/{max_retries}. Retrying after {wait_time:.2f}s... Error: {deadlock_err}")
                time.sleep(wait_time)
                # Continue to next iteration of the while loop

        except Exception as e:
            # Handle other unexpected errors
            logger.error(
                f"Unexpected error storing snippet for entity {entity_id} in article {article_id} on attempt {attempt+1}: {e}")
            conn.rollback()  # Rollback on other errors too
            return False  # Exit on other errors

    # Should not be reached if successful or max retries hit, but safety return
    return False


def get_entity_snippets(conn, entity_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Get snippets for a specific entity, ordered by influence and recency.

    Args:
        conn: Database connection
        entity_id: ID of the entity
        limit: Maximum number of snippets to return

    Returns:
        List[Dict[str, Any]]: List of snippet records
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT es.id, es.entity_id, es.article_id, es.snippet, es.is_influential,
                   es.created_at, a.title as article_title, a.pub_date as article_pub_date
            FROM entity_snippets es
            JOIN articles a ON es.article_id = a.id
            WHERE es.entity_id = %s
            ORDER BY es.is_influential DESC, es.created_at DESC
            LIMIT %s;
        """, (entity_id, limit))

        columns = [desc[0] for desc in cursor.description]
        snippets = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return snippets

    except Exception as e:
        logger.error(f"Error retrieving snippets for entity {entity_id}: {e}")
        return []


def get_article_entity_snippets(conn, article_id: int, entity_id: int) -> List[Dict[str, Any]]:
    """
    Get snippets for a specific entity in a specific article.

    Args:
        conn: Database connection
        article_id: ID of the article
        entity_id: ID of the entity

    Returns:
        List[Dict[str, Any]]: List of snippet records
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, entity_id, article_id, snippet, is_influential, created_at
            FROM entity_snippets
            WHERE article_id = %s AND entity_id = %s
            ORDER BY is_influential DESC, created_at DESC;
        """, (article_id, entity_id))

        columns = [desc[0] for desc in cursor.description]
        snippets = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return snippets

    except Exception as e:
        logger.error(
            f"Error retrieving snippets for entity {entity_id} in article {article_id}: {e}")
        return []


def delete_entity_snippets(conn, entity_id: int) -> int:
    """
    Delete all snippets for a specific entity.

    Args:
        conn: Database connection
        entity_id: ID of the entity

    Returns:
        int: Number of deleted snippets
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM entity_snippets
            WHERE entity_id = %s
            RETURNING id;
        """, (entity_id,))

        deleted_count = cursor.rowcount
        conn.commit()
        cursor.close()

        logger.info(f"Deleted {deleted_count} snippets for entity {entity_id}")
        return deleted_count

    except Exception as e:
        logger.error(f"Error deleting snippets for entity {entity_id}: {e}")
        conn.rollback()
        return 0


def delete_article_entity_snippets(conn, article_id: int, entity_id: int) -> int:
    """
    Delete snippets for a specific entity in a specific article.

    Args:
        conn: Database connection
        article_id: ID of the article
        entity_id: ID of the entity

    Returns:
        int: Number of deleted snippets
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM entity_snippets
            WHERE article_id = %s AND entity_id = %s
            RETURNING id;
        """, (article_id, entity_id))

        deleted_count = cursor.rowcount
        conn.commit()
        cursor.close()

        logger.info(
            f"Deleted {deleted_count} snippets for entity {entity_id} in article {article_id}")
        return deleted_count

    except Exception as e:
        logger.error(
            f"Error deleting snippets for entity {entity_id} in article {article_id}: {e}")
        conn.rollback()
        return 0


def get_article_snippets(conn, article_id: int, limit_per_entity: int = 5) -> Dict[int, List[Dict[str, Any]]]:
    """
    Get snippets for all entities in a specific article.

    Args:
        conn: Database connection
        article_id: ID of the article
        limit_per_entity: Maximum number of snippets to return per entity

    Returns:
        Dict[int, List[Dict[str, Any]]]: Dictionary mapping entity IDs to lists of snippet records
    """
    try:
        cursor = conn.cursor()

        # First get all entities in this article
        cursor.execute("""
            SELECT entity_id
            FROM article_entities
            WHERE article_id = %s
        """, (article_id,))

        entity_ids = [row[0] for row in cursor.fetchall()]

        if not entity_ids:
            return {}

        # Get snippets for each entity
        snippets_by_entity = {}

        for entity_id in entity_ids:
            cursor.execute("""
                SELECT id, entity_id, article_id, snippet, is_influential, created_at
                FROM entity_snippets
                WHERE article_id = %s AND entity_id = %s
                ORDER BY is_influential DESC, created_at DESC
                LIMIT %s;
            """, (article_id, entity_id, limit_per_entity))

            columns = [desc[0] for desc in cursor.description]
            entity_snippets = [dict(zip(columns, row))
                               for row in cursor.fetchall()]

            if entity_snippets:
                snippets_by_entity[entity_id] = entity_snippets

        cursor.close()
        return snippets_by_entity

    except Exception as e:
        logger.error(
            f"Error retrieving entity snippets for article {article_id}: {e}")
        return {}
