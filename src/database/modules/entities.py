"""
entities.py - Entity-related database operations

This module provides functions for managing entities in the database,
including inserting, linking, and retrieving entities.

Exported functions:
- insert_entity(conn, entity: Dict[str, Any]) -> Optional[int]
  Inserts an entity into the database
- link_article_entity(conn, article_id: int, entity_id: int, mention_count: int = 1) -> bool
  Links an article to an entity with a mention count
- get_entities_by_influence(conn, limit: int = 20) -> List[Dict[str, Any]]
  Gets entities sorted by influence score
- get_entity_influence_for_articles(conn, article_ids: List[int]) -> Dict[int, float]
  Gets entity influence scores for a list of articles
- find_or_create_entity(conn, name: str, entity_type: str) -> Optional[int]
  Finds an entity by name or creates it if it doesn't exist
- increment_entity_mentions(conn, article_id: int, entity_id: int, count: int = 1) -> bool
  Increments the mention count for an entity in an article
- increment_global_entity_mentions(conn, entity_id: int, count: int = 1) -> bool
  Increments the global mentions count for an entity in the entities table
- article_contains_entity(conn, article_id: int, entity_id: int) -> bool
  Checks if an article contains a specific entity
- get_article_entities(conn, article_id: int) -> List[Dict[str, Any]]
  Retrieves all entities for a specific article with their mention counts
- get_articles_by_entity(conn, entity_id: int, limit: int = 50) -> List[Dict[str, Any]]
  Retrieves all articles that contain a specific entity
- get_entities_by_type(conn, entity_type: str, limit: int = 50) -> List[Dict[str, Any]]
  Retrieves entities filtered by a specific type
- get_related_entities(conn, entity_id: int, limit: int = 20) -> List[Dict[str, Any]]
  Retrieves entities that frequently appear in articles with the given entity

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for entity operations
- Articles in modules/articles.py
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def insert_entity(conn, entity: Dict[str, Any]) -> Optional[int]:
    """
    Insert an entity into the database or update limited fields on conflict.

    Args:
        conn: Database connection
        entity: The entity data to insert (expects 'name', optionally 'entity_type', 'first_seen', 'last_seen')

    Returns:
        int or None: The new or existing entity ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()

        # Extract entity data with defaults
        name = entity.get('name')
        if not name:
            logger.error("Entity name cannot be empty")
            return None
        entity_type = entity.get('entity_type', 'unknown')  # Use entity_type
        first_seen = entity.get('first_seen', datetime.now())
        last_seen = entity.get('last_seen', datetime.now())

        # Insert the entity, handle conflict on name
        # ON CONFLICT, update last_seen if the new one is later
        # Note: influence_score and mentions have defaults in the schema
        cursor.execute("""
            INSERT INTO entities (
                name, entity_type, first_seen, last_seen
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET entity_type = EXCLUDED.entity_type, -- Optionally update type on conflict
                last_seen = GREATEST(entities.last_seen, EXCLUDED.last_seen)
            RETURNING id;
        """, (
            name,
            entity_type,
            first_seen,
            last_seen
        ))

        entity_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        return entity_id

    except Exception as e:
        logger.error(f"Error inserting entity '{name}': {e}", exc_info=True)
        conn.rollback()
        return None


def link_article_entity(conn, article_id: int, entity_id: int, mention_count: int = 1) -> bool:
    """
    Link an article to an entity with a mention count.

    Args:
        conn: Database connection
        article_id: ID of the article
        entity_id: ID of the entity
        mention_count: Number of mentions (default: 1)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Insert or update the link. created_at has default.
        # Removed RETURNING id as it's not in the schema.
        cursor.execute("""
            INSERT INTO article_entities (
                article_id, entity_id, mention_count
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (article_id, entity_id) DO UPDATE
            SET mention_count = article_entities.mention_count + EXCLUDED.mention_count; -- Accumulate mentions on conflict
        """, (
            article_id,
            entity_id,
            mention_count
        ))

        # Check if the operation affected any row
        success = cursor.rowcount > 0
        conn.commit()
        cursor.close()

        # Log success/failure based on rowcount
        if success:
            logger.debug(
                f"Successfully linked article {article_id} to entity {entity_id} with count {mention_count}.")
        else:
            # This might happen if ON CONFLICT DO UPDATE didn't change anything (e.g., mention_count was 0)
            # or if there was an issue not raising an exception but still failing.
            logger.warning(
                f"Linking article {article_id} to entity {entity_id} affected 0 rows.")
            # Consider if rowcount 0 should be treated as success if the link already existed.
            # For now, let's assume rowcount > 0 means success (insert or update occurred).
            pass  # Keep success as potentially False if rowcount is 0

        # Return True if insert/update happened, False otherwise or on error.
        # Re-evaluating: ON CONFLICT DO UPDATE returns rowcount=1 even if no columns changed.
        # We should rely on exceptions for failure cases primarily. Assume success if no exception.
        return True

    except Exception as e:
        logger.error(
            f"Error linking article {article_id} to entity {entity_id}: {e}", exc_info=True)
        conn.rollback()
        return False


def get_entities_by_influence(conn, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get entities sorted by influence score (descending).

    Args:
        conn: Database connection
        limit: Maximum number of entities to return

    Returns:
        List[Dict[str, Any]]: List of entities ordered by influence_score
    """
    try:
        cursor = conn.cursor()
        # Select directly from entities table, order by influence_score
        cursor.execute("""
            SELECT id, name, entity_type, influence_score, mentions, first_seen, last_seen, created_at
            FROM entities
            ORDER BY influence_score DESC NULLS LAST, mentions DESC
            LIMIT %s
        """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        entities = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return entities

    except Exception as e:
        logger.error(
            f"Error retrieving entities by influence: {e}", exc_info=True)
        return []


def get_entity_influence_for_articles(conn, article_ids: List[int]) -> Dict[int, Optional[float]]:
    """
    Gets the average influence score of entities linked to each specified article.

    Args:
        conn: Database connection
        article_ids: List of article IDs

    Returns:
        Dict[int, Optional[float]]: Dictionary mapping article IDs to the average
                                     influence score of their linked entities.
                                     Returns None for articles with no linked entities.
    """
    if not article_ids:
        return {}

    try:
        cursor = conn.cursor()

        # Use tuple for IN clause parameter binding
        article_ids_tuple = tuple(article_ids)

        # Query to calculate average entity influence score per article
        cursor.execute("""
            SELECT ae.article_id, AVG(e.influence_score) as avg_influence_score
            FROM article_entities ae
            JOIN entities e ON ae.entity_id = e.id
            WHERE ae.article_id = ANY(%s) -- Use ANY for array/tuple binding
            GROUP BY ae.article_id
        """, (article_ids_tuple,))  # Pass as a tuple

        # Initialize with None
        influence_scores = {article_id: None for article_id in article_ids}
        for row in cursor.fetchall():
            article_id, avg_score = row
            influence_scores[article_id] = float(
                avg_score) if avg_score is not None else None

        cursor.close()
        return influence_scores

    except Exception as e:
        logger.error(
            f"Error retrieving average entity influence for articles: {e}", exc_info=True)
        # Return dict with Nones for all requested IDs on error
        return {article_id: None for article_id in article_ids}


def find_or_create_entity(conn, name: str, entity_type: str) -> Optional[int]:
    """
    Find an entity by name or create it if it doesn't exist.
    Uses INSERT ... ON CONFLICT DO NOTHING then SELECT.

    Args:
        conn: Database connection
        name: Entity name
        entity_type: Entity type

    Returns:
        int or None: Entity ID or None if error
    """
    entity_id: Optional[int] = None
    try:
        cursor = conn.cursor()

        # Try to insert, do nothing if conflict (name exists)
        # Set first_seen and last_seen on initial creation
        cursor.execute("""
            INSERT INTO entities (name, entity_type, first_seen, last_seen)
            VALUES (%s, %s, NOW(), NOW())
            ON CONFLICT (name) DO NOTHING;
        """, (name, entity_type))

        # Commit the insert attempt (important before select)
        conn.commit()

        # Now select the ID based on the name
        cursor.execute("SELECT id FROM entities WHERE name = %s;", (name,))
        result = cursor.fetchone()
        if result:
            entity_id = result[0]
        else:
            # This case should be rare if insert succeeded or name existed
            logger.error(
                f"Could not find entity ID for name '{name}' after insert/conflict.")

        cursor.close()
        return entity_id

    except Exception as e:
        logger.error(
            f"Error finding or creating entity '{name}': {e}", exc_info=True)
        conn.rollback()  # Rollback on error during select or insert
        return None


def increment_entity_mentions(conn, article_id: int, entity_id: int, count: int = 1) -> bool:
    """
    Increment the mention count for an entity in an article.

    Args:
        conn: Database connection
        article_id: ID of the article
        entity_id: ID of the entity
        count: Number of mentions to add (default: 1)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Insert or update the link with incremented mention count
        cursor.execute("""
            INSERT INTO article_entities (
                article_id, entity_id, mention_count
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (article_id, entity_id) DO UPDATE
            SET mention_count = article_entities.mention_count + EXCLUDED.mention_count
            RETURNING id;
        """, (
            article_id,
            entity_id,
            count
        ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        return result

    except Exception as e:
        logger.error(
            f"Error incrementing mention count for article {article_id}, entity {entity_id}: {e}")
        conn.rollback()
        return False


def increment_global_entity_mentions(conn, entity_id: int, count: int = 1) -> bool:
    """
    Increment the global mentions count and update last_seen for an entity.

    Args:
        conn: Database connection
        entity_id: Entity ID
        count: Amount to increment by (default: 1)

    Returns:
        bool: True if successful, False otherwise
    """
    if count <= 0:  # Avoid unnecessary updates
        return True

    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE entities
            SET mentions = mentions + %s,
                last_seen = NOW()
            WHERE id = %s
            RETURNING id; -- Check if update occurred
        """, (count, entity_id))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()
        return result

    except Exception as e:
        logger.error(
            f"Error incrementing global mentions for entity {entity_id}: {e}", exc_info=True)
        conn.rollback()
        return False


def article_contains_entity(conn, article_id: int, entity_id: int) -> bool:
    """
    Check if an article has a specific entity.

    Args:
        conn: Database connection
        article_id: ID of the article
        entity_id: ID of the entity

    Returns:
        bool: True if the article contains the entity, False otherwise
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT EXISTS(
                SELECT 1
                FROM article_entities
                WHERE article_id = %s
                AND entity_id = %s
            );
        """, (article_id, entity_id))

        exists = cursor.fetchone()[0]
        cursor.close()

        return exists

    except Exception as e:
        logger.error(
            f"Error checking if article {article_id} contains entity {entity_id}: {e}")
        return False


def get_article_entities(conn, article_id: int) -> List[Dict[str, Any]]:
    """
    Retrieves all entities for a specific article with their mention counts
    and core entity details.

    Args:
        conn: Database connection
        article_id: ID of the article

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing entity details
                              (id, name, entity_type, influence_score, mentions)
                              and the mention_count from the article_entities link.
    """
    try:
        cursor = conn.cursor()
        # Select required fields, remove metadata
        cursor.execute("""
            SELECT e.id, e.name, e.entity_type, e.influence_score, e.mentions,
                   ae.mention_count
            FROM article_entities ae
            JOIN entities e ON ae.entity_id = e.id
            WHERE ae.article_id = %s
            ORDER BY ae.mention_count DESC, e.name ASC;
        """, (article_id,))

        columns = [desc[0] for desc in cursor.description]
        entities = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return entities

    except Exception as e:
        logger.error(
            f"Error retrieving entities for article {article_id}: {e}", exc_info=True)
        return []


def get_articles_by_entity(conn, entity_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve all articles that contain a specific entity.

    Args:
        conn: Database connection
        entity_id: ID of the entity
        limit: Maximum number of articles to return

    Returns:
        List[Dict[str, Any]]: List of article dictionaries, each containing:
            - id: Article ID
            - title: Article title
            - pub_date: Publication date
            - domain: Article domain
            - mention_count: Number of times the entity is mentioned in the article
    """
    try:
        cursor = conn.cursor()

        # Select existing columns: id, title, pub_date, domain
        cursor.execute("""
            SELECT a.id, a.title, a.pub_date, a.domain, ae.mention_count
            FROM articles a
            JOIN article_entities ae ON a.id = ae.article_id
            WHERE ae.entity_id = %s
            ORDER BY ae.mention_count DESC, a.pub_date DESC
            LIMIT %s
        """, (entity_id, limit))

        results = []
        # Match column names to selected data
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))

        cursor.close()
        return results

    except Exception as e:
        logger.error(
            f"Error retrieving articles for entity {entity_id}: {e}", exc_info=True)
        return []


def get_entities_by_type(conn, entity_type: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves entities filtered by a specific type.

    Args:
        conn: Database connection
        entity_type: The type to filter by (e.g., 'PERSON', 'ORG')
        limit: Maximum number of entities to return

    Returns:
        List[Dict[str, Any]]: List of matching entities with core details.
    """
    try:
        cursor = conn.cursor()
        # Update WHERE clause, select core fields, remove metadata
        cursor.execute("""
            SELECT id, name, entity_type, influence_score, mentions, first_seen, last_seen
            FROM entities
            WHERE entity_type = %s
            ORDER BY mentions DESC, name ASC
            LIMIT %s
        """, (entity_type, limit))

        columns = [desc[0] for desc in cursor.description]
        entities = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return entities

    except Exception as e:
        logger.error(
            f"Error retrieving entities by type '{entity_type}': {e}", exc_info=True)
        return []


def get_related_entities(conn, entity_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves entities that frequently appear in the same articles
    as the given entity.

    Args:
        conn: Database connection
        entity_id: ID of the primary entity
        limit: Maximum number of related entities to return

    Returns:
        List[Dict[str, Any]]: List of related entities with co-occurrence count
                              and core entity details.
    """
    try:
        cursor = conn.cursor()
        # Update selected fields, remove metadata
        cursor.execute("""
            SELECT e2.id, e2.name, e2.entity_type, e2.influence_score, e2.mentions,
                   COUNT(ae1.article_id) as co_occurrence_count
            FROM article_entities ae1
            JOIN article_entities ae2 ON ae1.article_id = ae2.article_id AND ae1.entity_id != ae2.entity_id
            JOIN entities e2 ON ae2.entity_id = e2.id
            WHERE ae1.entity_id = %s
            GROUP BY e2.id, e2.name, e2.entity_type, e2.influence_score, e2.mentions
            ORDER BY co_occurrence_count DESC, e2.name ASC
            LIMIT %s;
        """, (entity_id, limit))

        columns = [desc[0] for desc in cursor.description]
        related_entities = [dict(zip(columns, row))
                            for row in cursor.fetchall()]

        cursor.close()
        return related_entities

    except Exception as e:
        logger.error(
            f"Error retrieving related entities for entity {entity_id}: {e}", exc_info=True)
        return []
