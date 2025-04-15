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
    Insert an entity into the database.

    Args:
        conn: Database connection
        entity: The entity data to insert

    Returns:
        int or None: The new entity ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()

        # Extract entity data with defaults
        name = entity.get('name', '')
        entity_type = entity.get('type', 'unknown')
        metadata = entity.get('metadata', {})

        # Insert the entity
        cursor.execute("""
            INSERT INTO entities (
                name, type, metadata, created_at
            )
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (name) DO UPDATE
            SET type = EXCLUDED.type,
                metadata = COALESCE(entities.metadata, '{}') || EXCLUDED.metadata
            RETURNING id;
        """, (
            name,
            entity_type,
            json.dumps(metadata) if metadata else None
        ))

        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        return new_id

    except Exception as e:
        logger.error(f"Error inserting entity: {e}")
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

        # Insert or update the link
        cursor.execute("""
            INSERT INTO article_entities (
                article_id, entity_id, mention_count
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (article_id, entity_id) DO UPDATE
            SET mention_count = EXCLUDED.mention_count
            RETURNING id;
        """, (
            article_id,
            entity_id,
            mention_count
        ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        return result

    except Exception as e:
        logger.error(
            f"Error linking article {article_id} to entity {entity_id}: {e}")
        conn.rollback()
        return False


def get_entities_by_influence(conn, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get entities sorted by influence score.

    Args:
        conn: Database connection
        limit: Maximum number of entities to return

    Returns:
        List[Dict[str, Any]]: List of entities with influence scores
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT e.id, e.name, e.type, e.metadata,
                   COUNT(DISTINCT ae.article_id) as article_count,
                   SUM(ae.mention_count) as total_mentions
            FROM entities e
            JOIN article_entities ae ON e.id = ae.entity_id
            GROUP BY e.id
            ORDER BY total_mentions DESC
            LIMIT %s
        """, (limit,))

        entities = []
        for row in cursor.fetchall():
            metadata = row[3] if row[3] else {}

            entities.append({
                'id': row[0],
                'name': row[1],
                'type': row[2],
                'metadata': metadata,
                'article_count': row[4],
                'total_mentions': row[5],
                # mentions per article
                'influence_score': row[5] / max(row[4], 1)
            })

        cursor.close()
        return entities

    except Exception as e:
        logger.error(f"Error retrieving entities by influence: {e}")
        return []


def get_entity_influence_for_articles(conn, article_ids: List[int]) -> Dict[int, float]:
    """
    Get entity influence scores for a list of articles.

    Args:
        conn: Database connection
        article_ids: List of article IDs

    Returns:
        Dict[int, float]: Dictionary mapping article IDs to influence scores
    """
    if not article_ids:
        return {}

    try:
        cursor = conn.cursor()

        # Convert article_ids list to string format for SQL IN clause
        articles_str = ','.join(str(id) for id in article_ids)

        # Query to calculate average entity influence per article
        cursor.execute(f"""
            SELECT ae.article_id, 
                   AVG(
                     (SELECT COUNT(*) FROM article_entities 
                      WHERE entity_id = ae.entity_id)::float
                   ) as avg_entity_influence
            FROM article_entities ae
            WHERE ae.article_id IN ({articles_str})
            GROUP BY ae.article_id
        """)

        influence_scores = {}
        for row in cursor.fetchall():
            influence_scores[row[0]] = float(row[1])

        cursor.close()
        return influence_scores

    except Exception as e:
        logger.error(f"Error retrieving entity influence for articles: {e}")
        return {}
