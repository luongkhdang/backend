"""
relationships.py - Module for managing entity relationships in the Reader database

This module provides functions for recording and querying relationships between entities.
It captures co-occurrence contexts and maintains metadata about the evidence supporting
these relationships.

Exported functions:
- initialize_tables(conn) -> bool
- record_relationship_context(conn, entity_id_1, entity_id_2, context_type, article_id, evidence_snippet) -> bool
- get_entity_relationships(conn, entity_id, context_type=None, limit=None) -> List[Dict[str, Any]]
- get_relationship_snippets(conn, entity_id_1, entity_id_2, limit=None) -> List[str]
- get_relationship_articles(conn, entity_id_1, entity_id_2, limit=None) -> List[int]

Related files:
- src/database/reader_db_client.py: Uses this module for relationship operations
- src/database/modules/schema.py: Defines the database schema for relationships
- src/database/modules/entities.py: Manages entities that participate in relationships
"""

import logging
import json
import math
from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_tables(conn) -> bool:
    """
    Initialize entity_relationships table if it doesn't exist.
    Called during schema initialization.

    Args:
        conn: Database connection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Create entity_relationships table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_relationships (
            entity_id_1 INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            entity_id_2 INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            context_type TEXT NOT NULL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            article_count INTEGER DEFAULT 1,
            confidence_score FLOAT DEFAULT 0.1,
            metadata JSONB DEFAULT '{}'::jsonb,
            PRIMARY KEY (entity_id_1, entity_id_2, context_type)
        );
        """)

        # Create indexes
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_entity_relationships_entity1 ON entity_relationships(entity_id_1);
        CREATE INDEX IF NOT EXISTS idx_entity_relationships_entity2 ON entity_relationships(entity_id_2);
        CREATE INDEX IF NOT EXISTS idx_entity_relationships_context ON entity_relationships(context_type);
        CREATE INDEX IF NOT EXISTS idx_entity_relationships_confidence ON entity_relationships(confidence_score DESC);
        """)

        conn.commit()
        logger.info("Entity relationships table initialized successfully")
        return True

    except Exception as e:
        logger.error(
            f"Error initializing entity relationships table: {e}")
        conn.rollback()
        return False


def record_relationship_context(conn, entity_id_1: int, entity_id_2: int, context_type: str,
                                article_id: int, evidence_snippet: Optional[str] = None) -> bool:
    """
    Record or update a relationship context between two entities.

    This function will:
    - Insert a new relationship if it doesn't exist
    - Increment the article_count if it exists
    - Add the article_id and evidence_snippet to the metadata
    - Update the last_updated timestamp
    - Adjust confidence_score based on simple rules

    Args:
        conn: Database connection
        entity_id_1: ID of the first entity
        entity_id_2: ID of the second entity
        context_type: Type of relationship context (e.g., AGREEMENT_CONTEXT, CONFLICT_CONTEXT)
        article_id: ID of the article where this relationship was mentioned
        evidence_snippet: Text snippet providing evidence of the relationship

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure entity_id_1 is always the smaller ID for consistency
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1

        cursor = conn.cursor()

        # Check if relationship exists
        cursor.execute(
            """
            SELECT article_count, metadata, confidence_score 
            FROM entity_relationships 
            WHERE entity_id_1 = %s AND entity_id_2 = %s AND context_type = %s
            """,
            (entity_id_1, entity_id_2, context_type)
        )
        result = cursor.fetchone()

        # New metadata to store or update
        new_evidence = {
            "article_id": article_id
        }
        if evidence_snippet:
            new_evidence["snippet"] = evidence_snippet

        if result:
            # Update existing relationship
            article_count, metadata_json, confidence_score = result

            # Parse existing metadata
            metadata = metadata_json if metadata_json else {}

            # Add new evidence to metadata
            if "evidence" not in metadata:
                metadata["evidence"] = []

            # Check if this article is already in evidence
            article_ids = {e.get("article_id") for e in metadata["evidence"] if isinstance(
                e, dict) and "article_id" in e}

            if article_id not in article_ids:
                metadata["evidence"].append(new_evidence)

            # Update article count
            new_article_count = article_count + 1

            # Simple confidence score adjustment: more articles = higher confidence (logarithmic scale)
            new_confidence = min(
                0.9, 0.1 + (0.1 * math.log(new_article_count + 1)))

            cursor.execute(
                """
                UPDATE entity_relationships
                SET article_count = %s,
                    last_updated = CURRENT_TIMESTAMP,
                    metadata = %s,
                    confidence_score = %s
                WHERE entity_id_1 = %s AND entity_id_2 = %s AND context_type = %s
                """,
                (new_article_count, json.dumps(metadata),
                 new_confidence, entity_id_1, entity_id_2, context_type)
            )
        else:
            # Create new relationship
            metadata = {
                "evidence": [new_evidence]
            }

            cursor.execute(
                """
                INSERT INTO entity_relationships
                (entity_id_1, entity_id_2, context_type, article_count, metadata, confidence_score)
                VALUES (%s, %s, %s, 1, %s, 0.1)
                """,
                (entity_id_1, entity_id_2, context_type, json.dumps(metadata))
            )

        conn.commit()
        return True

    except Exception as e:
        logger.error(
            f"Error in record_relationship_context: {e}")
        conn.rollback()
        return False


def get_entity_relationships(conn, entity_id: int, context_type: Optional[str] = None,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get relationships for a specific entity, optionally filtered by context type.

    Args:
        conn: Database connection
        entity_id: ID of the entity
        context_type: Optional filter for specific context types
        limit: Maximum number of relationships to return

    Returns:
        List[Dict]: List of relationships with related entities
    """
    try:
        cursor = conn.cursor()

        # Build the base query
        query = """
        SELECT 
            CASE 
                WHEN er.entity_id_1 = %s THEN er.entity_id_2 
                ELSE er.entity_id_1 
            END AS related_entity_id,
            e.name AS related_entity_name,
            e.entity_type AS related_entity_type,
            er.context_type,
            er.article_count,
            er.confidence_score,
            er.first_seen,
            er.last_updated
        FROM entity_relationships er
        JOIN entities e ON (
            CASE 
                WHEN er.entity_id_1 = %s THEN er.entity_id_2 
                ELSE er.entity_id_1 
            END = e.id
        )
        WHERE (er.entity_id_1 = %s OR er.entity_id_2 = %s)
        """

        params = [entity_id, entity_id, entity_id, entity_id]

        # Add context type filter if provided
        if context_type:
            query += " AND er.context_type = %s"
            params.append(context_type)

        # Add ordering
        query += " ORDER BY er.confidence_score DESC, er.article_count DESC, er.last_updated DESC"

        # Add limit if provided
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query, params)

        relationships = []
        for row in cursor.fetchall():
            relationships.append({
                "entity_id": row[0],
                "entity_name": row[1],
                "entity_type": row[2],
                "context_type": row[3],
                "article_count": row[4],
                "confidence_score": row[5],
                "first_seen": row[6],
                "last_updated": row[7]
            })

        return relationships

    except Exception as e:
        logger.error(f"Error in get_entity_relationships: {e}")
        return []


def get_relationship_snippets(conn, entity_id_1: int, entity_id_2: int,
                              context_type: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
    """
    Get evidence snippets for a relationship between two entities.

    Args:
        conn: Database connection
        entity_id_1: ID of the first entity
        entity_id_2: ID of the second entity
        context_type: Optional filter for specific context type
        limit: Maximum number of snippets to return

    Returns:
        List[str]: List of evidence snippets
    """
    try:
        # Ensure entity_id_1 is always the smaller ID for consistency
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1

        cursor = conn.cursor()

        # Build the query based on whether context_type is provided
        if context_type:
            query = """
            SELECT metadata
            FROM entity_relationships
            WHERE entity_id_1 = %s AND entity_id_2 = %s AND context_type = %s
            """
            params = (entity_id_1, entity_id_2, context_type)
        else:
            query = """
            SELECT metadata, context_type
            FROM entity_relationships
            WHERE entity_id_1 = %s AND entity_id_2 = %s
            """
            params = (entity_id_1, entity_id_2)

        cursor.execute(query, params)
        results = cursor.fetchall()

        snippets = []
        for row in results:
            metadata = row[0]
            if metadata and "evidence" in metadata:
                for evidence in metadata["evidence"]:
                    if "snippet" in evidence:
                        # Add context type if it was part of the query result
                        if not context_type and len(row) > 1:
                            typed_snippet = f"[{row[1]}] {evidence['snippet']}"
                            snippets.append(typed_snippet)
                        else:
                            snippets.append(evidence["snippet"])

        # Apply limit if provided
        if limit and len(snippets) > limit:
            snippets = snippets[:limit]

        return snippets

    except Exception as e:
        logger.error(f"Error in get_relationship_snippets: {e}")
        return []


def get_relationship_articles(conn, entity_id_1: int, entity_id_2: int,
                              context_type: Optional[str] = None, limit: Optional[int] = None) -> List[int]:
    """
    Get article IDs that mention a relationship between two entities.

    Args:
        conn: Database connection
        entity_id_1: ID of the first entity
        entity_id_2: ID of the second entity
        context_type: Optional filter for specific context type
        limit: Maximum number of article IDs to return

    Returns:
        List[int]: List of article IDs
    """
    try:
        # Ensure entity_id_1 is always the smaller ID for consistency
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1

        cursor = conn.cursor()

        # Build the query based on whether context_type is provided
        if context_type:
            query = """
            SELECT metadata
            FROM entity_relationships
            WHERE entity_id_1 = %s AND entity_id_2 = %s AND context_type = %s
            """
            params = (entity_id_1, entity_id_2, context_type)
        else:
            query = """
            SELECT metadata
            FROM entity_relationships
            WHERE entity_id_1 = %s AND entity_id_2 = %s
            """
            params = (entity_id_1, entity_id_2)

        cursor.execute(query, params)
        results = cursor.fetchall()

        article_ids = set()
        for row in results:
            metadata = row[0]
            if metadata and "evidence" in metadata:
                for evidence in metadata["evidence"]:
                    if "article_id" in evidence:
                        article_ids.add(evidence["article_id"])

        article_id_list = list(article_ids)

        # Apply limit if provided
        if limit and len(article_id_list) > limit:
            article_id_list = article_id_list[:limit]

        return article_id_list

    except Exception as e:
        logger.error(f"Error in get_relationship_articles: {e}")
        return []
