"""
embeddings.py - Embeddings-related database operations

This module provides functions for managing article embeddings in the database,
including inserting, updating, and retrieving embeddings.

Exported functions:
- insert_embedding(conn, article_id: int, embedding_data: Dict[str, Any]) -> bool
  Inserts an embedding for an article
- batch_insert_embeddings(conn, embeddings: List[Dict[str, Any]]) -> Dict[str, int]
  Inserts multiple embeddings in batch
- get_all_embeddings(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets all embeddings
- get_all_embeddings_with_pub_date(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets all embeddings with publication dates
- get_embedding_for_article(conn, article_id: int) -> Optional[Dict[str, Any]]
  Gets embedding for a specific article

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for embeddings operations
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def insert_embedding(conn, article_id: int, embedding_data: Dict[str, Any]) -> bool:
    """
    Insert an embedding for an article.

    Args:
        conn: Database connection
        article_id: ID of the article this embedding belongs to
        embedding_data: Data containing the embedding vector and metadata

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Extract embedding vector and other fields
        vector = embedding_data.get('vector')
        model = embedding_data.get('model', 'unknown')
        created_at = embedding_data.get('created_at', datetime.now())

        # Insert or update the embedding
        cursor.execute("""
            INSERT INTO embeddings (
                article_id, vector, model, created_at
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (article_id) DO UPDATE
            SET vector = EXCLUDED.vector,
                model = EXCLUDED.model,
                created_at = EXCLUDED.created_at
            RETURNING id;
        """, (
            article_id,
            vector,
            model,
            created_at
        ))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        return result

    except Exception as e:
        logger.error(
            f"Error inserting embedding for article {article_id}: {e}")
        conn.rollback()
        return False


def batch_insert_embeddings(conn, embeddings: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Insert multiple embeddings into the database in batches.

    Args:
        conn: Database connection
        embeddings: List of dictionaries with article_id and embedding data

    Returns:
        Dict[str, int]: Dictionary with success and failure counts
    """
    if not embeddings:
        return {"success": 0, "failure": 0}

    successful_inserts = 0
    failed_inserts = 0

    # Process and insert embeddings in batches
    batch_size = 100
    total_batches = (len(embeddings) + batch_size - 1) // batch_size

    try:
        cursor = conn.cursor()

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(embeddings))
            current_batch = embeddings[start_idx:end_idx]

            for embedding_data in current_batch:
                try:
                    article_id = embedding_data.get('article_id')
                    vector = embedding_data.get('vector')

                    if not article_id or vector is None:
                        logger.warning(
                            f"Missing required fields for embedding: article_id={article_id}, vector present={vector is not None}")
                        failed_inserts += 1
                        continue

                    model = embedding_data.get('model', 'unknown')
                    created_at = embedding_data.get(
                        'created_at', datetime.now())

                    # Insert or update the embedding
                    cursor.execute("""
                        INSERT INTO embeddings (
                            article_id, vector, model, created_at
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (article_id) DO UPDATE
                        SET vector = EXCLUDED.vector,
                            model = EXCLUDED.model,
                            created_at = EXCLUDED.created_at
                        RETURNING id;
                    """, (
                        article_id,
                        vector,
                        model,
                        created_at
                    ))

                    if cursor.fetchone():
                        successful_inserts += 1
                    else:
                        failed_inserts += 1

                except Exception as e:
                    logger.error(
                        f"Error inserting embedding for article {embedding_data.get('article_id')}: {e}")
                    failed_inserts += 1

            conn.commit()

        cursor.close()
        logger.info(
            f"Batch insertion of embeddings complete: {successful_inserts} successful, {failed_inserts} failed")

    except Exception as e:
        logger.error(f"Error in batch insert process for embeddings: {e}")
        conn.rollback()

    return {"success": successful_inserts, "failure": failed_inserts}


def get_all_embeddings(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all embeddings from the database.

    Args:
        conn: Database connection
        limit: Optional limit on the number of embeddings to return

    Returns:
        List[Dict[str, Any]]: List of embedding records with article_id and vector
    """
    try:
        cursor = conn.cursor()

        if limit:
            cursor.execute("""
                SELECT e.article_id, e.vector, e.model, e.created_at
                FROM embeddings e
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT e.article_id, e.vector, e.model, e.created_at
                FROM embeddings e
            """)

        embeddings = []
        for row in cursor.fetchall():
            embeddings.append({
                'article_id': row[0],
                'vector': row[1],
                'model': row[2],
                'created_at': row[3]
            })

        cursor.close()
        return embeddings

    except Exception as e:
        logger.error(f"Error retrieving all embeddings: {e}")
        return []


def get_all_embeddings_with_pub_date(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all embeddings with their corresponding article publication dates.

    Args:
        conn: Database connection
        limit: Optional limit on the number of embeddings to return

    Returns:
        List[Dict[str, Any]]: List of embedding records with article_id, vector, and pub_date
    """
    try:
        cursor = conn.cursor()

        if limit:
            cursor.execute("""
                SELECT e.article_id, e.vector, a.pub_date, e.model
                FROM embeddings e
                JOIN articles a ON e.article_id = a.id
                WHERE a.pub_date IS NOT NULL
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT e.article_id, e.vector, a.pub_date, e.model
                FROM embeddings e
                JOIN articles a ON e.article_id = a.id
                WHERE a.pub_date IS NOT NULL
            """)

        embeddings = []
        for row in cursor.fetchall():
            embeddings.append({
                'article_id': row[0],
                'vector': row[1],
                'pub_date': row[2],
                'model': row[3]
            })

        cursor.close()

        logger.info(
            f"Retrieved {len(embeddings)} embeddings with publication dates")
        return embeddings

    except Exception as e:
        logger.error(
            f"Error retrieving embeddings with publication dates: {e}")
        return []


def get_embedding_for_article(conn, article_id: int) -> Optional[Dict[str, Any]]:
    """
    Get the embedding for a specific article.

    Args:
        conn: Database connection
        article_id: ID of the article

    Returns:
        Dict[str, Any] or None: The embedding data or None if not found
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT article_id, vector, model, created_at
            FROM embeddings
            WHERE article_id = %s
        """, (article_id,))

        row = cursor.fetchone()
        cursor.close()

        if row:
            return {
                'article_id': row[0],
                'vector': row[1],
                'model': row[2],
                'created_at': row[3]
            }

        return None

    except Exception as e:
        logger.error(
            f"Error retrieving embedding for article {article_id}: {e}")
        return None


def get_embeddings_for_articles(conn, article_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Get embeddings for multiple articles.

    Args:
        conn: Database connection
        article_ids: List of article IDs

    Returns:
        Dict[int, Dict[str, Any]]: Dictionary mapping article IDs to embedding data
    """
    if not article_ids:
        return {}

    try:
        cursor = conn.cursor()

        placeholders = ", ".join(["%s"] * len(article_ids))
        query = f"""
            SELECT article_id, vector, model, created_at
            FROM embeddings
            WHERE article_id IN ({placeholders})
        """

        cursor.execute(query, article_ids)

        # Map article_id to embedding
        embeddings_map = {}
        for row in cursor.fetchall():
            embeddings_map[row[0]] = {
                'article_id': row[0],
                'vector': row[1],
                'model': row[2],
                'created_at': row[3]
            }

        cursor.close()
        return embeddings_map

    except Exception as e:
        logger.error(f"Error retrieving embeddings for multiple articles: {e}")
        return {}


def delete_embedding(conn, article_id: int) -> bool:
    """
    Delete an embedding for a specific article.

    Args:
        conn: Database connection
        article_id: ID of the article

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM embeddings
            WHERE article_id = %s
            RETURNING id
        """, (article_id,))

        result = cursor.fetchone() is not None
        conn.commit()
        cursor.close()

        return result

    except Exception as e:
        logger.error(f"Error deleting embedding for article {article_id}: {e}")
        conn.rollback()
        return False


def get_embeddings_by_model(conn, model_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get embeddings created using a specific model.

    Args:
        conn: Database connection
        model_name: Name of the embedding model
        limit: Optional limit on the number of embeddings to return

    Returns:
        List[Dict[str, Any]]: List of embedding records from the specified model
    """
    try:
        cursor = conn.cursor()

        if limit:
            cursor.execute("""
                SELECT article_id, vector, model, created_at
                FROM embeddings
                WHERE model = %s
                LIMIT %s
            """, (model_name, limit))
        else:
            cursor.execute("""
                SELECT article_id, vector, model, created_at
                FROM embeddings
                WHERE model = %s
            """, (model_name,))

        embeddings = []
        for row in cursor.fetchall():
            embeddings.append({
                'article_id': row[0],
                'vector': row[1],
                'model': row[2],
                'created_at': row[3]
            })

        cursor.close()

        logger.info(
            f"Retrieved {len(embeddings)} embeddings for model '{model_name}'")
        return embeddings

    except Exception as e:
        logger.error(
            f"Error retrieving embeddings for model '{model_name}': {e}")
        return []
