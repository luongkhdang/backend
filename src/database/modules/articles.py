"""
articles.py - Article-related database operations

This module provides functions for article-related database operations,
including inserting, updating, and querying articles.

Exported functions:
- insert_article(conn, article: Dict[str, Any]) -> Optional[int]
  Inserts an article into the database
- batch_insert_articles(conn, articles: List[Dict[str, Any]]) -> Dict[str, int]
  Inserts multiple articles in batch
- get_article_by_id(conn, article_id: int) -> Optional[Dict[str, Any]]
  Gets an article by its ID
- get_article_by_scraper_id(conn, scraper_id: int) -> Optional[Dict[str, Any]]
  Gets an article by its scraper_id
- update_article_cluster(conn, article_id: int, cluster_id: int, is_hot: bool) -> bool
  Updates an article's cluster assignment and hotness
- batch_update_article_clusters(conn, assignments: List[Tuple[int, Optional[int], bool]]) -> Dict[str, int]
  Updates multiple article cluster assignments and hotness flags in batch
- get_articles_needing_embedding(conn, max_char_count: int) -> List[Dict[str, Any]]
  Gets articles that need embedding generation
- get_articles_needing_summarization(conn, min_char_count: int) -> List[Dict[str, Any]]
  Gets articles that need summarization
- get_error_articles(conn) -> List[Dict[str, Any]]
  Gets articles with errors
- get_hot_articles(conn, limit: int) -> List[Dict[str, Any]]
  Gets hot articles
- get_sample_titles_for_articles(conn, article_ids: List[int], sample_size: int) -> List[str]
  Gets a sample of titles for a list of article IDs
- get_publication_dates_for_articles(conn, article_ids: List[int]) -> Dict[int, Optional[datetime]]
  Gets publication dates for a list of article IDs
- get_recent_unprocessed_articles(conn, days: int = 2, limit: int = 2000) -> List[Dict[str, Any]]
  Gets recent unprocessed articles
- mark_article_processed(conn, article_id: int) -> bool
  Marks an article as having had its entities extracted

Related modules:
- Connection management from connection.py
- Used by ReaderDBClient for article operations
"""

import logging
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

try:
    import dateutil.parser as dateutil_parser
except ImportError:
    dateutil_parser = None

# Configure logging
logger = logging.getLogger(__name__)


def insert_article(conn, article: Dict[str, Any]) -> Optional[int]:
    """
    Insert an article into the database.

    Args:
        conn: Database connection
        article: The article data to insert

    Returns:
        int or None: The new article ID if successful, None otherwise
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO articles (
                scraper_id, title, content, 
                pub_date, domain
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (scraper_id) DO UPDATE
            SET title = EXCLUDED.title,
                content = EXCLUDED.content,
                domain = EXCLUDED.domain,
                pub_date = EXCLUDED.pub_date,
                processed_at = CURRENT_TIMESTAMP
            RETURNING id;
        """, (
            article.get('id') or article.get('scraper_id'),
            article.get('title', ''),
            article.get('content', ''),
            article.get('pub_date'),
            article.get('domain', '')
        ))

        new_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        return new_id

    except Exception as e:
        logger.error(f"Error inserting article: {e}")
        conn.rollback()
        return None


def batch_insert_articles(conn, articles: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Insert multiple articles into the database in batches.

    Args:
        conn: Database connection
        articles: List of article dictionaries to insert

    Returns:
        Dict[str, int]: Dictionary with success and failure counts
    """
    if not articles:
        return {"success": 0, "failure": 0}

    successful_inserts = 0
    failed_inserts = 0

    # Process and insert articles in batches
    batch_size = 100
    total_batches = (len(articles) + batch_size - 1) // batch_size

    try:
        cursor = conn.cursor()

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(articles))
            current_batch = articles[start_idx:end_idx]

            for article in current_batch:
                try:
                    # Handle processed_at field
                    processed_at = article.get('processed_at')

                    # Convert processed_at to proper format if it's a string
                    if isinstance(processed_at, str) and dateutil_parser:
                        try:
                            processed_at = dateutil_parser.parse(processed_at)
                        except:
                            # If parsing fails, use current timestamp
                            processed_at = datetime.now()

                    # Insert article with content and processed_at fields
                    cursor.execute("""
                        INSERT INTO articles (
                            scraper_id, title, pub_date, domain, content, processed_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (scraper_id) DO UPDATE
                        SET title = EXCLUDED.title,
                            pub_date = EXCLUDED.pub_date,
                            domain = EXCLUDED.domain,
                            content = EXCLUDED.content,
                            processed_at = EXCLUDED.processed_at
                        RETURNING id;
                    """, (
                        article.get('scraper_id'),
                        article.get(
                            'title', f"Article {article.get('scraper_id')}"),
                        article.get('pub_date'),
                        article.get('domain', ''),
                        article.get('content', ''),
                        processed_at  # Use the processed value
                    ))

                    if cursor.fetchone():
                        successful_inserts += 1
                    else:
                        failed_inserts += 1

                except Exception as e:
                    logger.error(f"Error inserting article in batch: {e}")
                    failed_inserts += 1

            conn.commit()

        cursor.close()
        logger.info(
            f"Batch insertion complete: {successful_inserts} successful, {failed_inserts} failed")

    except Exception as e:
        logger.error(f"Error in batch insert process: {e}")
        conn.rollback()

    return {"success": successful_inserts, "failure": failed_inserts}


def get_article_by_id(conn, article_id: int) -> Optional[Dict[str, Any]]:
    """
    Get an article by its ID.

    Args:
        conn: Database connection
        article_id: The article ID

    Returns:
        Dict or None: The article data or None if not found
    """
    try:
        cursor = conn.cursor()

        query = "SELECT * FROM articles WHERE id = %s"
        cursor.execute(query, (article_id,))

        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()

        cursor.close()

        if row:
            return dict(zip(columns, row))
        return None

    except Exception as e:
        logger.error(f"Error retrieving article by ID: {e}")
        return None


def get_article_by_scraper_id(conn, scraper_id: int) -> Optional[Dict[str, Any]]:
    """
    Get an article by its scraper_id.

    Args:
        conn: Database connection
        scraper_id: The scraper_id of the article

    Returns:
        Dict or None: The article data or None if not found
    """
    try:
        cursor = conn.cursor()

        query = "SELECT * FROM articles WHERE scraper_id = %s"
        cursor.execute(query, (scraper_id,))

        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()

        cursor.close()

        if row:
            return dict(zip(columns, row))
        return None

    except Exception as e:
        logger.error(f"Error retrieving article by scraper_id: {e}")
        return None


def update_article_cluster(conn, article_id: int, cluster_id: Optional[int], is_hot: bool = False) -> bool:
    """
    Update an article's cluster assignment and hotness flag.

    Args:
        conn: Database connection
        article_id: The article ID
        cluster_id: The cluster ID (or None for no cluster/noise)
        is_hot: Whether the article should be marked as hot

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        if cluster_id is None:
            # Update the article with NULL cluster_id
            cursor.execute("""
                UPDATE articles
                SET cluster_id = NULL, is_hot = %s
                WHERE id = %s;
            """, (is_hot, article_id))
        else:
            # Update the article with cluster_id
            cursor.execute("""
                UPDATE articles
                SET cluster_id = %s, is_hot = %s
                WHERE id = %s;
            """, (cluster_id, is_hot, article_id))

            # Update the cluster's article count
            cursor.execute("""
                UPDATE clusters
                SET article_count = (
                    SELECT COUNT(*) 
                    FROM articles 
                    WHERE cluster_id = %s
                )
                WHERE id = %s;
            """, (cluster_id, cluster_id))

        conn.commit()
        cursor.close()
        return True

    except Exception as e:
        logger.error(f"Error updating article cluster: {e}")
        conn.rollback()
        return False


def batch_update_article_clusters(conn, assignments: List[Tuple[int, Optional[int], bool]]) -> Dict[str, int]:
    """
    Update multiple article cluster assignments and hotness flags in batch.

    Args:
        conn: Database connection
        assignments: List of tuples (article_id, cluster_id, is_hot) where cluster_id can be None

    Returns:
        Dict[str, int]: Dictionary with success and failure counts
    """
    if not assignments:
        return {"success": 0, "failure": 0}

    try:
        cursor = conn.cursor()

        # Split assignments into batches
        batch_size = 1000
        success_count = 0
        failure_count = 0

        for i in range(0, len(assignments), batch_size):
            batch = assignments[i:i+batch_size]

            try:
                # Convert batch to SQL-friendly format and handle NULL properly
                values = []
                for article_id, cluster_id, is_hot in batch:
                    is_hot_value = "TRUE" if is_hot else "FALSE"
                    if cluster_id is None:
                        values.append(f"({article_id}, NULL, {is_hot_value})")
                    else:
                        values.append(
                            f"({article_id}, {cluster_id}, {is_hot_value})")

                values_str = ", ".join(values)

                # SQL query using temporary table for efficient update
                query = f"""
                UPDATE articles
                SET cluster_id = temp.cluster_id,
                    is_hot = temp.is_hot
                FROM (VALUES {values_str}) AS temp(article_id, cluster_id, is_hot)
                WHERE articles.id = temp.article_id
                """

                cursor.execute(query)
                success_count += len(batch)
            except Exception as e:
                logger.error(f"Error updating batch {i//batch_size}: {e}")
                failure_count += len(batch)
                # Continue with next batch rather than failing completely

        # Update cluster article counts
        try:
            # Get unique cluster IDs (excluding None)
            cluster_ids = set()
            for _, cluster_id, _ in assignments:
                if cluster_id is not None:
                    cluster_ids.add(cluster_id)

            # Update count for each cluster
            for cluster_id in cluster_ids:
                cursor.execute("""
                    UPDATE clusters
                    SET article_count = (
                        SELECT COUNT(*) 
                        FROM articles 
                        WHERE cluster_id = %s
                    )
                    WHERE id = %s;
                """, (cluster_id, cluster_id))
        except Exception as e:
            logger.error(f"Error updating cluster article counts: {e}")
            # Don't count this as a failure for article assignment

        conn.commit()
        cursor.close()

        logger.info(
            f"Updated {success_count} article cluster assignments and hotness flags (failed: {failure_count})")
        return {"success": success_count, "failure": failure_count}

    except Exception as e:
        logger.error(f"Failed to update cluster assignments: {e}")
        conn.rollback()
        return {"success": 0, "failure": len(assignments)}


def get_articles_needing_embedding(conn, max_char_count: int = 8000) -> List[Dict[str, Any]]:
    """
    Get articles that need embedding generation.

    Args:
        conn: Database connection
        max_char_count: Maximum character count for articles to process

    Returns:
        List[Dict[str, Any]]: List of articles that need embeddings
    """
    try:
        cursor = conn.cursor()

        # Query for articles that have valid content but no embeddings yet
        query = """
            SELECT a.id, a.content
            FROM articles a
            LEFT JOIN embeddings e ON a.id = e.article_id
            WHERE
                a.content IS NOT NULL
                AND a.content != 'ERROR'
                AND LENGTH(a.content) <= %s
                AND e.article_id IS NULL;
        """
        cursor.execute(query, (max_char_count,))

        columns = [desc[0] for desc in cursor.description]
        articles_to_embed = []

        for row in cursor.fetchall():
            articles_to_embed.append(dict(zip(columns, row)))

        cursor.close()

        logger.info(
            f"Found {len(articles_to_embed)} articles needing embeddings (content <= {max_char_count} chars)")
        return articles_to_embed

    except Exception as e:
        logger.error(f"Error retrieving articles needing embeddings: {e}")
        return []


def get_articles_needing_summarization(conn, min_char_count: int = 8000) -> List[Dict[str, Any]]:
    """
    Get articles that need summarization because their content length exceeds the specified limit.

    Args:
        conn: Database connection
        min_char_count: Minimum character count for an article to require summarization

    Returns:
        List[Dict[str, Any]]: List of articles needing summarization
    """
    try:
        cursor = conn.cursor()

        # Query for articles that have valid content, exceed the char count, and don't have embeddings yet
        query = """
            SELECT a.id, a.content
            FROM articles a
            LEFT JOIN embeddings e ON a.id = e.article_id
            WHERE
                a.content IS NOT NULL
                AND a.content != 'ERROR'
                AND LENGTH(a.content) > %s
                AND e.article_id IS NULL;
        """
        cursor.execute(query, (min_char_count,))

        columns = [desc[0] for desc in cursor.description]
        articles_to_summarize = []

        for row in cursor.fetchall():
            articles_to_summarize.append(dict(zip(columns, row)))

        cursor.close()

        logger.info(
            f"Found {len(articles_to_summarize)} articles needing summarization (content > {min_char_count} chars)")
        return articles_to_summarize

    except Exception as e:
        logger.error(f"Error retrieving articles needing summarization: {e}")
        return []


def get_error_articles(conn) -> List[Dict[str, Any]]:
    """
    Get all articles with null or error content.

    Args:
        conn: Database connection

    Returns:
        List[Dict[str, Any]]: List of articles with errors
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, scraper_id, title, content, pub_date, domain, processed_at
            FROM articles
            WHERE content IS NULL OR content = 'ERROR' OR LENGTH(TRIM(content)) = 0
            ORDER BY processed_at DESC;
        """)

        columns = [desc[0] for desc in cursor.description]
        error_articles = []

        for row in cursor.fetchall():
            error_articles.append(dict(zip(columns, row)))

        cursor.close()
        return error_articles

    except Exception as e:
        logger.error(f"Error retrieving articles with errors: {e}")
        return []


def get_hot_articles(conn, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get articles marked as 'hot'.

    Args:
        conn: Database connection
        limit: Maximum number of results to return

    Returns:
        List[Dict[str, Any]]: List of hot articles
    """
    try:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT *
            FROM articles
            WHERE is_hot = TRUE
            ORDER BY processed_at DESC
            LIMIT %s;
        """, (limit,))

        columns = [desc[0] for desc in cursor.description]
        hot_articles = []

        for row in cursor.fetchall():
            hot_articles.append(dict(zip(columns, row)))

        cursor.close()
        return hot_articles

    except Exception as e:
        logger.error(f"Error retrieving hot articles: {e}")
        return []


def get_sample_titles_for_articles(conn, article_ids: List[int], sample_size: int) -> List[str]:
    """
    Get a sample of titles for a list of article IDs.

    Args:
        conn: Database connection
        article_ids: List of article IDs
        sample_size: Maximum number of titles to return

    Returns:
        List[str]: Sample list of article titles
    """
    if not article_ids:
        return []

    try:
        cursor = conn.cursor()

        # Create a string with the article IDs for the IN clause
        ids_placeholder = ', '.join(['%s'] * len(article_ids))

        # Get a random sample of articles
        cursor.execute(f"""
            SELECT title 
            FROM articles 
            WHERE id IN ({ids_placeholder})
            ORDER BY RANDOM() 
            LIMIT %s
        """, article_ids + [sample_size])

        # Extract titles
        titles = [row[0] for row in cursor.fetchall()]
        cursor.close()

        return titles

    except Exception as e:
        logger.error(f"Error getting sample titles: {e}")
        return []


def get_publication_dates_for_articles(conn, article_ids: List[int]) -> Dict[int, Optional[datetime]]:
    """
    Get publication dates for a list of article IDs.

    Args:
        conn: Database connection
        article_ids: List of article IDs

    Returns:
        Dict[int, Optional[datetime]]: Dictionary mapping article IDs to their publication dates
    """
    if not article_ids:
        return {}

    result = {}

    try:
        cursor = conn.cursor()

        # Create a placeholder string for the IN clause
        ids_placeholder = ', '.join(['%s'] * len(article_ids))

        # Query publication dates efficiently
        cursor.execute(f"""
            SELECT id, pub_date
            FROM articles
            WHERE id IN ({ids_placeholder})
        """, article_ids)

        # Build the result dictionary
        for row in cursor.fetchall():
            article_id, pub_date = row
            result[article_id] = pub_date

        cursor.close()

        # Initialize missing articles with None
        for article_id in article_ids:
            if article_id not in result:
                result[article_id] = None

        return result

    except Exception as e:
        logger.error(f"Error fetching publication dates: {e}")
        # Return dict with None values on error
        return {article_id: None for article_id in article_ids}


def get_recent_unprocessed_articles(conn, days: int = 2, limit: int = 2000) -> List[Dict[str, Any]]:
    """
    Get recent unprocessed articles.

    Args:
        conn: Database connection
        days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        List of article dictionaries
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, domain, cluster_id, content
            FROM articles
            WHERE extracted_entities = FALSE
            AND pub_date >= (CURRENT_DATE - INTERVAL '%s DAYS')
            ORDER BY pub_date DESC
            LIMIT %s
        """, (days, limit))

        columns = [desc[0] for desc in cursor.description]
        articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()

        return articles
    except Exception as e:
        logger.error(f"Error fetching recent unprocessed articles: {e}")
        return []


def mark_article_processed(conn, article_id: int) -> bool:
    """
    Mark an article as having had its entities extracted.

    Args:
        conn: Database connection
        article_id: ID of the article to mark as processed

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE articles
            SET extracted_entities = TRUE,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (article_id,))

        success = cursor.rowcount > 0
        conn.commit()
        cursor.close()
        return success
    except Exception as e:
        logger.error(f"Error marking article {article_id} as processed: {e}")
        conn.rollback()
        return False
