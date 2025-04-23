"""
articles.py - Article-related database operations

This module provides functions for article-related database operations,
including inserting, updating, and querying articles.

The module now supports storing narrative frame phrases (frame_phrases) for articles,
which are extracted during entity processing in Step 3 of the data refinery pipeline.
These frame phrases represent the dominant narrative framing used in each article.

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
- get_all_unprocessed_articles(conn, limit: int = None) -> List[Dict[str, Any]]
  Gets all unprocessed articles without date limitation
- mark_article_processed(conn, article_id: int) -> bool
  Marks an article as having had its entities extracted
- update_article_frames_and_mark_processed(conn, article_id: int, frame_phrases: Optional[List[str]]) -> bool
  Updates an article's frame_phrases and marks it as having had its entities extracted
- get_recent_day_unprocessed_articles(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets unprocessed articles published yesterday and today only
- get_recent_day_processed_articles_with_details(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]
  Gets processed articles published yesterday or today with domain goodness scores
- mark_duplicate_articles_as_error(conn) -> int
  Finds articles with duplicate content (excluding those already marked 'ERROR') and updates their content to 'ERROR' and sets processed_at timestamp

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
    Update multiple article cluster assignments and hotness flags in batch using unnest for parameterization.

    Args:
        conn: Database connection
        assignments: List of tuples (article_id, cluster_id, is_hot) where cluster_id can be None

    Returns:
        Dict[str, int]: Dictionary with success and failure counts
    """
    if not assignments:
        return {"success": 0, "failure": 0}

    success_count = 0
    failure_count = 0
    # Prepare data for unnest: Separate lists for each column type
    article_ids = [a[0] for a in assignments]
    cluster_ids = [a[1] for a in assignments]  # Can contain None
    is_hots = [a[2] for a in assignments]

    try:
        cursor = conn.cursor()

        # Use unnest for parameterization
        query = """
        UPDATE articles a
        SET cluster_id = u.cluster_id,
            is_hot = u.is_hot
        FROM unnest(%s::int[], %s::int[], %s::bool[]) AS u(article_id, cluster_id, is_hot)
        WHERE a.id = u.article_id;
        """
        # Note: We cast cluster_ids to int[] which works even with NULLs in the array.

        cursor.execute(query, (article_ids, cluster_ids, is_hots))
        # cursor.rowcount might give the number of updated rows if supported
        # Assuming success if no exception is raised for now
        updated_rows = cursor.rowcount if cursor.rowcount >= 0 else - \
            1  # -1 indicates rowcount not available/reliable

        if updated_rows == -1:
            # If rowcount is not available, assume success unless an error occurred (handled below)
            # This is optimistic but avoids false failure reports if rowcount is unsupported
            success_count = len(assignments)
            failure_count = 0
            logger.warning(
                f"Batch update rowcount not available. Assuming {success_count} successes based on lack of exceptions.")
        elif updated_rows != len(assignments):
            success_count = updated_rows
            failure_count = len(assignments) - success_count
            logger.warning(
                f"Batch update using unnest reported {updated_rows} updated rows for {len(assignments)} assignments. Potential failures or duplicates occurred.")
        else:
            success_count = updated_rows
            failure_count = 0

        # Update cluster article counts (This part remains the same, but adjusted variable names)
        try:
            cluster_id_set = set(cid for cid in cluster_ids if cid is not None)

            for cluster_id_val in cluster_id_set:
                cursor.execute("""
                    UPDATE clusters
                    SET article_count = (
                        SELECT COUNT(*)
                        FROM articles
                        WHERE cluster_id = %s
                    )
                    WHERE id = %s;
                """, (cluster_id_val, cluster_id_val))
        except Exception as e:
            logger.error(
                f"Error updating cluster article counts after batch article update: {e}")
            # Don't rollback the main article update for this error

        conn.commit()
        cursor.close()

        logger.info(
            f"Updated {success_count} article cluster assignments using unnest (estimated failures: {failure_count})")
        return {"success": success_count, "failure": failure_count}

    except Exception as e:
        logger.error(
            f"Failed to update cluster assignments using unnest: {e}", exc_info=True)
        conn.rollback()  # Rollback on main update failure
        # If the main update failed, all are considered failures
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


def get_all_unprocessed_articles(conn, limit: int = None) -> List[Dict[str, Any]]:
    """
    Get all unprocessed articles without date limitation.

    Args:
        conn: Database connection
        limit: Optional maximum number of articles to return (returns all if None)

    Returns:
        List of article dictionaries
    """
    try:
        cursor = conn.cursor()

        # Base query to get all unprocessed articles
        query = """
            SELECT id, domain, cluster_id, content, pub_date
            FROM articles
            WHERE extracted_entities = FALSE
            AND content IS NOT NULL
            AND content != 'ERROR'
            ORDER BY pub_date DESC
        """

        # Add limit if specified
        if limit is not None:
            query += " LIMIT %s"
            cursor.execute(query, (limit,))
        else:
            cursor.execute(query)

        columns = [desc[0] for desc in cursor.description]
        articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()

        logger.info(f"Fetched {len(articles)} unprocessed articles")
        return articles
    except Exception as e:
        logger.error(f"Error fetching all unprocessed articles: {e}")
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
                processed_at = CURRENT_TIMESTAMP
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


def update_article_frames_and_mark_processed(conn, article_id: int, frame_phrases: Optional[List[str]]) -> bool:
    """
    Update an article's frame_phrases and mark it as having had its entities extracted.

    Args:
        conn: Database connection
        article_id: ID of the article to update
        frame_phrases: List of narrative frame phrases extracted from the article, or None

    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # If frame_phrases is None, only mark as processed
        if frame_phrases is None:
            cursor.execute("""
                UPDATE articles
                SET extracted_entities = TRUE,
                    processed_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (article_id,))
        else:
            cursor.execute("""
                UPDATE articles
                SET extracted_entities = TRUE,
                    processed_at = CURRENT_TIMESTAMP,
                    frame_phrases = %s
                WHERE id = %s
            """, (frame_phrases, article_id))

        success = cursor.rowcount > 0
        conn.commit()
        cursor.close()
        return success
    except Exception as e:
        logger.error(f"Error updating frames for article {article_id}: {e}")
        conn.rollback()
        return False


def get_recent_day_unprocessed_articles(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get unprocessed articles published yesterday and today only.

    Args:
        conn: Database connection
        limit: Optional maximum number of articles to return (returns all if None)

    Returns:
        List of article dictionaries
    """
    try:
        cursor = conn.cursor()

        # Query to get unprocessed articles from yesterday and today only
        query = """
            SELECT id, domain, cluster_id, content, pub_date
            FROM articles
            WHERE extracted_entities = FALSE
            AND content IS NOT NULL
            AND content != 'ERROR'
            AND pub_date >= (CURRENT_DATE - INTERVAL '1 DAY')
            AND pub_date <= CURRENT_DATE
            ORDER BY pub_date DESC
        """

        # Add limit if specified
        if limit is not None:
            query += " LIMIT %s"
            cursor.execute(query, (limit,))
        else:
            cursor.execute(query)

        columns = [desc[0] for desc in cursor.description]
        articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()

        logger.info(
            f"Fetched {len(articles)} unprocessed articles from yesterday and today")
        return articles
    except Exception as e:
        logger.error(f"Error fetching recent day unprocessed articles: {e}")
        return []


def get_recent_day_processed_articles_with_details(conn, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get processed articles published yesterday or today with domain goodness scores.

    Args:
        conn: Database connection
        limit: Optional maximum number of articles to return (returns all if None)

    Returns:
        List of article dictionaries with domain goodness scores
    """
    try:
        cursor = conn.cursor()

        # Calculate the date range for 'yesterday' and 'today'
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        # Base query to select articles within the date range
        query = """
            SELECT a.id, a.scraper_id, a.title, a.content, a.pub_date, a.domain, 
                   a.processed_at, a.is_hot, a.cluster_id, d.goodness_score
            FROM articles a
            LEFT JOIN domain_statistics d ON a.domain = d.domain
            WHERE a.pub_date >= %s AND a.pub_date < %s
            AND a.processed_at IS NOT NULL AND a.content != 'ERROR'
            ORDER BY a.pub_date DESC
        """

        # Add limit if provided
        if limit:
            query += " LIMIT %s"
            cursor.execute(
                query, (yesterday, today + timedelta(days=1), limit))
        else:
            cursor.execute(query, (yesterday, today + timedelta(days=1)))

        columns = [desc[0] for desc in cursor.description]
        articles_list = [dict(zip(columns, row)) for row in cursor.fetchall()]

        cursor.close()
        return articles_list

    except Exception as e:
        logger.error(
            f"Error retrieving recent day processed articles with details: {e}")
        return []


def mark_duplicate_articles_as_error(conn) -> int:
    """
    Find articles with duplicate content (excluding those already marked 'ERROR')
    and update their content to 'ERROR' and set processed_at timestamp.

    Args:
        conn: Database connection

    Returns:
        int: Number of articles marked as duplicates
    """
    try:
        cursor = conn.cursor()

        # Find IDs of articles with duplicate content, excluding those already marked 'ERROR'
        find_duplicates_query = """
            SELECT id
            FROM articles
            WHERE content IN (
                SELECT content
                FROM articles
                WHERE content IS NOT NULL AND content != 'ERROR'
                GROUP BY content
                HAVING COUNT(*) > 1
            )
            AND content != 'ERROR';
        """
        cursor.execute(find_duplicates_query)
        duplicate_ids = [row[0] for row in cursor.fetchall()]

        if not duplicate_ids:
            logger.debug("No duplicate articles found to mark as ERROR.")
            cursor.close()
            return 0

        # Update the content and processed_at for the identified duplicate articles
        update_query = """
            UPDATE articles
            SET content = 'ERROR',
                processed_at = CURRENT_TIMESTAMP
            WHERE id = ANY(%s);
        """
        cursor.execute(update_query, (duplicate_ids,))
        updated_count = cursor.rowcount
        conn.commit()
        cursor.close()

        logger.info(
            f"Marked {updated_count} articles with duplicate content as 'ERROR'.")
        return updated_count

    except Exception as e:
        logger.error(f"Error marking duplicate articles: {e}", exc_info=True)
        conn.rollback()
        return 0
