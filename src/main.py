#!/usr/bin/env python3
"""
main.py - Step 1 of Data Refinery Pipeline

Step 1: Data Collection and Initial Storage
- Fetches article IDs with 'ReadyForReview' status from news-db 
- Inserts them into reader-db as scraper_id with essential metadata (id, title, pub_date, domain)
- Leaves content, processed_at, is_hot, and cluster_id as NULL for future steps

Future steps in the pipeline will:
- Fetch complete article content for enrichment (Step 2)
- Extract entities and generate embeddings (Step 3)
- Cluster articles and generate summaries (Step 4)
"""
from database.news_api_client import NewsAPIClient
from database.reader_db_client import ReaderDBClient
import logging
import sys
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Load environment variables
load_dotenv()

# Import database clients after adding to path


def fetch_articles_minimal_data() -> List[Dict[str, Any]]:
    """
    STEP 1.1: Data Collection
    Fetch only essential fields (id, title, pub_date, domain) from news-db 
    where proceeding_status = 'ReadyForReview'

    Returns:
        List[Dict[str, Any]]: List of articles with minimal data
    """
    # Connect to the News API (try local connection first)
    news_client = NewsAPIClient(api_base_url="http://localhost:8000")

    try:
        # Get articles with 'ReadyForReview' status
        full_articles = news_client.get_articles_ready_for_review()

        # Extract only the essential fields
        minimal_articles = []
        for article in full_articles:
            minimal_articles.append({
                'id': article.get('id'),
                'title': article.get('title', f"Article {article.get('id')}"),
                'pub_date': article.get('pub_date'),
                'domain': article.get('domain', '')
            })

        logger.info(
            f"STEP 1.1 COMPLETE: Retrieved {len(minimal_articles)} articles with minimal data")
        return minimal_articles

    except Exception as e:
        logger.error(f"STEP 1.1 FAILED: Error fetching articles: {e}")
        return []

    finally:
        news_client.close()


def insert_articles_minimal_data(articles: List[Dict[str, Any]]) -> int:
    """
    STEP 1.2: Initial Storage
    Insert articles into reader-db with only essential fields filled.
    Leave content, processed_at, is_hot, and cluster_id as NULL for future steps.

    Args:
        articles: List of articles with minimal data to insert

    Returns:
        int: Number of successfully inserted records
    """
    if not articles:
        logger.info("STEP 1.2 SKIPPED: No articles to insert")
        return 0

    logger.info(
        f"STEP 1.2 STARTED: Preparing to insert {len(articles)} articles into reader-db...")

    # Connect to reader-db (using local connection parameters)
    reader_client = ReaderDBClient(
        host="localhost",
        port=5433,
        dbname="reader_db",
        user="postgres",
        password="postgres"
    )

    successful_inserts = 0
    failed_inserts = 0

    try:
        # First verify connection and tables
        test_result = reader_client.test_connection()
        if "error" in test_result:
            logger.error(
                f"STEP 1.2 FAILED: Cannot connect to reader-db: {test_result['error']}")
            return 0

        logger.info(
            f"DB Connection verified, available tables: {', '.join(test_result.get('tables', []))}")

        # Check if content is required in articles table
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        # Alter table to make content nullable if necessary
        try:
            cursor.execute("""
                ALTER TABLE articles 
                ALTER COLUMN content DROP NOT NULL;
            """)
            conn.commit()
            logger.info(
                "Modified articles table to make content field nullable")

            # Also alter the processed_at column to be nullable
            cursor.execute("""
                ALTER TABLE articles 
                ALTER COLUMN processed_at DROP NOT NULL;
            """)
            conn.commit()
            logger.info(
                "Modified articles table to make processed_at field nullable")
        except Exception as e:
            logger.error(f"Error altering table: {e}")
            # Continue anyway - might fail if already nullable
            conn.rollback()

        reader_client.release_connection(conn)

        # Process and insert articles in batches
        batch_size = 100
        total_batches = (len(articles) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(articles))
            current_batch = articles[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_num+1}/{total_batches} ({len(current_batch)} articles)")

            conn = None
            try:
                conn = reader_client.get_connection()
                cursor = conn.cursor()

                for article in current_batch:
                    # Insert only essential fields, explicitly setting others to NULL
                    cursor.execute("""
                        INSERT INTO articles (
                            scraper_id, title, pub_date, domain, content, processed_at
                        )
                        VALUES (%s, %s, %s, %s, NULL, NULL)
                        ON CONFLICT (scraper_id) DO UPDATE
                        SET title = EXCLUDED.title,
                            pub_date = EXCLUDED.pub_date,
                            domain = EXCLUDED.domain
                        RETURNING id;
                    """, (
                        article.get('id'),
                        article.get('title', f"Article {article.get('id')}"),
                        article.get('pub_date'),
                        article.get('domain', '')
                    ))

                    new_id = cursor.fetchone()[0]
                    if new_id:
                        successful_inserts += 1
                    else:
                        failed_inserts += 1

                conn.commit()
                if successful_inserts % 100 == 0:
                    logger.info(
                        f"Inserted {successful_inserts} articles so far")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num+1}: {e}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    reader_client.release_connection(conn)

        logger.info(
            f"STEP 1.2 COMPLETE: Inserted {successful_inserts} articles, {failed_inserts} failed")
        return successful_inserts

    except Exception as e:
        logger.error(f"STEP 1.2 FAILED: Error inserting articles: {e}")
        return successful_inserts

    finally:
        reader_client.close()


def main():
    """
    Main function executing Step 1 of the Data Refinery Pipeline:
    - Collect essential article data from news-db (Step 1.1)
    - Store articles in reader-db with only essential fields (Step 1.2)

    This creates the foundation for future refinement steps.
    Content, processing timestamps, and clustering will be handled in later steps.
    """
    logger.info(
        "========= STARTING STEP 1: DATA COLLECTION AND INITIAL STORAGE =========")

    # STEP 1.1: Fetch articles with minimal data from news-db
    articles = fetch_articles_minimal_data()

    if not articles:
        logger.warning("No articles retrieved. Step 1 cannot proceed.")
        sys.exit(0)

    # STEP 1.2: Insert articles with minimal data into reader-db
    inserted_count = insert_articles_minimal_data(articles)

    if inserted_count > 0:
        logger.info(
            f"========= STEP 1 COMPLETE: {inserted_count} of {len(articles)} articles prepared for refinement =========")
    else:
        logger.error(
            "========= STEP 1 FAILED: No articles were inserted =========")


if __name__ == "__main__":
    main()
