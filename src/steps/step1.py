#!/usr/bin/env python3
"""
step1.py - Step 1 of Data Refinery Pipeline

Step 1: Data Collection, Processing and Storage
- Fetches articles with 'ReadyForReview' status from news-db
- Processes article content (noise removal, standardization)
- Validates content and flags invalid articles
- Inserts data into reader-db with complete metadata
- Leaves additional processing for embedding and clustering to future steps

Related files:
- src/main.py: Orchestrates all pipeline steps
- src/database/news_api_client.py: Client for fetching articles
- src/database/reader_db_client.py: Client for storing processed articles
- src/refinery/content_processor.py: Functions for processing article content
- src/gemini/gemini_client.py: Client for generating embeddings
"""
from src.database.news_api_client import NewsAPIClient
from src.database.reader_db_client import ReaderDBClient
from src.refinery.content_processor import process_article_content, validate_and_prepare_for_storage
import logging
import sys
import os
import time
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from dotenv import load_dotenv
import concurrent.futures
from multiprocessing import cpu_count
from threading import Lock

# Import for Step 1.6: Embedding Generation
from src.gemini.gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_project_root():
    """
    Get the absolute path to the project root directory.

    Returns:
        str: Absolute path to the project root
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def fetch_articles_for_processing() -> List[Dict[str, Any]]:
    """
    STEP 1.1: Data Collection
    Fetch articles with proceeding_status = 'ReadyForReview' from news-db,
    including all necessary fields for processing (id, title, content, pub_date, domain)

    Returns:
        List[Dict[str, Any]]: List of articles with complete data for processing
    """
    # Get News API base URL from environment or use default with host.docker.internal
    news_api_url = os.getenv(
        "NEWS_API_BASE_URL", "http://host.docker.internal:8000")

    # Connect to the News API
    logger.info(f"Connecting to News API at {news_api_url}")
    news_client = NewsAPIClient(api_base_url=news_api_url)

    try:
        # Test connection first
        test_result = news_client.test_connection()
        if "error" in test_result:
            logger.warning(
                f"Connection test to News API failed: {test_result['error']}")
            logger.warning("Will try to retrieve articles anyway...")
        else:
            logger.info(
                f"Successfully connected to News API: {test_result.get('api_status', 'unknown')}")

        # Get articles with 'ReadyForReview' status
        articles = news_client.get_articles_ready_for_review()

        logger.info(
            f"STEP 1.1 COMPLETE: Retrieved {len(articles)} articles for processing")
        return articles

    except Exception as e:
        logger.error(f"STEP 1.1 FAILED: Error fetching articles: {e}")
        return []

    finally:
        news_client.close()


def initialize_reader_db_client() -> ReaderDBClient:
    """
    Initialize and return a new ReaderDBClient using environment variables.

    Returns:
        ReaderDBClient: A new initialized database client
    """
    # Get reader database connection parameters from environment variables
    reader_db_host = os.getenv("READER_DB_HOST", "postgres")
    reader_db_port = int(os.getenv("READER_DB_PORT", "5432"))
    reader_db_name = os.getenv("READER_DB_NAME", "reader_db")
    reader_db_user = os.getenv("READER_DB_USER", "postgres")
    reader_db_password = os.getenv("READER_DB_PASSWORD", "postgres")

    # Log connection details (without password)
    logger.info(
        f"Connecting to Reader DB at {reader_db_host}:{reader_db_port}/{reader_db_name} as {reader_db_user}")

    # Connect to reader-db
    return ReaderDBClient(
        host=reader_db_host,
        port=reader_db_port,
        dbname=reader_db_name,
        user=reader_db_user,
        password=reader_db_password
    )


def get_processed_scraper_ids(reader_client: ReaderDBClient) -> Set[int]:
    """
    Get a set of scraper_ids that have already been processed.

    Args:
        reader_client: An initialized ReaderDBClient

    Returns:
        Set[int]: Set of scraper_ids for articles that have already been processed
    """
    conn = None
    try:
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        # Query for articles that have already been processed or are problematic
        cursor.execute("""
            SELECT scraper_id 
            FROM articles 
            WHERE (processed_at IS NOT NULL OR content = 'ERROR')
            AND scraper_id IS NOT NULL;
        """)

        result = cursor.fetchall()
        # Extract scraper_ids and convert to set for fast lookups
        processed_ids = {row[0] for row in result if row[0] is not None}

        logger.info(
            f"Found {len(processed_ids)} already processed or problematic articles")
        return processed_ids

    except Exception as e:
        logger.error(f"Error retrieving processed article IDs: {e}")
        return set()

    finally:
        if conn:
            reader_client.release_connection(conn)


def insert_processed_articles(processed_articles: list) -> int:
    """
    Insert processed articles into the reader database.

    Args:
        processed_articles: List of processed article dictionaries to insert

    Returns:
        int: Number of successfully inserted articles
    """
    if not processed_articles:
        logger.warning("No processed articles to insert")
        return 0

    # Use the existing function with a new connection
    # This maintains backward compatibility for other code calling this function
    reader_client = initialize_reader_db_client()

    try:
        # Insert articles into the database
        count = reader_client.batch_insert_articles(processed_articles)
        logger.info(f"Inserted {count} articles into the database")
        return count
    except Exception as e:
        logger.error(f"Error inserting articles into the database: {e}")
        return 0
    finally:
        reader_client.close()


def output_error_articles_json(reader_client: ReaderDBClient) -> None:
    """
    Identify articles with null or error content and output them to a JSON file for debugging.

    Args:
        reader_client: An initialized ReaderDBClient instance
    """
    logger.info("Checking for articles with null/error content...")

    try:
        # Get articles with errors using the client method
        error_articles = reader_client.get_error_articles()

        if not error_articles:
            logger.info("No articles with null/error content found")
            return

        # Convert datetime objects to strings for JSON serialization
        for article in error_articles:
            for key, value in article.items():
                if isinstance(value, datetime.datetime):
                    article[key] = value.isoformat()

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join("/app", "logs")

        # Make sure logs directory exists
        os.makedirs(logs_dir, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            logs_dir, f"error_articles_{timestamp}.json")

        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump({"error_articles": error_articles,
                      "count": len(error_articles)}, f, indent=2)

        logger.info(f"Error articles debug info written to {output_file}")

    except Exception as e:
        logger.error(f"Error creating error articles debug file: {e}")


def process_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single article's content and prepare it for storage.

    This function is designed to be run in parallel by multiple workers.

    Args:
        article: Raw article data dictionary

    Returns:
        Dict[str, Any]: Processed article data ready for storage
    """
    # Extract article ID for logging
    article_id = article.get('id')

    # Extract raw content
    raw_content = article.get('content')

    if not raw_content:
        logger.warning(
            f"Article {article_id} has no content, marking as invalid")
        # Return with error status
        return {
            'scraper_id': article_id,
            'title': article.get('title', f"Article {article_id}"),
            'pub_date': article.get('pub_date'),
            'domain': article.get('domain', ''),
            'content': 'ERROR',
            'processed_at': datetime.datetime.now()  # Set current time for processed_at
        }

    # Process the content
    try:
        processed_content = process_article_content(raw_content)
        # Validate and prepare for storage
        processed_article = validate_and_prepare_for_storage(
            article, processed_content)

        # Ensure processed_at is set to now if not already set
        if 'processed_at' not in processed_article or processed_article['processed_at'] is None:
            processed_article['processed_at'] = datetime.datetime.now()

        return processed_article
    except Exception as e:
        logger.error(f"Error processing article {article_id}: {e}")
        # Return with error status
        return {
            'scraper_id': article_id,
            'title': article.get('title', f"Article {article_id}"),
            'pub_date': article.get('pub_date'),
            'domain': article.get('domain', ''),
            'content': 'ERROR',
            'processed_at': datetime.datetime.now()  # Set current time for processed_at
        }


def run(max_workers: int = None) -> int:
    """
    Main function executing Step 1 of the Data Refinery Pipeline with parallel processing:
    - Collect article data from news-db (Step 1.1)
    - Pre-process: Filter out already processed articles
    - Process article content in parallel to remove noise and standardize (Step 1.2)
    - Validate article content and prepare for storage (Step 1.3)
    - Store processed articles in reader-db incrementally (Step 1.4)
    - Output debug information for articles with errors (Step 1.5)
    - Generate embeddings for suitable articles (Step 1.6)

    Args:
        max_workers: Maximum number of worker processes to use for parallel processing.
                    If None, defaults to number of CPU cores.

    Returns:
        int: Number of successfully processed articles, or 0 if the step failed
    """
    # Determine number of worker processes
    if max_workers is None:
        # Use CPU count with a minimum of 2 and maximum of 16 workers
        max_workers = min(max(cpu_count(), 2), 16)

    # Configure incremental database update settings
    batch_size = int(os.getenv("DB_UPDATE_BATCH_SIZE", "20")
                     )  # Update DB every N articles
    # Also update every N seconds
    checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL_SECONDS", "60"))

    logger.info(
        f"========= STARTING STEP 1: DATA COLLECTION, PROCESSING AND STORAGE (USING {max_workers} WORKERS) =========")
    logger.info(
        f"Will update database every {batch_size} articles or {checkpoint_interval} seconds, whichever comes first")

    # Wait for services to be ready
    wait_seconds = int(os.getenv("WAIT_SECONDS", "5"))
    logger.info(f"Waiting {wait_seconds} seconds for services to be ready...")
    time.sleep(wait_seconds)

    # STEP 1.1: Fetch articles from news-db
    raw_articles = fetch_articles_for_processing()

    if not raw_articles:
        logger.warning("No articles retrieved. Step 1 cannot proceed.")
        return 0

    # PRE-PROCESSING: Get list of already processed articles
    reader_client = initialize_reader_db_client()
    try:
        processed_ids = get_processed_scraper_ids(reader_client)
    finally:
        reader_client.close()

    # Filter out already processed articles
    articles_to_process = []
    for article in raw_articles:
        article_id = article.get('id')
        if article_id not in processed_ids:
            articles_to_process.append(article)

    skipped_count = len(raw_articles) - len(articles_to_process)
    logger.info(f"Filtered out {skipped_count} already processed articles")
    logger.info(f"Remaining articles to process: {len(articles_to_process)}")

    if not articles_to_process:
        logger.info(
            "All articles have already been processed. Step 1 complete.")
        return 0

    # Initialize a DB client that will be kept open for the duration of processing
    # This avoids the overhead of creating connections for each batch
    reader_db = initialize_reader_db_client()

    try:
        # Test DB connection
        test_result = reader_db.test_connection()
        if "error" in test_result:
            logger.error(
                f"Cannot connect to reader-db: {test_result['error']}")
            return 0

        logger.info(
            f"DB Connection verified, available tables: {', '.join(test_result.get('tables', []))}")

        # STEP 1.2 & 1.3: Process articles and prepare for storage in parallel
        logger.info(
            f"Processing {len(articles_to_process)} articles with {max_workers} parallel workers...")
        start_time = time.time()
        last_db_update_time = start_time

        # Track processing statistics
        total_articles = len(articles_to_process)
        processed_count = 0
        inserted_count = 0
        pending_articles = []  # Articles ready for DB insertion

        # Initialize a thread-safe lock for concurrent access to the pending_articles list
        pending_articles_lock = Lock()

        def update_database():
            """Helper function to update the database with pending articles"""
            nonlocal inserted_count, pending_articles

            if not pending_articles:
                return 0

            with pending_articles_lock:
                articles_to_insert = pending_articles.copy()
                pending_articles = []

            if articles_to_insert:
                logger.info(
                    f"Performing incremental database update with {len(articles_to_insert)} articles...")
                # Use batch insert method directly on the persistent connection
                count = reader_db.batch_insert_articles(articles_to_insert)
                inserted_count += count
                return count
            return 0

        # Use thread pool for I/O bound tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all articles for processing
            future_to_article = {executor.submit(
                process_article, article): article for article in articles_to_process}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_article):
                article = future_to_article[future]
                processed_count += 1

                try:
                    processed_article = future.result()

                    # Add to pending articles for database insertion
                    with pending_articles_lock:
                        pending_articles.append(processed_article)

                    # Check if it's time for a database update
                    current_time = time.time()
                    time_since_last_update = current_time - last_db_update_time

                    if len(pending_articles) >= batch_size or time_since_last_update >= checkpoint_interval:
                        update_count = update_database()
                        last_db_update_time = time.time()
                        logger.info(
                            f"Incremental update: inserted {update_count} articles")

                    # Log progress periodically
                    if processed_count % 10 == 0 or processed_count == total_articles:
                        elapsed = current_time - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {processed_count}/{total_articles} articles "
                                    f"({processed_count/total_articles*100:.1f}%, {rate:.2f} articles/sec)")

                except Exception as e:
                    logger.error(
                        f"Exception processing article {article.get('id')}: {e}")

        # Final database update for any remaining articles
        if pending_articles:
            final_count = update_database()
            logger.info(
                f"Final database update: inserted {final_count} articles")

        elapsed_time = time.time() - start_time
        logger.info(f"STEP 1.2 & 1.3 COMPLETE: Processed {processed_count} articles "
                    f"in {elapsed_time:.2f} seconds ({processed_count/elapsed_time:.2f} articles/sec)")
        logger.info(f"Total articles inserted into database: {inserted_count}")

        # STEP 1.5: Generate debug information for articles with errors
        output_error_articles_json(reader_db)

        if inserted_count > 0:
            logger.info(
                f"========= STEP 1 COMPLETE: {inserted_count} of {len(articles_to_process)} NEW articles processed and stored =========")
            logger.info(
                f"Total articles considered: {len(raw_articles)} (including {skipped_count} previously processed)")

            # STEP 1.6: Generate embeddings for suitable articles
            logger.info(
                "========= STARTING STEP 1.6: EMBEDDING GENERATION =========")

            try:
                # Get articles that need embeddings (content < 1450 words)
                articles_for_embedding = reader_db.get_articles_needing_embedding()

                if not articles_for_embedding:
                    logger.info(
                        "No articles need embeddings. Step 1.6 complete.")
                else:
                    logger.info(
                        f"Found {len(articles_for_embedding)} articles suitable for embedding generation")

                    # Track embedding statistics
                    embedding_success = 0
                    embedding_failure = 0

                    # Initialize a single Gemini client for all embedding tasks
                    gemini_client = GeminiClient()
                    logger.info(
                        "Initialized Gemini client for embedding generation")

                    # Define worker function for embedding generation
                    def process_embedding_task(article_data: Dict[str, Any]) -> Tuple[int, bool]:
                        article_id = article_data['id']
                        content = article_data['content']

                        try:
                            # Generate embedding
                            embedding = gemini_client.generate_embedding(
                                content)

                            if embedding is not None:
                                # Insert embedding into database
                                result = reader_db.insert_embedding(
                                    article_id, embedding)
                                if result:
                                    return (article_id, True)
                                else:
                                    logger.error(
                                        f"Failed to insert embedding for article {article_id}")
                            else:
                                logger.error(
                                    f"Failed to generate embedding for article {article_id}")

                            return (article_id, False)

                        except Exception as e:
                            logger.error(
                                f"Error in embedding process for article {article_id}: {e}")
                            return (article_id, False)

                    # Process embeddings in parallel
                    embedding_start_time = time.time()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit embedding tasks
                        future_to_article = {executor.submit(process_embedding_task, article): article
                                             for article in articles_for_embedding}

                        # Process results as they complete
                        for i, future in enumerate(concurrent.futures.as_completed(future_to_article)):
                            article = future_to_article[future]

                            try:
                                article_id, success = future.result()

                                if success:
                                    embedding_success += 1
                                else:
                                    embedding_failure += 1

                                # Log progress periodically
                                if (i + 1) % 50 == 0 or (i + 1) == len(articles_for_embedding):
                                    logger.info(f"Embedding progress: {i+1}/{len(articles_for_embedding)} "
                                                f"({(i+1)/len(articles_for_embedding)*100:.1f}%)")

                            except Exception as e:
                                logger.error(
                                    f"Exception processing embedding task: {e}")
                                embedding_failure += 1

                    embedding_elapsed_time = time.time() - embedding_start_time
                    logger.info(f"STEP 1.6 COMPLETE: Processed {embedding_success + embedding_failure} embeddings "
                                f"in {embedding_elapsed_time:.2f} seconds")
                    logger.info(
                        f"Embedding results: {embedding_success} successful, {embedding_failure} failed")

            except Exception as e:
                logger.error(f"Error during embedding generation step: {e}")

            return inserted_count
        else:
            logger.error(
                "========= STEP 1 FAILED: No articles were inserted =========")
            return 0

    finally:
        # Always close the DB connection
        if reader_db:
            reader_db.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    # If run directly, execute Step 1
    # Allow setting max_workers via environment variable
    max_workers = os.getenv("MAX_WORKERS")
    if max_workers:
        try:
            max_workers = int(max_workers)
        except ValueError:
            max_workers = None

    run(max_workers=max_workers)
