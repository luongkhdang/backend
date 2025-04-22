#!/usr/bin/env python3
"""
step1.py - Step 1 of Data Refinery Pipeline

Step 1: Data Collection, Processing and Storage
- Fetches articles with 'ReadyForReview' status from news-db
- Processes article content (noise removal, standardization)
- Validates content and flags invalid articles
- Inserts data into reader-db with complete metadata
- Leaves additional processing for clustering to future steps

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
import importlib.util

# Import for Step 1.6: Embedding Generation
from src.gemini.gemini_client import GeminiClient
# Import RateLimiter
from src.utils.rate_limit import RateLimiter
# Import LocalNLPClient conditionally - will be checked at runtime
# from src.localnlp.localnlp_client import LocalNLPClient
from typing import Optional  # Add Optional type hint

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import for Step 1.7: Chunking and Summarization
# Check if spaCy is installed
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None
if not SPACY_AVAILABLE:
    logger.warning(
        "spaCy package not installed. Step 1.7 (summarization) requires spaCy for chunking.")
else:
    import spacy

# Check if transformers is installed
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
if not TRANSFORMERS_AVAILABLE:
    logger.warning(
        "Transformers package not installed. Step 1.7 (summarization) will be disabled.")

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
    # logger.info( # Removed DB connection log
    #     f"Connecting to Reader DB at {reader_db_host}:{reader_db_port}/{reader_db_name} as {reader_db_user}")

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

        # logger.info( # Changed to DEBUG
        #     f"Found {len(processed_ids)} already processed or problematic articles")
        logger.debug(
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
        result = reader_client.batch_insert_articles(processed_articles)
        # Extract success count from the result dictionary
        if isinstance(result, dict):
            return result.get("success", 0)
        # For backward compatibility, in case it returns an int
        return result if isinstance(result, int) else 0
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
    # logger.info("Checking for articles with null/error content...") # Changed to DEBUG
    logger.debug("Checking for articles with null/error content...")

    try:
        # Get articles with errors using the client method
        error_articles = reader_client.get_error_articles()

        if not error_articles:
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


def run(max_workers: int = None) -> Dict[str, Any]:
    """
    Main function to run Step 1: Data Collection, Processing, and Storage.

    Args:
        max_workers (int): Maximum number of workers for parallel processing.

    Returns:
        Dict[str, Any]: Summary status dictionary for Step 1.
    """
    # Initialize status dictionary
    overall_status = {
        "step1.1_status": None,
        "step1.2_status": None,
        "step1.3_status": None,
        "step1.4_status": None,
        "step1.5_status": None,
        "step1.6_status": None,
        "step1.7_status": None,
        "total_runtime_seconds": 0,
        "error": None
    }

    start_time = time.time()

    try:
        # Load configuration
        cfg = config.load_step1_config()

        # Determine max_workers
        if max_workers is None:
            max_workers = cfg.max_workers
        logger.info(f"Using {max_workers} workers for Step 1.")

        # Step 1.1: Data Collection
        logger.info("Starting Step 1.1: Data Collection from News API")
        collection_start = time.time()
        api_client = NewsAPIClient()
        all_articles_metadata = api_client.get_all_articles_metadata(
            processed_since=cfg.process_articles_since)
        collection_time = time.time() - collection_start
        if all_articles_metadata:
            logger.info(
                f"Step 1.1: Found {len(all_articles_metadata)} articles to potentially process.")
        else:
            logger.warning("Step 1.1: No articles found from News API.")
            overall_status["step1.1_status"] = {
                "success": True, "count": 0, "runtime": collection_time}
            # If no articles, can potentially skip subsequent steps
            overall_status["total_runtime_seconds"] = time.time() - start_time
            return overall_status

        overall_status["step1.1_status"] = {"success": True, "count": len(
            all_articles_metadata), "runtime": collection_time}

        # Initialize DB Client
        db_client = ReaderDBClient()

        # Step 1.2: Content Processing (Fetch full content)
        logger.info("Starting Step 1.2: Article Content Processing")
        processing_start = time.time()
        # Ensure it passes cfg.max_workers
        articles_with_content = processing.process_articles_concurrently(
            api_client, all_articles_metadata, db_client, cfg.db_update_batch_size, cfg.checkpoint_interval_seconds, max_workers)
        processing_time = time.time() - processing_start
        processed_count = len(articles_with_content)
        logger.info(
            f"Step 1.2: Fetched content for {processed_count} articles.")
        overall_status["step1.2_status"] = {
            "success": True, "count": processed_count, "runtime": processing_time}

        # Filter out articles that failed processing (content is None or ERROR)
        valid_articles = [a for a in articles_with_content if a.get(
            'content') and a.get('content') != 'ERROR']
        failed_count = processed_count - len(valid_articles)
        if failed_count > 0:
            logger.warning(
                f"Step 1.2: {failed_count} articles failed content fetching.")

        if not valid_articles:
            logger.warning(
                "No valid articles after content processing. Ending Step 1.")
            overall_status["total_runtime_seconds"] = time.time() - start_time
            return overall_status

        # Step 1.3: Content Validation (Placeholder)
        logger.info("Starting Step 1.3: Article Content Validation")
        validation_start = time.time()
        validated_articles = validation.validate_articles(valid_articles)
        validation_time = time.time() - validation_start
        validated_count = len(validated_articles)
        logger.info(f"Step 1.3: Validated {validated_count} articles.")
        overall_status["step1.3_status"] = {
            "success": True, "count": validated_count, "runtime": validation_time}

        if not validated_articles:
            logger.warning("No articles passed validation. Ending Step 1.")
            overall_status["total_runtime_seconds"] = time.time() - start_time
            return overall_status

        # Step 1.4: Incremental Database Storage
        logger.info("Starting Step 1.4: Incremental Database Storage")
        storage_start = time.time()
        storage_status = storage.store_articles(
            db_client, validated_articles, cfg.db_update_batch_size)
        storage_time = time.time() - storage_start
        logger.info(
            f"Step 1.4: Storage results - Success: {storage_status['success']}, Failure: {storage_status['failure']}")
        # Update overall status with detailed storage info
        storage_status['runtime'] = storage_time
        overall_status["step1.4_status"] = storage_status
        # Add simple inserted count for main.py logging
        overall_status["step1.4_inserted"] = storage_status.get('success', 0)

        # Step 1.5: Error Reporting
        logger.info("Starting Step 1.5: Error Reporting")
        reporting_start = time.time()
        error_articles = reporting.get_error_articles(db_client)
        reporting.save_error_report(error_articles, cfg.error_report_path)
        reporting_time = time.time() - reporting_start
        logger.info(
            f"Step 1.5: Found {len(error_articles)} articles with errors. Report saved to {cfg.error_report_path}")
        overall_status["step1.5_status"] = {"success": True, "count": len(
            error_articles), "runtime": reporting_time}

        # Step 1.6: Embedding Generation
        logger.info("Starting Step 1.6: Embedding Generation")
        embedding_start = time.time()
        # Combine articles needing direct embedding and those needing summarization first
        articles_for_direct_embedding = db_client.get_articles_needing_embedding(
            max_char_count=cfg.max_chars_for_embedding
        )
        articles_needing_summarization = db_client.get_articles_needing_summarization(
            min_char_count=cfg.max_chars_for_embedding
        )
        logger.info(
            f"Found {len(articles_for_direct_embedding)} articles for direct embedding.")
        logger.info(
            f"Found {len(articles_needing_summarization)} articles needing summarization before embedding.")

        # Use Gemini for all embedding generation
        gemini_client = GeminiClient()
        embedding_status_direct = embedding.generate_embeddings_gemini(
            db_client,
            gemini_client,
            articles_for_direct_embedding,
            batch_size=cfg.embedding_batch_size
        )
        embedding_time = time.time() - embedding_start
        logger.info(
            f"Step 1.6 (Direct): Generated embeddings - Success: {embedding_status_direct['success']}, Failure: {embedding_status_direct['failure']}")

        # Store status for Step 1.6 (direct embeddings)
        overall_status["step1.6_status"] = {
            "success": embedding_status_direct['success'],
            "failure": embedding_status_direct['failure'],
            "runtime": embedding_time
        }

        # Step 1.7: Local NLP Summarization for long articles (now runs unconditionally)
        logger.info("Starting Step 1.7: Local NLP Summarization (if needed)")
        summarization_start = time.time()
        if articles_needing_summarization:
            summarization_status = summarization.summarize_and_embed_long_articles(
                db_client,
                gemini_client,
                articles_needing_summarization,
                batch_size=cfg.embedding_batch_size,
                max_workers=max_workers
            )
            summarization_time = time.time() - summarization_start
            logger.info(f"Step 1.7: Summarized and embedded {summarization_status.get('processed', 0)} long articles. "
                        f"Success: {summarization_status.get('embeddings_success', 0)}, Failure: {summarization_status.get('embeddings_failure', 0)}")
            overall_status["step1.7_status"] = {
                "summarized_count": summarization_status.get('processed', 0),
                "embeddings_success": summarization_status.get('embeddings_success', 0),
                "embeddings_failure": summarization_status.get('embeddings_failure', 0),
                "runtime": summarization_time
            }
        else:
            logger.info("Step 1.7: No articles needed summarization.")
            overall_status["step1.7_status"] = {
                "success": True, "count": 0, "runtime": 0}

    except Exception as e:
        logger.error(f"Critical error in Step 1: {e}", exc_info=True)
        overall_status["error"] = str(e)
    finally:
        if 'db_client' in locals() and db_client:
            db_client.close()
            logger.debug("Database connection closed in Step 1.")

    overall_status["total_runtime_seconds"] = time.time() - start_time
    logger.info(
        f"Step 1 completed in {overall_status['total_runtime_seconds']:.2f} seconds.")
    return overall_status


# Example usage:
if __name__ == "__main__":
    status_report = run()
    print("\n--- Step 1 Summary Report ---")
    print(json.dumps(status_report, indent=2))
