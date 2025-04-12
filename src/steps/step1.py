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
import importlib.util

# Import for Step 1.6: Embedding Generation
from src.gemini.gemini_client import GeminiClient
# Import RateLimiter
from src.steps.rate_limit import RateLimiter
# Import LocalNLPClient conditionally - will be checked at runtime
# from src.localnlp.localnlp_client import LocalNLPClient
from typing import Optional  # Add Optional type hint

# Import for Step 1.7: Chunking and Summarization
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def run(max_workers: int = None) -> int:
    """
    Main function executing Step 1 of the Data Refinery Pipeline with parallel processing:
    - Collect article data from news-db (Step 1.1)
    - Pre-process: Filter out already processed articles
    - Process article content in parallel to remove noise and standardize (Step 1.2)
    - Validate article content and prepare for storage (Step 1.3)
    - Store processed articles in reader-db incrementally (Step 1.4)
    - Output debug information for articles with errors (Step 1.5)
    - Generate embeddings for suitable short articles (Step 1.6) - Always executed
    - Summarize and generate embeddings for long articles (Step 1.7) - Runs if enabled

    Args:
        max_workers: Maximum number of worker processes to use for parallel processing.
                    If None, defaults to number of CPU cores.

    Returns:
        int: Number of successfully processed articles (from step 1.4), or 0 if the step failed
    """
    # Determine number of worker processes
    if max_workers is None:
        max_workers = min(max(cpu_count(), 2), 16)

    # Configure incremental database update settings
    batch_size = int(os.getenv("DB_UPDATE_BATCH_SIZE", "20"))
    checkpoint_interval = int(os.getenv("CHECKPOINT_INTERVAL_SECONDS", "60"))

    # Configure Step 1.7 settings from environment variables
    run_step_1_7 = os.getenv("RUN_STEP_1_7", "true").lower() == "true"
    local_nlp_model = os.getenv("LOCAL_NLP_MODEL", "facebook/bart-large-cnn")
    # Reduce the default max summary length to avoid warnings - BART models typically have a 1024 token limit
    # and summarization outputs should be shorter than inputs
    max_summary_tokens = int(os.getenv("MAX_SUMMARY_TOKENS", "512"))
    min_summary_tokens = int(os.getenv("MIN_SUMMARY_TOKENS", "150"))

    # Configure chunking parameters for Step 1.7
    article_token_chunk_threshold = int(
        os.getenv("ARTICLE_TOKEN_CHUNK_THRESHOLD", "2000"))
    target_chunk_token_size = int(os.getenv("TARGET_CHUNK_TOKEN_SIZE", "1000"))
    chunk_max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "300"))
    chunk_min_tokens = int(os.getenv("CHUNK_MIN_TOKENS", "75"))

    logger.info(
        f"========= STARTING STEP 1: DATA COLLECTION, PROCESSING AND STORAGE (USING {max_workers} WORKERS) =========")

    # Wait for services to be ready
    wait_seconds = int(os.getenv("WAIT_SECONDS", "5"))
    time.sleep(wait_seconds)

    # STEP 1.1: Fetch articles from news-db
    raw_articles = fetch_articles_for_processing()

    processed_count = 0
    inserted_count = 0

    if raw_articles:
        # PRE-PROCESSING: Get list of already processed articles
        # Use a separate client for this check
        reader_client_pre = initialize_reader_db_client()
        try:
            processed_ids = get_processed_scraper_ids(reader_client_pre)
        finally:
            reader_client_pre.close()  # Close the connection immediately

        # Filter out already processed articles
        articles_to_process = [
            a for a in raw_articles if a.get('id') not in processed_ids]

        skipped_count = len(raw_articles) - len(articles_to_process)
        logger.info(
            f"Remaining articles to process: {len(articles_to_process)}")

        if articles_to_process:
            # Initialize a persistent DB client for processing and insertion steps
            reader_db = initialize_reader_db_client()
            try:
                # Test DB connection
                test_result = reader_db.test_connection()
                if "error" in test_result:
                    logger.error(
                        f"Cannot connect to reader-db: {test_result['error']}")
                    return 0

                # STEP 1.2 & 1.3: Process articles and prepare for storage in parallel
                logger.info(
                    f"Processing {len(articles_to_process)} articles with {max_workers} parallel workers...")
                start_time = time.time()
                last_db_update_time = start_time

                total_articles = len(articles_to_process)
                processed_count = 0  # Reset for this phase
                inserted_count = 0  # Reset for this phase
                # Articles ready for DB insertion (Step 1.4)
                pending_articles = []
                pending_articles_lock = Lock()

                def update_database_step1_4():
                    nonlocal inserted_count, pending_articles, last_db_update_time
                    if not pending_articles:
                        return 0
                    with pending_articles_lock:
                        articles_to_insert = pending_articles[:]  # Copy
                        pending_articles = []
                    if articles_to_insert:
                        count = reader_db.batch_insert_articles(
                            articles_to_insert)
                        inserted_count += count
                        last_db_update_time = time.time()
                        return count
                    return 0

                # Use thread pool for I/O bound tasks (content processing/validation)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_article = {executor.submit(
                        process_article, article): article for article in articles_to_process}
                    for future in concurrent.futures.as_completed(future_to_article):
                        article = future_to_article[future]
                        processed_count += 1
                        try:
                            processed_article = future.result()
                            with pending_articles_lock:
                                pending_articles.append(processed_article)
                            current_time = time.time()
                            time_since_last_update = current_time - last_db_update_time
                            if len(pending_articles) >= batch_size or time_since_last_update >= checkpoint_interval:
                                update_database_step1_4()
                        except Exception as e:
                            logger.error(
                                f"Exception processing article {article.get('id')} (Steps 1.2/1.3): {e}")

                # Final database update for Step 1.4
                if pending_articles:
                    final_count = update_database_step1_4()

                elapsed_time = time.time() - start_time
                logger.info(f"STEP 1.2 & 1.3 COMPLETE: Processed {processed_count} articles "
                            f"in {elapsed_time:.2f} seconds ({processed_count/elapsed_time:.2f} articles/sec)")
                logger.info(
                    f"STEP 1.4 COMPLETE: Total articles inserted into database: {inserted_count}")

                # STEP 1.5: Generate debug information for articles with errors
                output_error_articles_json(reader_db)

            except Exception as e:
                logger.error(
                    f"Critical error during steps 1.2-1.5: {e}", exc_info=True)
                # Ensure DB connection is closed even if steps 1.2-1.5 fail
                if reader_db:
                    reader_db.close()
                return inserted_count  # Return count so far

            finally:
                # Close the DB connection if it wasn't closed due to an error above
                # Check if reader_db is still initialized and has a pool (not None)
                if 'reader_db' in locals() and reader_db and reader_db.connection_pool:
                    reader_db.close()
        else:  # No new articles to process
            logger.info(
                "All articles have already been processed in previous runs (Steps 1.2-1.5).")
    else:  # No raw articles fetched
        logger.warning("No articles retrieved from news-db for processing.")

    # --- Embedding Generation Steps (Always run, use new DB connection) ---

    # Initialize RateLimiter and GeminiClient (these are needed for both 1.6 and 1.7)
    rate_limiter = RateLimiter()
    gemini_client = GeminiClient(rate_limiter=rate_limiter)

    # Create a new database connection specifically for embedding steps
    reader_db_embed = initialize_reader_db_client()

    try:
        # Test connection
        test_result = reader_db_embed.test_connection()
        if "error" in test_result:
            logger.error(
                f"Cannot connect to reader-db for embedding steps: {test_result['error']}")
            # Decide if we should return or continue. If DB is down, embedding fails anyway.
            return inserted_count  # Return count from step 1.4

        # --- STEP 1.6: Embedding Generation for Short Articles ---
        logger.info(
            "========= STARTING STEP 1.6: EMBEDDING GENERATION (Short Articles) =========")

        # This filters articles by character count internally
        max_char_limit = 8000
        articles_for_embedding = reader_db_embed.get_articles_needing_embedding(
            max_char_count=max_char_limit)

        if not articles_for_embedding:
            logger.info(
                "No short articles need embeddings. Step 1.6 complete.")
        else:
            logger.info(
                f"Found {len(articles_for_embedding)} short articles for embedding generation")

            # Track embedding statistics for Step 1.6
            embedding_success_1_6 = 0
            embedding_failure_1_6 = 0
            pending_embeddings_1_6 = []
            pending_embeddings_lock_1_6 = Lock()
            embedding_checkpoint_interval = 60  # Reuse checkpoint interval
            embedding_batch_size = 20  # Reuse batch size

            embedding_start_time_1_6 = time.time()
            last_db_update_time_1_6 = embedding_start_time_1_6

            def update_embedding_database_1_6():
                nonlocal pending_embeddings_1_6, last_db_update_time_1_6, embedding_success_1_6
                # Use a lock to safely access the shared list
                with pending_embeddings_lock_1_6:
                    if not pending_embeddings_1_6:
                        return 0  # Nothing to insert
                    # Make a copy of the list to insert and clear the original
                    embeddings_to_insert = pending_embeddings_1_6[:]
                    pending_embeddings_1_6 = []

                # Perform the insertion outside the lock
                if embeddings_to_insert:
                    success_count_in_batch = 0
                    try:
                        # Loop through articles in the copied list
                        for article_id, embedding in embeddings_to_insert:
                            result = reader_db_embed.insert_embedding(
                                article_id, embedding
                            )
                            if result:
                                success_count_in_batch += 1
                        # Update total count and timestamp after processing the batch
                        embedding_success_1_6 += success_count_in_batch
                        last_db_update_time_1_6 = time.time()
                        return success_count_in_batch  # Return count for this batch
                    except Exception as e:
                        logger.error(
                            f"Error during batch insert (Step 1.6): {e}")
                        # Even if there's an error, update the total count with successes so far
                        embedding_success_1_6 += success_count_in_batch
                        last_db_update_time_1_6 = time.time()  # Update time anyway
                        return success_count_in_batch  # Return partial count
                else:
                    # This case should ideally not be reached if the check at the start works,
                    # but returning 0 is safe.
                    return 0

            def process_embedding_task_1_6(article_data: Dict[str, Any]) -> Tuple[int, bool]:
                nonlocal last_db_update_time_1_6
                article_id = article_data['id']
                content = article_data['content']
                try:
                    embedding = gemini_client.generate_embedding(content)
                    if embedding is not None and isinstance(embedding, list) and len(embedding) > 0:
                        with pending_embeddings_lock_1_6:
                            pending_embeddings_1_6.append(
                                (article_id, embedding))
                            current_time = time.time()
                            time_since_last_update = current_time - last_db_update_time_1_6
                            should_update = (len(pending_embeddings_1_6) >= embedding_batch_size or
                                             time_since_last_update >= embedding_checkpoint_interval)
                        if should_update:
                            update_embedding_database_1_6()
                        return (article_id, True)
                    else:
                        logger.error(
                            f"Failed to generate valid embedding for short article {article_id}")
                        return (article_id, False)
                except Exception as e:
                    logger.error(
                        f"Error in embedding process for short article {article_id}: {e}")
                    return (article_id, False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_article_1_6 = {executor.submit(process_embedding_task_1_6, article): article
                                         for article in articles_for_embedding}
                for i, future in enumerate(concurrent.futures.as_completed(future_to_article_1_6)):
                    try:
                        _, success = future.result()
                        if not success:
                            embedding_failure_1_6 += 1
                    except Exception as e:
                        logger.error(
                            f"Exception processing embedding task future (Step 1.6): {e}")
                        embedding_failure_1_6 += 1
                    total = len(articles_for_embedding)
                    processed = i + 1

            # Final database update for Step 1.6
            if pending_embeddings_1_6:
                final_count = update_embedding_database_1_6()

            embedding_elapsed_time_1_6 = time.time() - embedding_start_time_1_6
            logger.info(
                f"STEP 1.6 COMPLETE: Processed {embedding_success_1_6 + embedding_failure_1_6} short article embeddings in {embedding_elapsed_time_1_6:.2f} seconds")
            logger.info(
                f"Results (Step 1.6): {embedding_success_1_6} successful, {embedding_failure_1_6} failed")

        # --- STEP 1.7: Summarization and Embedding for Long Articles ---
        if run_step_1_7:
            logger.info(
                "========= STARTING STEP 1.7: SUMMARIZATION & EMBEDDING (Long Articles) =========")

            if not TRANSFORMERS_AVAILABLE:
                logger.error(
                    "Cannot run Step 1.7: 'transformers' package is not installed.")
                logger.error(
                    "To enable Step 1.7, please ensure transformers is installed in the Docker container:")
                logger.error("1. Add transformers to requirements.txt")
                logger.error(
                    "2. Rebuild the Docker image with: docker-compose build --no-cache article-transfer")
                logger.error(
                    "3. Or install inside running container: docker exec -it article-transfer pip install transformers torch sentencepiece")
            else:
                # Import LocalNLPClient here after we've confirmed transformers is available
                try:
                    from src.localnlp.localnlp_client import LocalNLPClient

                    # Initialize Local NLP Client
                    localnlp_client = LocalNLPClient(
                        model_name=local_nlp_model)

                    # Check if NLP client initialized correctly
                    if not localnlp_client.pipeline or not localnlp_client.tokenizer:
                        logger.error(
                            "LocalNLPClient failed to initialize. Skipping Step 1.7.")
                    else:
                        # Get articles needing summarization (use same max_char_limit as min_char_count)
                        articles_to_summarize = reader_db_embed.get_articles_needing_summarization(
                            min_char_count=max_char_limit)

                        if not articles_to_summarize:
                            logger.info(
                                "No long articles need summarization/embedding. Step 1.7 complete.")
                        else:
                            logger.info(
                                f"Found {len(articles_to_summarize)} long articles for summarization and embedding")

                            # Track stats for Step 1.7
                            summarization_success = 0
                            summarization_failure = 0
                            embedding_success_1_7 = 0
                            embedding_failure_1_7 = 0
                            pending_embeddings_1_7 = []
                            pending_embeddings_lock_1_7 = Lock()
                            # Define batch size and checkpoint interval for Step 1.7
                            embedding_checkpoint_interval = 60  # Same as in Step 1.6
                            embedding_batch_size = 20  # Same as in Step 1.6

                            summarization_start_time = time.time()
                            last_db_update_time_1_7 = summarization_start_time

                            def update_embedding_database_1_7():
                                nonlocal pending_embeddings_1_7, last_db_update_time_1_7, embedding_success_1_7
                                success_count = 0
                                with pending_embeddings_lock_1_7:
                                    if not pending_embeddings_1_7:
                                        return 0
                                    # Copy
                                    embeddings_to_insert = pending_embeddings_1_7[:]
                                    pending_embeddings_1_7 = []

                                if embeddings_to_insert:
                                    for article_id, embedding in embeddings_to_insert:
                                        result = reader_db_embed.insert_embedding(
                                            article_id, embedding)
                                        if result:
                                            success_count += 1
                                    embedding_success_1_7 += success_count  # Update total count
                                    last_db_update_time_1_7 = time.time()
                                    return success_count

                            def process_summarization_embedding_task(article_data: Dict[str, Any]) -> Tuple[int, bool, bool]:
                                nonlocal summarization_success, summarization_failure, last_db_update_time_1_7
                                article_id = article_data['id']
                                content = article_data['content']
                                summary_success_flag = False
                                embedding_success_flag = False

                                try:
                                    # Load spaCy model if not already loaded (can be moved outside for efficiency)
                                    nlp = spacy.load("en_core_web_lg")

                                    # Process with spaCy to get token count and break into sentences
                                    # Divide text into chunks based on sentences
                                    # Step 1: Process with spaCy to get sentence boundaries
                                    doc = nlp(content)

                                    # Step 2: Initialize chunking variables
                                    chunks = []
                                    current_chunk_tokens = 0
                                    current_chunk_sentences = []

                                    # Step 3: Build chunks based on sentences
                                    for sent in doc.sents:
                                        sent_token_count = len(sent)
                                        # If adding this sentence doesn't exceed chunk size limit
                                        if current_chunk_tokens + sent_token_count <= target_chunk_token_size:
                                            current_chunk_sentences.append(
                                                sent.text)
                                            current_chunk_tokens += sent_token_count
                                        else:
                                            # This chunk is full, save it and start a new one
                                            if current_chunk_sentences:  # Ensure chunk isn't empty
                                                chunks.append(
                                                    " ".join(current_chunk_sentences))
                                            # Start a new chunk with this sentence
                                            current_chunk_sentences = [
                                                sent.text]
                                            current_chunk_tokens = sent_token_count

                                    # Add the final chunk if it's not empty
                                    if current_chunk_sentences:
                                        chunks.append(
                                            " ".join(current_chunk_sentences))

                                    chunk_count = len(chunks)

                                    # If no chunks were created (unusual case), handle gracefully
                                    if not chunks:
                                        logger.warning(
                                            f"No chunks created for article {article_id}, content might be empty")
                                        summarization_failure += 1
                                        return (article_id, False, False)

                                    # Step 4: Summarize each chunk
                                    chunk_summaries = []

                                    for i, chunk in enumerate(chunks):
                                        try:
                                            chunk_summary = localnlp_client.summarize_text(
                                                chunk,
                                                max_summary_tokens=chunk_max_tokens,
                                                min_summary_tokens=chunk_min_tokens
                                            )

                                            if chunk_summary:
                                                chunk_summaries.append(
                                                    chunk_summary)
                                            else:
                                                logger.warning(
                                                    f"Failed to summarize chunk {i+1} for article {article_id}")
                                        except Exception as chunk_e:
                                            logger.error(
                                                f"Error summarizing chunk {i+1} for article {article_id}: {chunk_e}")
                                            # Continue with other chunks even if one fails

                                    # Step 5: Concatenate summaries
                                    if not chunk_summaries:
                                        logger.error(
                                            f"All chunk summarizations failed for article {article_id}")
                                        summarization_failure += 1
                                        return (article_id, False, False)

                                    # Join chunk summaries with spaces
                                    concatenated_summary = " ".join(
                                        chunk_summaries)

                                    # Successfully created summary
                                    summarization_success += 1
                                    summary_success_flag = True

                                    # Step 6: Generate embedding for the concatenated summary
                                    embedding = gemini_client.generate_embedding(
                                        concatenated_summary)

                                    if embedding is not None and isinstance(embedding, list) and len(embedding) > 0:
                                        with pending_embeddings_lock_1_7:
                                            pending_embeddings_1_7.append(
                                                (article_id, embedding))
                                            # Check DB update timing
                                            current_time = time.time()
                                            time_since_last_update = current_time - last_db_update_time_1_7
                                            should_update = (len(pending_embeddings_1_7) >= embedding_batch_size or
                                                             time_since_last_update >= embedding_checkpoint_interval)
                                        if should_update:
                                            update_embedding_database_1_7()
                                        embedding_success_flag = True
                                    else:
                                        logger.error(
                                            f"Failed to generate embedding for summarized article {article_id}")

                                except Exception as e:
                                    logger.error(
                                        f"Error in chunking/summarization/embedding task for article {article_id}: {e}")
                                    if summary_success_flag:  # Summarization worked, embedding failed
                                        embedding_success_flag = False
                                    else:  # Summarization failed
                                        summarization_failure += 1
                                        summary_success_flag = False

                                return (article_id, summary_success_flag, embedding_success_flag)

                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                future_to_article_1_7 = {executor.submit(process_summarization_embedding_task, article): article
                                                         for article in articles_to_summarize}

                                for i, future in enumerate(concurrent.futures.as_completed(future_to_article_1_7)):
                                    # Update failure count for embedding here if needed
                                    try:
                                        _, summary_ok, embedding_ok = future.result()
                                        if summary_ok and not embedding_ok:
                                            embedding_failure_1_7 += 1
                                        # Note: summarization counts updated inside worker
                                    except Exception as e:
                                        logger.error(
                                            f"Exception processing summarization task future (Step 1.7): {e}")
                                        embedding_failure_1_7 += 1  # Assume embedding failed if exception here

                            # Final database update for Step 1.7
                            if pending_embeddings_1_7:
                                final_count = update_embedding_database_1_7()

                            summarization_elapsed_time = time.time() - summarization_start_time
                            total_processed_1_7 = summarization_success + summarization_failure
                            logger.info(
                                f"STEP 1.7 COMPLETE: Processed {total_processed_1_7} long articles in {summarization_elapsed_time:.2f} seconds")
                            logger.info(
                                f"Summarization Results: {summarization_success} successful, {summarization_failure} failed")
                            # Note: embedding_success_1_7 is updated incrementally in DB update function
                            logger.info(
                                f"Embedding Results (Step 1.7): {embedding_success_1_7} successful, {embedding_failure_1_7} failed")
                except ImportError as e:
                    logger.error(f"Failed to import LocalNLPClient: {e}")
                    logger.error("Step 1.7 will be skipped.")
        else:
            logger.info("Skipping Step 1.7 as RUN_STEP_1_7 is not 'true'")

    except Exception as e:
        logger.error(
            f"Error during embedding generation steps (1.6 or 1.7): {e}", exc_info=True)
    finally:
        # Always close the embedding DB connection
        if 'reader_db_embed' in locals() and reader_db_embed and reader_db_embed.connection_pool:
            reader_db_embed.close()

    logger.info(
        f"========= STEP 1 COMPLETE (Final article insertion count from 1.4: {inserted_count}) =========")
    return inserted_count  # Return the count of articles inserted in step 1.4


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
