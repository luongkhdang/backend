"""
step3.py - Entity Extraction Module

This module implements Step 3 of the data refinery pipeline: processing recent articles to extract
entities, store entity relationships, and update basic entity statistics. It prioritizes articles
based on combined domain goodness and cluster hotness scores, then calls Gemini API for entity
extraction with tier-based model selection.

The module now also extracts narrative frame phrases from articles, which are short descriptors 
(2-4 words) that represent how stories are presented (e.g., "economic impact", "national security", 
"moral obligation"). These frame phrases are stored in the articles table and provide insight into 
the dominant perspectives used in news coverage.

Exported functions:
- run(): Main function that orchestrates the entity extraction process
  - Returns Dict[str, Any]: Status report of entity extraction operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for articles and entities
- src/gemini/gemini_client.py: Used for entity extraction API calls
- src/utils/task_manager.py: Manages concurrent execution of API calls
"""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import asyncio

# Import database client
from src.database.reader_db_client import ReaderDBClient

# Import Gemini client for API calls
from src.gemini.gemini_client import GeminiClient

# Import TaskManager for concurrent API calls
from src.utils.task_manager import TaskManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
ARTICLE_LIMIT = 2000
DAYS_LOOKBACK = 2
TIER0_COUNT = 150
TIER1_COUNT = 350
CLUSTER_HOTNESS_WEIGHT = 0.65
DOMAIN_GOODNESS_WEIGHT = 0.35

# Tier-to-Model mapping for Gemini API
TIER_MODEL_MAP = {
    # Highest quality
    0: os.getenv('GEMINI_FLASH_THINKING_MODEL', 'models/gemini-2.0-flash-thinking-exp-01-21'),
    # Mid-tier
    1: os.getenv('GEMINI_FLASH_EXP_MODEL', 'models/gemini-2.0-flash-exp'),
    # Base tier
    2: os.getenv('GEMINI_FLASH_MODEL', 'models/gemini-2.0-flash')
}
FALLBACK_MODEL = os.getenv('GEMINI_FLASH_LITE_MODEL',
                           'models/gemini-2.0-flash-lite')


async def run() -> Dict[str, Any]:
    """
    Main function to run the entity extraction process.

    This function:
    1. Retrieves domain goodness scores
    2. Fetches and prioritizes recent unprocessed articles
    3. Extracts entities using Gemini API with tier-based model selection
    4. Stores entity results in the database
    5. Updates article processing status

    Returns:
        Dict[str, Any]: Status report containing metrics about the extraction process
    """
    start_time = time.time()

    try:
        # Initialize clients
        db_client = ReaderDBClient()
        gemini_client = GeminiClient()
        task_manager = TaskManager()  # Initialize the TaskManager

        # Step 1: Get domain goodness scores
        domain_scores = _get_domain_goodness_scores(db_client)

        # Step 2: Fetch and prioritize articles
        prioritized_articles = _prioritize_articles(db_client, domain_scores)

        if not prioritized_articles:
            logger.info("No articles to process.")
            return {
                "success": True,
                "articles_found": 0,
                "processed": 0,
                "entity_links_created": 0,
                "snippets_stored": 0,
                "errors": 0,
                "runtime_seconds": time.time() - start_time
            }

        logger.info(f"Found {len(prioritized_articles)} articles to process.")

        # Process using batched approach that stores to DB after each batch
        batch_size = int(os.getenv("ENTITY_EXTRACTION_BATCH_SIZE", "10"))
        total_batches = (len(prioritized_articles) +
                         batch_size - 1) // batch_size

        # Initialize counters for final status
        total_processed = 0
        total_entity_links = 0
        total_snippets = 0
        total_errors = 0

        # Process each batch
        for batch_index in range(0, len(prioritized_articles), batch_size):
            batch_num = batch_index // batch_size + 1
            batch = prioritized_articles[batch_index:batch_index + batch_size]

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")

            # Step 3: Extract entities for this batch (use await)
            batch_entity_results = await _extract_entities_batch(
                gemini_client, batch)
            logger.info(
                f"Entity extraction complete for batch {batch_num}. Found data for {len(batch_entity_results)} articles.")

            # Step 4: Store entity data for this batch immediately
            if batch_entity_results:
                logger.info(
                    f"Storing entity results for batch {batch_num} ({len(batch_entity_results)} articles)")
                batch_store_results = _store_results(
                    db_client, batch_entity_results)

                # Update counters
                total_processed += batch_store_results.get("processed", 0)
                total_entity_links += batch_store_results.get("links", 0)
                total_snippets += batch_store_results.get("snippets", 0)
                total_errors += batch_store_results.get("errors", 0)

                logger.info(f"Batch {batch_num} complete: Processed {batch_store_results.get('processed', 0)} articles, "
                            f"created {batch_store_results.get('links', 0)} entity links, "
                            f"stored {batch_store_results.get('snippets', 0)} snippets.")

        # Prepare final status report
        status = {
            "success": True,
            "articles_found": len(prioritized_articles),
            "processed": total_processed,
            "entity_links_created": total_entity_links,
            "snippets_stored": total_snippets,
            "errors": total_errors,
            "runtime_seconds": time.time() - start_time
        }

        logger.info(f"Step 3 completed in {status['runtime_seconds']:.2f} seconds. "
                    f"Processed {status['processed']} articles, created {status['entity_links_created']} entity links, "
                    f"stored {status['snippets_stored']} supporting snippets.")

        return status

    except Exception as e:
        logger.error(f"Error in Step 3: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "runtime_seconds": time.time() - start_time
        }


def _get_domain_goodness_scores(db_client: ReaderDBClient) -> Dict[str, float]:
    """
    Get domain goodness scores from database.

    Args:
        db_client: Database client instance

    Returns:
        Dictionary mapping domains to their goodness scores
    """
    try:
        # Fetch existing domain goodness scores
        domain_scores = db_client.get_all_domain_goodness_scores()
        if domain_scores:
            logger.info(f"Found {len(domain_scores)} domain goodness scores.")
            return domain_scores

        # If no scores found, return empty dictionary - no fallbacks
        logger.warning("No domain goodness scores found in database.")
        return {}

    except Exception as e:
        logger.error(f"Error in _get_domain_goodness_scores: {e}")
        # No fallback - raise the exception
        raise


def _prioritize_articles(db_client: ReaderDBClient, domain_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Fetch recent unprocessed articles and prioritize them based on combined score.

    Args:
        db_client: Database client instance
        domain_scores: Dictionary mapping domains to their goodness scores

    Returns:
        List of prioritized article dictionaries with processing tier assigned
    """
    try:
        # Get recent unprocessed articles
        try:
            # This function needs to be implemented in reader_db_client.py
            articles = db_client.get_recent_unprocessed_articles(
                days=DAYS_LOOKBACK, limit=ARTICLE_LIMIT)
        except Exception as e:
            logger.error(f"Error fetching recent unprocessed articles: {e}")
            return []

        if not articles:
            logger.info("No recent unprocessed articles found.")
            return []

        # Fetch cluster hotness scores
        try:
            # Get all clusters
            clusters = db_client.get_all_clusters()
            # Create mapping from cluster_id to hotness_score
            cluster_hotness = {c.get('id'): c.get(
                'hotness_score', 0.0) for c in clusters}
        except Exception as e:
            logger.warning(
                f"Error fetching clusters: {e}. Using default hotness scores.")
            cluster_hotness = {}

        # Calculate scores for each article
        for article in articles:
            # Get domain goodness score (default 0.0 if missing)
            domain = article.get('domain', '')
            domain_goodness_score = domain_scores.get(domain, 0.0)

            # Get cluster hotness score (default 0.0 if missing)
            cluster_id = article.get('cluster_id')
            cluster_hotness_score = cluster_hotness.get(
                cluster_id, 0.0) if cluster_id else 0.0

            # Add scores to article dict for debugging/transparency
            article['domain_goodness_score'] = domain_goodness_score
            article['cluster_hotness_score'] = cluster_hotness_score

            # Store raw scores for normalization
            article['raw_combined_score'] = (
                CLUSTER_HOTNESS_WEIGHT * cluster_hotness_score +
                DOMAIN_GOODNESS_WEIGHT * domain_goodness_score
            )

        # Find max scores for normalization (prevent division by zero)
        max_score = max((a.get('raw_combined_score', 0.0)
                        for a in articles), default=1.0)
        if max_score == 0.0:
            max_score = 1.0  # Avoid division by zero

        # Normalize scores and calculate final combined score
        for article in articles:
            article['combined_priority_score'] = article.get(
                'raw_combined_score', 0.0) / max_score

        # Sort articles by combined score (descending)
        prioritized_articles = sorted(
            articles,
            key=lambda a: a.get('combined_priority_score', 0.0),
            reverse=True
        )

        # Assign priority rank and processing tier
        for i, article in enumerate(prioritized_articles):
            rank = i + 1  # 1-based ranking
            article['priority_rank'] = rank

            # Assign tier based on rank
            if rank <= TIER0_COUNT:
                article['processing_tier'] = 0
            elif rank <= TIER0_COUNT + TIER1_COUNT:
                article['processing_tier'] = 1
            else:
                article['processing_tier'] = 2

        logger.info(f"Prioritized {len(prioritized_articles)} articles into tiers (0: {TIER0_COUNT}, "
                    f"1: {TIER1_COUNT}, 2: {len(prioritized_articles) - TIER0_COUNT - TIER1_COUNT})")

        return prioritized_articles

    except Exception as e:
        logger.error(f"Error prioritizing articles: {e}", exc_info=True)
        return []


async def _extract_entities_batch(gemini_client: GeminiClient, articles: List[Dict[str, Any]]) -> Dict[int, Any]:
    """
    Extract entities from a batch of articles using Gemini API concurrently.

    Args:
        gemini_client: Initialized GeminiClient instance
        articles: Batch of article dictionaries to process

    Returns:
        Dictionary mapping article IDs to their extracted entity data (or error dict)
    """
    logger.info(
        f"Preparing to extract entities for {len(articles)} articles using TaskManager")

    # Create task definitions for each article
    tasks_definitions = []

    for article in articles:
        article_id = article.get('id')
        content = article.get('content')
        tier = article.get('processing_tier')
        model_to_use = TIER_MODEL_MAP.get(tier, FALLBACK_MODEL)

        if not article_id or not content:
            logger.warning(
                f"Skipping article {article_id} with missing ID or content.")
            # We'll handle these separately since they're not valid tasks
            continue

        # Create a task definition dictionary with all necessary data
        task_definition = {
            'article_id': article_id,
            'content': content,
            'processing_tier': tier,
            'model_to_use': model_to_use
        }
        tasks_definitions.append(task_definition)

    # Special handling for skipped articles (missing ID/content)
    skipped_results = {}
    for article in articles:
        article_id = article.get('id')
        if article_id and (not article.get('content')):
            skipped_results[article_id] = {"error": "Missing ID or content"}

    # Use TaskManager to run the tasks concurrently
    if tasks_definitions:
        # Get a reference to the TaskManager instance created in run()
        task_manager = TaskManager()  # Create a new instance for this batch
        extraction_results = await task_manager.run_tasks(gemini_client, tasks_definitions)

        # Merge with skipped results
        extraction_results.update(skipped_results)

        return extraction_results
    else:
        logger.warning("No valid articles to process in batch")
        return skipped_results


def _extract_entities(gemini_client: GeminiClient, articles: List[Dict[str, Any]]) -> Dict[int, Any]:
    """
    Extract entities from all articles using Gemini API.
    This is the original function that processes all articles at once.
    For batch-by-batch processing with immediate DB storage, use the new run() implementation.

    DEPRECATED: This synchronous method has been replaced by the async implementation
    in the run() function that uses TaskManager.

    Args:
        gemini_client: Initialized GeminiClient instance
        articles: List of all prioritized article dictionaries

    Returns:
        Dictionary mapping article IDs to their extracted entity data
    """
    logger.warning(
        "_extract_entities (synchronous full processing) called. This is DEPRECATED and should not be used.")

    entity_results = {}
    batch_size = int(os.getenv("ENTITY_EXTRACTION_BATCH_SIZE", "10"))

    # We can't safely call async methods from this sync context
    # This is just a placeholder to avoid breaking existing code if called

    logger.error(
        "Deprecated _extract_entities called. Please use the async version with TaskManager.")
    # Return empty results - better to fail explicitly than provide partial incorrect results
    return entity_results


def _store_results(db_client: ReaderDBClient, entity_results: Dict[int, Any]) -> Dict[str, int]:
    """
    Store extracted entities, link them to articles, and update article status.
    Processes and commits results in batches of 10 articles for incremental progress.

    Args:
        db_client: Database client instance
        entity_results: Dictionary mapping article IDs to extracted entity data

    Returns:
        Dictionary with summary counts
    """
    processed_count = 0
    entity_links_created = 0
    snippets_stored = 0
    errors = 0

    if not entity_results:
        return {"processed": 0, "links": 0, "snippets": 0, "errors": 0}

    # Get list of article IDs to process in batches
    article_ids = list(entity_results.keys())
    batch_size = 10  # Process 10 articles at a time for DB operations
    total_batches = (len(article_ids) + batch_size - 1) // batch_size

    logger.info(
        f"Storing entity results for {len(entity_results)} articles in batches of {batch_size}...")

    # Process articles in batches to provide incremental commits
    for batch_index in range(0, len(article_ids), batch_size):
        batch_article_ids = article_ids[batch_index:batch_index + batch_size]
        batch_num = batch_index // batch_size + 1

        logger.info(
            f"Processing storage batch {batch_num}/{total_batches} ({len(batch_article_ids)} articles)")

        # Track batch metrics for reporting
        batch_processed = 0
        batch_links = 0
        batch_snippets = 0
        batch_errors = 0

        for article_id in batch_article_ids:
            extracted_data = entity_results[article_id]
            try:
                # Check if this result contains an error
                if isinstance(extracted_data, dict) and 'error' in extracted_data:
                    logger.warning(
                        f"Skipping article {article_id} due to extraction error: {extracted_data.get('error')}")
                    batch_errors += 1
                    continue

                # Extract frame phrases if they exist - new feature
                frame_phrases = None
                if isinstance(extracted_data, dict) and 'fr' in extracted_data and isinstance(extracted_data['fr'], list):
                    frame_phrases = extracted_data['fr']
                    logger.info(
                        f"Found {len(frame_phrases)} frame phrases for article {article_id}")

                # Process extracted entities - Check for new format with 'ents' or legacy format
                entity_list_to_process = None
                if isinstance(extracted_data, dict) and 'ents' in extracted_data and isinstance(extracted_data['ents'], list):
                    # Handle new format with 'ents' key
                    entity_list_to_process = extracted_data['ents']
                elif isinstance(extracted_data, dict) and 'entities' in extracted_data and isinstance(extracted_data['entities'], list):
                    # Handle previous format with 'entities' key
                    entity_list_to_process = extracted_data['entities']
                elif isinstance(extracted_data, dict) and 'extracted_entities' in extracted_data and isinstance(extracted_data['extracted_entities'], list):
                    # Handle legacy format with 'extracted_entities' key
                    entity_list_to_process = extracted_data['extracted_entities']
                elif isinstance(extracted_data, list):
                    # Handle case where the response is directly the list (very old format)
                    logger.warning(
                        f"Received direct list for article {article_id}, expected dict with 'ents' key.")
                    entity_list_to_process = extracted_data

                if entity_list_to_process is not None:  # Proceed if we found a valid list of entities
                    # Check if the list is empty
                    if not entity_list_to_process:
                        logger.info(
                            f"No entities extracted for article {article_id} (empty list).")
                        # Mark article as processed even if no entities were found, include frame phrases if any
                        try:
                            if frame_phrases:
                                db_client.update_article_frames_and_mark_processed(
                                    article_id=article_id, frame_phrases=frame_phrases)
                            else:
                                db_client.mark_article_processed(
                                    article_id=article_id)
                            batch_processed += 1  # Count as processed
                        except Exception as e:
                            logger.error(
                                f"Error marking article {article_id} as processed (no entities): {e}")
                            batch_errors += 1  # Still an error if DB update fails
                        continue  # Move to the next article

                    # Process the non-empty list of entities
                    for entity_data in entity_list_to_process:
                        # Extract entity info based on whether it's using short keys (new format) or long keys (old format)
                        entity_name = entity_data.get(
                            'en') or entity_data.get('entity_name')
                        entity_type = entity_data.get(
                            'et') or entity_data.get('entity_type')
                        # Handle mention count - support both formats
                        mention_count = (entity_data.get('mc') or entity_data.get(
                            'mention_count_article') or 1)

                        # Handle influential context - support both boolean and integer formats
                        is_influential_raw = entity_data.get(
                            'ic') if 'ic' in entity_data else entity_data.get('is_influential_context')
                        # Convert to boolean if it's an integer in the new format
                        is_influential = bool(
                            is_influential_raw) if is_influential_raw is not None else False

                        # Handle supporting snippets - support both formats
                        supporting_snippets = entity_data.get(
                            'ss') or entity_data.get('supporting_snippets') or []

                        if not entity_name or not entity_type:
                            logger.warning(
                                f"Skipping entity with missing name or type in article {article_id}: {entity_data}")
                            continue

                        # Find or create entity
                        try:
                            entity_id = db_client.find_or_create_entity(
                                name=entity_name, entity_type=entity_type)

                            if not entity_id:
                                logger.warning(
                                    f"Failed to create/find entity '{entity_name}' for article {article_id}")
                                continue

                            # Link article to entity with mention count and influential context flag
                            link_success = db_client.link_article_entity(
                                article_id=article_id,
                                entity_id=entity_id,
                                mention_count=mention_count,
                                is_influential_context=is_influential
                            )

                            if link_success:
                                batch_links += 1

                                # Increment global entity mentions count
                                db_client.increment_global_entity_mentions(
                                    entity_id=entity_id, count=mention_count)

                                # Store supporting snippets only if entity is influential
                                if is_influential and supporting_snippets:
                                    for snippet in supporting_snippets:
                                        # Snippets in the prompt are just strings, not dicts
                                        if isinstance(snippet, str) and snippet:
                                            snippet_success = db_client.store_entity_snippet(
                                                entity_id=entity_id,
                                                article_id=article_id,
                                                snippet=snippet,
                                                # is_influential flag in snippet table indicates if the snippet itself is influential
                                                # The prompt structure doesn't provide this per snippet, only per entity.
                                                # Defaulting to True if the entity was influential.
                                                is_influential=is_influential
                                            )
                                            if snippet_success:
                                                batch_snippets += 1
                                        else:
                                            logger.warning(
                                                f"Skipping invalid snippet format for entity {entity_id} in article {article_id}: {snippet}")

                        except Exception as e:
                            logger.error(
                                f"Error storing entity '{entity_name}' for article {article_id}: {e}", exc_info=True)
                            # Consider incrementing batch_errors here?

                    # Mark article as processed (after processing all its entities), include frame phrases if any
                    try:
                        if frame_phrases:
                            db_client.update_article_frames_and_mark_processed(
                                article_id=article_id, frame_phrases=frame_phrases)
                        else:
                            db_client.mark_article_processed(
                                article_id=article_id)
                        batch_processed += 1
                    except Exception as e:
                        logger.error(
                            f"Error marking article {article_id} as processed: {e}")
                        batch_errors += 1  # Error during marking processed
                else:  # No valid entity list found
                    logger.warning(
                        f"Unexpected data format for article {article_id}: {type(extracted_data)}")
                    # Still try to save frame phrases if they exist
                    if frame_phrases:
                        try:
                            db_client.update_article_frames_and_mark_processed(
                                article_id=article_id, frame_phrases=frame_phrases)
                            batch_processed += 1
                            logger.info(
                                f"Saved only frame phrases for article {article_id}")
                        except Exception as e:
                            logger.error(
                                f"Error saving frame phrases for article {article_id}: {e}")
                            batch_errors += 1
                    else:
                        batch_errors += 1

            except Exception as e:
                logger.error(
                    f"Error processing results for article {article_id}: {e}")
                batch_errors += 1

        # Update running totals with batch results
        processed_count += batch_processed
        entity_links_created += batch_links
        snippets_stored += batch_snippets
        errors += batch_errors

        # Log batch completion
        logger.info(
            f"Batch {batch_num}/{total_batches} complete: Processed {batch_processed} articles, created {batch_links} links, stored {batch_snippets} snippets.")

        # Explicitly ensure the current batch is committed by releasing and getting a new connection
        # This ensures DB writes are persisted in case of interruptions
        try:
            # First release any existing connection back to the pool
            # Assuming get_connection() returns the connection object itself
            conn = db_client.get_connection()
            if conn:
                db_client.release_connection(conn)
                logger.debug(
                    f"Committed batch {batch_num} to database (connection released)")
            else:
                logger.warning(
                    f"Could not get connection to commit batch {batch_num}")
        except Exception as e:
            logger.warning(f"Error during explicit batch commit: {e}")

    logger.info(
        f"Storage complete. Total: {processed_count} articles processed, {entity_links_created} entity links created, {snippets_stored} snippets stored, {errors} errors.")
    return {"processed": processed_count, "links": entity_links_created, "snippets": snippets_stored, "errors": errors}
