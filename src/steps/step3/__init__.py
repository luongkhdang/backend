"""
step3.py - Entity Extraction Module

This module implements Step 3 of the data refinery pipeline: processing recent articles to extract
entities, store entity relationships, and update basic entity statistics. It prioritizes articles
based on combined domain goodness and cluster hotness scores, then calls Gemini API for entity
extraction with tier-based model selection.

Exported functions:
- run(): Main function that orchestrates the entity extraction process
  - Returns Dict[str, Any]: Status report of entity extraction operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for articles and entities
- src/gemini/gemini_client.py: Used for entity extraction API calls
"""

import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Import database client
from src.database.reader_db_client import ReaderDBClient

# Import Gemini client for API calls
from src.gemini.gemini_client import GeminiClient

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


def run() -> Dict[str, Any]:
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
                "errors": 0,
                "runtime_seconds": time.time() - start_time
            }

        logger.info(f"Found {len(prioritized_articles)} articles to process.")

        # Step 3: Extract entities via API
        entity_results = _extract_entities(gemini_client, prioritized_articles)

        # Step 4: Store entity data and update article status
        store_results = _store_results(db_client, entity_results)

        # Prepare final status report
        status = {
            "success": True,
            "articles_found": len(prioritized_articles),
            "processed": store_results.get("processed", 0),
            "entity_links_created": store_results.get("links", 0),
            "errors": store_results.get("errors", 0),
            "runtime_seconds": time.time() - start_time
        }

        logger.info(f"Step 3 completed in {status['runtime_seconds']:.2f} seconds. "
                    f"Processed {status['processed']} articles, created {status['entity_links_created']} entity links.")

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
        # Try to fetch existing domain goodness scores
        try:
            # This function needs to be implemented in reader_db_client.py
            domain_scores = db_client.get_all_domain_goodness_scores()
            if domain_scores:
                logger.info(
                    f"Found {len(domain_scores)} domain goodness scores.")
                return domain_scores
        except Exception as e:
            logger.warning(f"Error fetching domain goodness scores: {e}")

        # Fallback: Use default scores
        logger.warning(
            "Using default domain goodness scores (0.5 for all domains).")
        return defaultdict(lambda: 0.5)

    except Exception as e:
        logger.error(f"Error in _get_domain_goodness_scores: {e}")
        return defaultdict(lambda: 0.5)


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


def _extract_entities(gemini_client: GeminiClient, articles: List[Dict[str, Any]]) -> Dict[int, Any]:
    """
    Extract entities from articles using Gemini API based on assigned tier.

    Args:
        gemini_client: Initialized GeminiClient instance
        articles: List of prioritized article dictionaries

    Returns:
        Dictionary mapping article IDs to their extracted entity data (or error info)
    """
    entity_results = {}
    batch_size = int(os.getenv("ENTITY_EXTRACTION_BATCH_SIZE", "10"))
    processed_count = 0

    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        logger.info(
            f"Processing article batch {i // batch_size + 1} / {(len(articles) + batch_size - 1) // batch_size}")

        # In a real scenario, you might use threading or asyncio here
        for article in batch:
            article_id = article.get('id')
            content = article.get('content')
            tier = article.get('processing_tier')
            model_to_use = TIER_MODEL_MAP.get(tier, FALLBACK_MODEL)

            if not article_id or not content:
                logger.warning(
                    f"Skipping article with missing ID or content: {article_id}")
                entity_results[article_id] = {"error": "Missing ID or content"}
                continue

            try:
                # Use GeminiClient to generate text (entities)
                extraction_result = gemini_client.generate_text_with_prompt(
                    article_content=content,
                    processing_tier=tier,
                    model_override=model_to_use  # Pass the selected model
                )

                if extraction_result:
                    # Attempt to parse the JSON string response
                    try:
                        parsed_entities = json.loads(extraction_result)
                        entity_results[article_id] = parsed_entities
                        logger.debug(
                            f"Successfully extracted entities for article {article_id}")
                    except json.JSONDecodeError as json_err:
                        logger.warning(
                            f"Failed to parse JSON from Gemini for article {article_id}: {json_err}")
                        entity_results[article_id] = {
                            "error": "Invalid JSON response", "raw_response": extraction_result}
                else:
                    logger.warning(
                        f"No entity extraction result from Gemini for article {article_id}")
                    entity_results[article_id] = {
                        "error": "No response from API"}

            except Exception as api_err:
                logger.error(
                    f"Error calling Gemini API for article {article_id}: {api_err}", exc_info=True)
                entity_results[article_id] = {
                    "error": f"API call failed: {api_err}"}

            processed_count += 1
            # Optional: Add a small delay between API calls if needed
            # time.sleep(0.1)

    logger.info(
        f"Finished entity extraction API calls for {processed_count} articles.")
    return entity_results


def _store_results(db_client: ReaderDBClient, entity_results: Dict[int, Any]) -> Dict[str, int]:
    """
    Store extracted entities, link them to articles, and update article status.

    Args:
        db_client: Database client instance
        entity_results: Dictionary mapping article IDs to extracted entity data

    Returns:
        Dictionary with summary counts
    """
    processed_count = 0
    entity_links_created = 0
    errors = 0

    if not entity_results:
        return {"processed": 0, "links": 0, "errors": 0}

    logger.info(
        f"Storing entity results for {len(entity_results)} articles...")

    for article_id, extracted_data in entity_results.items():
        try:
            # Check if this result contains an error
            if isinstance(extracted_data, dict) and 'error' in extracted_data:
                logger.warning(
                    f"Skipping article {article_id} due to extraction error: {extracted_data.get('error')}")
                errors += 1
                continue

            # Process extracted entities
            if isinstance(extracted_data, list):
                for entity_data in extracted_data:
                    # Extract entity info
                    entity_name = entity_data.get('entity_name')
                    entity_type = entity_data.get('entity_type')
                    # Default to 1 if no mentions list
                    mention_count = len(entity_data.get('mentions', [1]))

                    if not entity_name or not entity_type:
                        logger.warning(
                            f"Skipping entity with missing name or type: {entity_data}")
                        continue

                    # Find or create entity
                    try:
                        # This function needs to be implemented in reader_db_client.py
                        entity_id = db_client.find_or_create_entity(
                            name=entity_name, entity_type=entity_type)

                        if not entity_id:
                            logger.warning(
                                f"Failed to create/find entity {entity_name}")
                            continue

                        # Link article to entity with mention count
                        link_success = db_client.link_article_entity(
                            article_id=article_id,
                            entity_id=entity_id,
                            mention_count=mention_count,
                            is_influential_context=entity_data.get(
                                'is_influential_context', False)
                        )

                        if link_success:
                            entity_links_created += 1

                            # Increment global entity mentions count
                            # This function needs to be implemented in reader_db_client.py
                            db_client.increment_global_entity_mentions(
                                entity_id=entity_id, count=mention_count)

                    except Exception as e:
                        logger.error(
                            f"Error storing entity {entity_name} for article {article_id}: {e}")

                # Mark article as processed
                try:
                    # This function needs to be implemented in reader_db_client.py
                    db_client.mark_article_processed(article_id=article_id)
                    processed_count += 1
                except Exception as e:
                    logger.error(
                        f"Error marking article {article_id} as processed: {e}")
            else:
                logger.warning(
                    f"Unexpected data format for article {article_id}: {type(extracted_data)}")
                errors += 1

        except Exception as e:
            logger.error(
                f"Error processing results for article {article_id}: {e}")
            errors += 1

    logger.info(
        f"Storage complete. Processed {processed_count} articles, created {entity_links_created} entity links.")
    return {"processed": processed_count, "links": entity_links_created, "errors": errors}
