#!/usr/bin/env python3
"""
step3/__init__.py - Entity Extraction Module

Entity extraction process that prioritizes articles, extracts entities using Gemini API,
and stores results in the database. The process uses tier-based model selection.

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
import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Import DB and API clients
from src.database.reader_db_client import ReaderDBClient
from src.gemini.gemini_client import GeminiClient
from src.utils.task_manager import TaskManager

# Configure logging
logger = logging.getLogger(__name__)

# Constants for configuration
DAYS_LOOKBACK = int(os.getenv("ENTITY_LOOKBACK_DAYS", "3"))
ARTICLE_LIMIT = int(os.getenv("ENTITY_MAX_PRIORITY_ARTICLES", "2000"))
BATCH_SIZE = int(os.getenv("ENTITY_EXTRACTION_BATCH_SIZE", "10"))

# Tier-specific models and fallbacks (REMOVED HARDCODED VALUES)
# TIER_MODELS = { ... }
# FALLBACK_MODEL = '...'

# Error handling constants
MAX_CONSECUTIVE_FAILURES = 3
FAILURE_COOLDOWN_SECONDS = 30
INTER_BATCH_DELAY_SECONDS = 10  # Delay between batches

# --- Helper Function for Time Formatting ---


def format_seconds(seconds: float) -> str:
    """Formats seconds into a human-readable string (e.g., 1m 30s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds_rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {seconds_rem}s"
    hours, minutes_rem = divmod(minutes, 60)
    return f"{hours}h {minutes_rem}m {seconds_rem}s"
# --- End Helper Function ---


async def run() -> Dict[str, Any]:
    """
    Main function to run the entity extraction process.

    This function:
    1. Retrieves domain goodness scores
    2. Fetches and prioritizes unprocessed articles
    3. Divides articles into processing tiers
    4. Creates balanced batches with articles from each tier
    5. Extracts entities using Gemini API with tier-based model selection
    6. Stores entity results in the database

    Returns:
        Dict[str, Any]: Status report containing metrics about the extraction process
    """
    start_time = time.time()

    try:
        # Initialize clients
        db_client = ReaderDBClient()
        gemini_client = GeminiClient()
        # Initialize a single TaskManager instance for the entire process
        task_manager = TaskManager()

        # --- Load Model IDs from Environment --- #
        # Tier 0 (Highest): Flash Exp primary, Flash fallback
        tier0_primary_model = os.getenv(
            "GEMINI_FLASH_EXP_MODEL_ID", "gemini-2.0-flash-exp")
        tier0_fallback_model = os.getenv(
            "GEMINI_FLASH_MODEL_ID", "gemini-2.0-flash")
        # Tier 1 (Medium): Flash primary, Lite fallback
        tier1_primary_model = os.getenv(
            "GEMINI_FLASH_MODEL_ID", "gemini-2.0-flash")
        tier1_fallback_model = os.getenv(
            "GEMINI_FLASH_LITE_MODEL_ID", "gemini-2.0-flash-lite")
        # Tier 2 (Lowest): Flash primary, Lite fallback
        tier2_primary_model = os.getenv(
            "GEMINI_FLASH_MODEL_ID", "gemini-2.0-flash")
        tier2_fallback_model = os.getenv(
            "GEMINI_FLASH_LITE_MODEL_ID", "gemini-2.0-flash-lite")

        # Log the models being used for each tier
        logger.info(
            f"Tier 0 Models: Primary={tier0_primary_model}, Fallback={tier0_fallback_model}")
        logger.info(
            f"Tier 1 Models: Primary={tier1_primary_model}, Fallback={tier1_fallback_model}")
        logger.info(
            f"Tier 2 Models: Primary={tier2_primary_model}, Fallback={tier2_fallback_model}")
        # -------------------------------------- #

        # For circuit breaker pattern
        consecutive_failures = 0

        # Step 1: Get domain goodness scores
        domain_scores = _get_domain_goodness_scores(db_client)

        # Step 2: Fetch and prioritize articles - already assigns tiers
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

        # Step 3: Divide articles into three tiers based on assigned processing_tier
        tier0_articles = [
            a for a in prioritized_articles if a.get('processing_tier') == 0]
        tier1_articles = [
            a for a in prioritized_articles if a.get('processing_tier') == 1]
        tier2_articles = [
            a for a in prioritized_articles if a.get('processing_tier') == 2]

        logger.info(f"Articles by tier: {len(tier0_articles)} in tier 0, "
                    f"{len(tier1_articles)} in tier 1, {len(tier2_articles)} in tier 2")

        # Calculate initial total articles and estimate total batches
        total_articles_to_process = len(prioritized_articles)
        # Estimate total batches based on average batch size (adjust if needed)
        estimated_total_batches = (
            total_articles_to_process + BATCH_SIZE - 1) // BATCH_SIZE if BATCH_SIZE > 0 else 1

        # Initialize counters for final status
        total_processed = 0
        total_entity_links = 0
        total_snippets = 0
        total_errors = 0
        total_events = 0
        total_policies = 0
        total_relationships = 0
        batch_index = 0  # For batch numbering

        # Add heartbeat logging and timing variables
        last_heartbeat = time.time()
        heartbeat_interval = 60  # Log a heartbeat every minute
        total_batches_processed = 0
        total_time_processed = 0.0

        # Process articles in balanced batches until all tiers are empty
        while tier0_articles or tier1_articles or tier2_articles:
            batch_start_time = time.time()  # Record batch start time
            batch_index += 1

            # Compose a balanced batch with the defined ratio (4/5/1)
            current_batch = []
            target_batch_size = 10

            # Initialize counters for original and rebalanced tier counts
            original_tier_counts = {0: 0, 1: 0, 2: 0}
            rebalanced_tier_counts = {0: 0, 1: 0, 2: 0}

            # Add up to 4 articles from tier 0
            tier0_to_add = min(4, len(tier0_articles))
            if tier0_to_add > 0:
                current_batch.extend(tier0_articles[:tier0_to_add])
                tier0_articles = tier0_articles[tier0_to_add:]
                original_tier_counts[0] = tier0_to_add

            # Add up to 5 articles from tier 1
            tier1_to_add = min(5, len(tier1_articles))
            if tier1_to_add > 0:
                current_batch.extend(tier1_articles[:tier1_to_add])
                tier1_articles = tier1_articles[tier1_to_add:]
                original_tier_counts[1] = tier1_to_add

            # Add up to 1 article from tier 2
            tier2_to_add = min(1, len(tier2_articles))
            if tier2_to_add > 0:
                current_batch.extend(tier2_articles[:tier2_to_add])
                tier2_articles = tier2_articles[tier2_to_add:]
                original_tier_counts[2] = tier2_to_add

            # Smart rebalancing logic
            remaining_slots = target_batch_size - len(current_batch)
            if remaining_slots > 0:
                # First, try to fill with tier 0 (highest priority)
                if len(tier0_articles) > 0:
                    additional_tier0 = min(
                        remaining_slots, len(tier0_articles))
                    current_batch.extend(tier0_articles[:additional_tier0])
                    tier0_articles = tier0_articles[additional_tier0:]
                    remaining_slots -= additional_tier0
                    rebalanced_tier_counts[0] = additional_tier0

                # Then, try to fill with tier 1 (medium priority)
                if remaining_slots > 0 and len(tier1_articles) > 0:
                    additional_tier1 = min(
                        remaining_slots, len(tier1_articles))
                    current_batch.extend(tier1_articles[:additional_tier1])
                    tier1_articles = tier1_articles[additional_tier1:]
                    remaining_slots -= additional_tier1
                    rebalanced_tier_counts[1] = additional_tier1

                # Finally, try to fill with tier 2 (lowest priority)
                if remaining_slots > 0 and len(tier2_articles) > 0:
                    additional_tier2 = min(
                        remaining_slots, len(tier2_articles))
                    current_batch.extend(tier2_articles[:additional_tier2])
                    tier2_articles = tier2_articles[additional_tier2:]
                    rebalanced_tier_counts[2] = additional_tier2

            # If no articles were added to the batch, we're done
            if not current_batch:
                break

            # Assign specific models based on tier
            for article in current_batch:
                # Default to tier 2 if missing
                tier = article.get('processing_tier', 2)

                # Assign primary and fallback models based on tier using loaded env vars
                if tier == 0:
                    article['model_to_use'] = tier0_primary_model
                    article['fallback_model'] = tier0_fallback_model
                elif tier == 1:
                    article['model_to_use'] = tier1_primary_model
                    article['fallback_model'] = tier1_fallback_model
                else:  # Tier 2 or default
                    article['model_to_use'] = tier2_primary_model
                    article['fallback_model'] = tier2_fallback_model

            # Log detailed information about tier distribution
            was_rebalanced = any(
                count > 0 for count in rebalanced_tier_counts.values())

            if was_rebalanced:
                logger.info(
                    f"Processing batch {batch_index} with {len(current_batch)} articles | "
                    f"Original distribution: Tier 0: {original_tier_counts[0]}, "
                    f"Tier 1: {original_tier_counts[1]}, Tier 2: {original_tier_counts[2]} | "
                    f"Added during rebalancing: Tier 0: {rebalanced_tier_counts[0]}, "
                    f"Tier 1: {rebalanced_tier_counts[1]}, Tier 2: {rebalanced_tier_counts[2]}"
                )
            else:
                logger.info(
                    f"Processing batch {batch_index} with {len(current_batch)} articles | "
                    f"Standard distribution: Tier 0: {original_tier_counts[0]}, "
                    f"Tier 1: {original_tier_counts[1]}, Tier 2: {original_tier_counts[2]}"
                )

            # Heartbeat logging (modified)
            current_time = time.time()
            if current_time - last_heartbeat >= heartbeat_interval:
                processed_count = total_processed  # Use current total processed count
                remaining_articles = len(
                    tier0_articles) + len(tier1_articles) + len(tier2_articles)
                progress = ((processed_count) / total_articles_to_process) * \
                    100 if total_articles_to_process > 0 else 0

                etc_str = "N/A"
                avg_time_str = "N/A"
                if total_batches_processed > 0:
                    average_time_per_batch = total_time_processed / total_batches_processed
                    avg_time_str = f"{average_time_per_batch:.1f}s/batch"
                    # Estimate remaining batches based on remaining articles and avg batch size seen so far
                    avg_batch_size = total_processed / \
                        total_batches_processed if total_batches_processed > 0 else BATCH_SIZE
                    estimated_remaining_batches = (
                        remaining_articles / avg_batch_size) if avg_batch_size > 0 else 0

                    estimated_seconds_remaining = average_time_per_batch * estimated_remaining_batches
                    if estimated_seconds_remaining >= 0:
                        etc_datetime = datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                        etc_str = f"ETC: {etc_datetime.strftime('%H:%M:%S')} ({format_seconds(estimated_seconds_remaining)} remaining)"
                    else:
                        etc_str = "ETC: Calculating..."

                logger.info(
                    f"Heartbeat: Batch {batch_index}/{estimated_total_batches} | Progress: {progress:.1f}% "
                    f"({processed_count}/{total_articles_to_process} articles) | Avg: {avg_time_str} | {etc_str}"
                )
                last_heartbeat = current_time

            # Circuit breaker pattern
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    f"Circuit breaker triggered after {consecutive_failures} failures. Cooling down for {FAILURE_COOLDOWN_SECONDS} seconds")
                await asyncio.sleep(FAILURE_COOLDOWN_SECONDS)
                consecutive_failures = 0  # Reset counter after cooldown

            try:
                # Step 4: Extract entities for this batch (use await)
                batch_entity_results = await _extract_entities_batch(
                    gemini_client, current_batch, task_manager)  # Pass the task_manager instance

                if batch_entity_results:
                    # Reset failure counter on success
                    consecutive_failures = 0
                    logger.info(
                        f"Entity extraction complete for batch {batch_index}. Found data for {len(batch_entity_results)} articles.")

                    # Step 5: Store entity data for this batch immediately
                    logger.info(
                        f"Storing entity results for batch {batch_index} ({len(batch_entity_results)} articles)")
                    batch_store_results = _store_results(
                        db_client, batch_entity_results)

                    # Update counters
                    total_processed += batch_store_results.get("processed", 0)
                    total_entity_links += batch_store_results.get("links", 0)
                    total_snippets += batch_store_results.get("snippets", 0)
                    total_errors += batch_store_results.get("errors", 0)
                    # Add new relational data counters
                    total_events = total_events + \
                        batch_store_results.get("events", 0)
                    total_policies = total_policies + \
                        batch_store_results.get("policies", 0)
                    total_relationships = total_relationships + \
                        batch_store_results.get("relationships", 0)

                    logger.info(f"Batch {batch_index} complete: Processed {batch_store_results.get('processed', 0)} articles, "
                                f"created {batch_store_results.get('links', 0)} entity links, "
                                f"stored {batch_store_results.get('snippets', 0)} snippets, "
                                f"created {batch_store_results.get('events', 0)} event links, "
                                f"created {batch_store_results.get('policies', 0)} policy links, "
                                f"recorded {batch_store_results.get('relationships', 0)} relationships.")
                else:
                    # Increment failure counter on empty results
                    consecutive_failures += 1
                    logger.warning(
                        f"Batch {batch_index} returned no results. Consecutive failures: {consecutive_failures}")
            except Exception as e:
                # Increment failure counter on exception
                consecutive_failures += 1
                logger.error(
                    f"Error processing batch {batch_index}: {e}", exc_info=True)
                logger.warning(f"Consecutive failures: {consecutive_failures}")

            # Track batch completion time and update timing totals
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            total_batches_processed += 1
            total_time_processed += batch_duration

            # Add a delay between batches to allow rate limits to reset
            # Log rate limiter status for all models
            logger.info(f"Rate limiter status before cooling period:")
            # Iterate over models configured in the rate limiter
            rate_limited_models = gemini_client.rate_limiter.model_rpm_limits.keys() if hasattr(
                gemini_client, 'rate_limiter') and gemini_client.rate_limiter else []
            for model_name in rate_limited_models:
                if hasattr(gemini_client, 'rate_limiter') and gemini_client.rate_limiter:
                    current_rpm = gemini_client.rate_limiter.get_current_rpm(
                        model_name)
                    wait_time = gemini_client.rate_limiter.get_wait_time(
                        model_name)
                    logger.info(
                        f"  Model: {model_name}, Current RPM: {current_rpm}, Wait time: {wait_time:.2f}s")

            # Add cooling period between batches
            logger.info(
                f"Adding cooling period of {INTER_BATCH_DELAY_SECONDS} seconds between batches")
            await asyncio.sleep(INTER_BATCH_DELAY_SECONDS)

        # Prepare final status report
        status = {
            "success": True,
            "articles_found": len(prioritized_articles),
            "processed": total_processed,
            "entity_links_created": total_entity_links,
            "snippets_stored": total_snippets,
            "errors": total_errors,
            "events": total_events,
            "policies": total_policies,
            "relationships": total_relationships,
            "runtime_seconds": time.time() - start_time
        }

        logger.info(f"Step 3 completed in {status['runtime_seconds']:.2f} seconds. "
                    f"Processed {status['processed']} articles, created {status['entity_links_created']} entity links, "
                    f"stored {status['snippets_stored']} supporting snippets, "
                    f"created {status['events']} event links, "
                    f"created {status['policies']} policy links, "
                    f"recorded {status['relationships']} relationships.")

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
            logger.debug(f"Found {len(domain_scores)} domain goodness scores.")
            return domain_scores

        # If no scores found, return empty dictionary - no fallbacks
        logger.warning("No domain goodness scores found in database.")
        return {}

    except Exception as e:
        logger.error(f"Error in _get_domain_goodness_scores: {e}")
        # No fallback - raise the exception
        raise


def _prioritize_articles(db_client: ReaderDBClient, domain_scores: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Fetch unprocessed articles from yesterday and today and prioritize them based on combined score.

    Calculates the Combined_Priority_Score = (0.65 * cluster_hotness_score) + (0.35 * goodness_score)
    and assigns articles to processing tiers based on their ranking:
    - Tier 0: Top ~30%
    - Tier 1: Next ~50%
    - Tier 2: Remainder ~20%

    Args:
        db_client: Database client instance
        domain_scores: Optional pre-loaded dictionary mapping domains to their goodness scores

    Returns:
        List of prioritized article dictionaries with processing tier assigned
    """
    try:
        # Get unprocessed articles from yesterday and today only
        try:
            articles = db_client.get_recent_day_unprocessed_articles()
            logger.info("Fetching articles published yesterday and today only")
        except Exception as e:
            logger.error(
                f"Error fetching articles from yesterday and today: {e}")
            # Fallback to recent unprocessed articles with days limit
            logger.info("Falling back to recent unprocessed articles")
            articles = db_client.get_recent_unprocessed_articles(
                days=DAYS_LOOKBACK, limit=ARTICLE_LIMIT)

        if not articles:
            logger.info(
                "No unprocessed articles found for yesterday and today.")
            return []

        logger.info(
            f"Found {len(articles)} unprocessed articles from yesterday and today for prioritization")

        # Get all domain scores if not provided
        if domain_scores is None:
            domain_scores = db_client.get_all_domain_goodness_scores()

        # Extract unique domains and cluster_ids for batch lookups
        domains_to_lookup = list(set(article.get('domain', '')
                                 for article in articles if article.get('domain')))
        cluster_ids_to_lookup = list(set(article.get('cluster_id') for article in articles
                                         if article.get('cluster_id') is not None))

        # Batch lookup domain goodness scores
        if domains_to_lookup and not domain_scores:
            domain_scores = db_client.get_domain_goodness_scores(
                domains_to_lookup)

        # Batch lookup cluster hotness scores
        cluster_hotness = {}
        if cluster_ids_to_lookup:
            cluster_hotness = db_client.get_cluster_hotness_scores(
                cluster_ids_to_lookup)

        # Calculate scores for each article using the specified weights
        CLUSTER_WEIGHT = 0.65  # Weight for cluster hotness (65%)
        DOMAIN_WEIGHT = 0.35   # Weight for domain goodness (35%)

        for article in articles:
            # Get domain goodness score (default 0.5 if missing)
            domain = article.get('domain', '')
            domain_goodness_score = domain_scores.get(domain, 0.5)

            # Get cluster hotness score (default 0.0 if missing)
            cluster_id = article.get('cluster_id')
            cluster_hotness_score = cluster_hotness.get(
                cluster_id, 0.0) if cluster_id else 0.0

            # Calculate combined score
            combined_score = (CLUSTER_WEIGHT * cluster_hotness_score +
                              DOMAIN_WEIGHT * domain_goodness_score)

            # Store scores in article dict for debugging and future use
            article['domain_goodness_score'] = domain_goodness_score
            article['cluster_hotness_score'] = cluster_hotness_score
            article['combined_score'] = combined_score

        # Sort articles by combined score (descending)
        sorted_articles = sorted(
            articles, key=lambda x: x.get('combined_score', 0.0), reverse=True)

        # Assign processing tiers based on percentages of the sorted list
        total_articles = len(sorted_articles)
        # Calculate tier cutoffs based on percentages
        tier0_cutoff = int(total_articles * 0.3)  # Top 30%
        # Top 80% (so next 50% after tier0)
        tier1_cutoff = int(total_articles * 0.8)

        for i, article in enumerate(sorted_articles):
            if i < tier0_cutoff:
                article['processing_tier'] = 0  # Top 30% - Highest quality
            elif i < tier1_cutoff:
                article['processing_tier'] = 1  # Next 50% - Medium quality
            else:
                article['processing_tier'] = 2  # Bottom 20% - Standard quality

        # Count articles in each tier for logging
        tier0_count = sum(
            1 for a in sorted_articles if a.get('processing_tier') == 0)
        tier1_count = sum(
            1 for a in sorted_articles if a.get('processing_tier') == 1)
        tier2_count = sum(
            1 for a in sorted_articles if a.get('processing_tier') == 2)

        logger.info(
            f"Prioritized {len(sorted_articles)} articles: {tier0_count} in tier 0 (30%), "
            f"{tier1_count} in tier 1 (50%), {tier2_count} in tier 2 (20%)")

        return sorted_articles

    except Exception as e:
        logger.error(f"Error prioritizing articles: {e}", exc_info=True)
        return []


async def _extract_entities_batch(gemini_client: GeminiClient, articles: List[Dict[str, Any]],
                                  task_manager: TaskManager) -> Dict[int, Any]:
    """
    Extract entities from a batch of articles using Gemini API concurrently.

    Args:
        gemini_client: Initialized GeminiClient instance
        articles: Batch of article dictionaries to process
        task_manager: TaskManager instance to use for concurrent processing

    Returns:
        Dictionary mapping article IDs to their extracted entity data (or error dict)
    """
    logger.debug(
        f"Preparing to extract entities for {len(articles)} articles using TaskManager")

    # Create task definitions for each article
    tasks_definitions = []

    # Get model preferences from client
    preferred_models = gemini_client.preferred_model_ids
    fallback_model_id = gemini_client.fallback_model_id
    num_preferred = len(preferred_models)

    for article in articles:
        article_id = article.get('id')
        content = article.get('content')
        tier = article.get('processing_tier')  # Tier 0, 1, or 2

        # --- Determine model_to_use and fallback_model based on tier ---
        model_to_use = fallback_model_id  # Default to fallback
        fallback_model = fallback_model_id  # Default to fallback

        if tier == 0:  # Highest priority tier
            model_to_use = preferred_models[0] if num_preferred > 0 else fallback_model_id
            # Fallback for tier 0 is the second preference or the global fallback
            if num_preferred > 1:
                fallback_model = preferred_models[1]
            else:
                fallback_model = fallback_model_id
        elif tier == 1:  # Medium priority tier
            # Use second preference or global fallback as primary
            if num_preferred > 1:
                model_to_use = preferred_models[1]
            else:
                model_to_use = fallback_model_id
            # Fallback for tier 1 is always the global fallback
            fallback_model = fallback_model_id
        # Tier 2 uses fallback_model_id for both primary and fallback (already set as default)

        # model_to_use = article.get( # Old logic using TIER_MODELS
        #     'model_to_use', TIER_MODELS[tier]['primary'])
        # fallback_model = article.get(
        #     'fallback_model', TIER_MODELS[tier]['fallback'])
        # --- End model selection ---

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
            'model_to_use': model_to_use,
            'fallback_model': fallback_model
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
        # Use the provided TaskManager instance
        extraction_results = await task_manager.run_tasks(gemini_client, tasks_definitions)

        # Log rate limiter status after batch completion
        # if hasattr(gemini_client, 'rate_limiter') and gemini_client.rate_limiter:
        #     for model in gemini_client.ALL_MODEL_RPMS.keys(): # Old way
        #         current_rpm = gemini_client.rate_limiter.get_current_rpm(model)
        #         wait_time = gemini_client.rate_limiter.get_wait_time(model)
        #         logger.info(
        #             f"Rate limiter status - Model: {model}, Current RPM: {current_rpm}, Wait time: {wait_time:.2f}s")
        # ^^^ This logging is now done inside the main run loop, removed from here ^^^

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
    Also processes relational data: event mentions, policy mentions, and entity co-occurrence contexts.

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
    # New counters for relational data
    events_created = 0
    policies_created = 0
    relationships_recorded = 0

    if not entity_results:
        return {"processed": 0, "links": 0, "snippets": 0, "errors": 0,
                "events": 0, "policies": 0, "relationships": 0}

    # First, collect all unique entity types from the results to add them to the database
    unique_entity_types = set()

    # Extract all entity types from the results
    for article_id, extracted_data in entity_results.items():
        # Skip if error or not a dict
        if not isinstance(extracted_data, dict) or 'error' in extracted_data:
            continue

        # Find entity list in various formats
        entity_list = None
        if 'ents' in extracted_data and isinstance(extracted_data['ents'], list):
            entity_list = extracted_data['ents']
        elif 'entities' in extracted_data and isinstance(extracted_data['entities'], list):
            entity_list = extracted_data['entities']
        elif 'extracted_entities' in extracted_data and isinstance(extracted_data['extracted_entities'], list):
            entity_list = extracted_data['extracted_entities']
        elif isinstance(extracted_data, list):
            entity_list = extracted_data

        if entity_list:
            for entity in entity_list:
                # Get entity type from either short or long format
                entity_type = entity.get('et') or entity.get('entity_type')
                if entity_type:
                    unique_entity_types.add(entity_type)

    # Add all unique entity types to the database with default weights
    logger.info(
        f"Adding {len(unique_entity_types)} unique entity types to the database")
    for entity_type in unique_entity_types:
        try:
            db_client.add_entity_type_weight(entity_type)
        except Exception as e:
            logger.warning(f"Could not add entity type {entity_type}: {e}")

    # Get list of article IDs to process in batches
    article_ids = list(entity_results.keys())
    batch_size = 10  # Process 10 articles at a time for DB operations
    total_batches = (len(article_ids) + batch_size - 1) // batch_size

    logger.debug(
        f"Storing entity results for {len(entity_results)} articles in batches of {batch_size}...")

    # Process articles in batches to provide incremental commits
    for batch_index in range(0, len(article_ids), batch_size):
        batch_article_ids = article_ids[batch_index:batch_index + batch_size]
        batch_num = batch_index // batch_size + 1

        logger.debug(
            f"Processing storage batch {batch_num}/{total_batches} ({len(batch_article_ids)} articles)")

        # Track batch metrics for reporting
        batch_processed = 0
        batch_links = 0
        batch_snippets = 0
        batch_errors = 0
        batch_events = 0
        batch_policies = 0
        batch_relationships = 0

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
                    logger.debug(
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

                # ----------------------------------------------------
                # Process Event Mentions if present
                # ----------------------------------------------------
                event_mentions = None
                if isinstance(extracted_data, dict) and 'ev_mentions' in extracted_data and isinstance(extracted_data['ev_mentions'], list):
                    event_mentions = extracted_data['ev_mentions']
                    logger.debug(
                        f"Found {len(event_mentions)} event mentions for article {article_id}")

                    for event in event_mentions:
                        try:
                            # Extract event information
                            event_title = event.get('ti')
                            event_type = event.get('ty')
                            date_mention = event.get('dt')
                            entity_mentions = event.get('ent_mens', [])

                            if not event_title or not event_type:
                                logger.warning(
                                    f"Skipping event with missing title or type in article {article_id}: {event}")
                                continue

                            # Create or find the event
                            event_id = db_client.find_or_create_event(
                                title=event_title,
                                event_type=event_type,
                                date_mention=date_mention
                            )

                            if not event_id:
                                logger.warning(
                                    f"Failed to create/find event '{event_title}' for article {article_id}")
                                continue

                            # Process entity mentions for this event
                            for entity_mention in entity_mentions:
                                # Check if entity_mention is a string (old format) or dict (new format)
                                if isinstance(entity_mention, dict):
                                    entity_name = entity_mention.get('en')
                                    entity_role = entity_mention.get(
                                        'role', 'MENTIONED')
                                else:
                                    entity_name = entity_mention
                                    entity_role = 'MENTIONED'  # Default for backward compatibility

                                if not entity_name:
                                    continue

                                # Find or create the entity - use ORGANIZATION as default type since we don't have specific type info
                                entity_id = db_client.find_or_create_entity(
                                    name=entity_name,
                                    entity_type="ORGANIZATION"  # Default type for entities mentioned in events
                                )

                                if not entity_id:
                                    logger.warning(
                                        f"Failed to create/find entity '{entity_name}' for event {event_id}")
                                    continue

                                # Link the entity to the event with the specified role
                                link_success = db_client.link_event_entity(
                                    event_id=event_id,
                                    entity_id=entity_id,
                                    role=entity_role
                                )

                                if link_success:
                                    batch_events += 1
                        except Exception as e:
                            logger.error(
                                f"Error processing event mention in article {article_id}: {e}", exc_info=True)

                # ----------------------------------------------------
                # Process Policy Mentions if present
                # ----------------------------------------------------
                policy_mentions = None
                if isinstance(extracted_data, dict) and 'pol_mentions' in extracted_data and isinstance(extracted_data['pol_mentions'], list):
                    policy_mentions = extracted_data['pol_mentions']
                    logger.debug(
                        f"Found {len(policy_mentions)} policy mentions for article {article_id}")

                    for policy in policy_mentions:
                        try:
                            # Extract policy information
                            policy_title = policy.get('ti')
                            policy_type = policy.get('ty')
                            date_mention = policy.get('edt')
                            entity_mentions = policy.get('ent_mens', [])

                            if not policy_title or not policy_type:
                                logger.warning(
                                    f"Skipping policy with missing title or type in article {article_id}: {policy}")
                                continue

                            # Create or find the policy
                            policy_id = db_client.find_or_create_policy(
                                title=policy_title,
                                policy_type=policy_type,
                                date_mention=date_mention
                            )

                            if not policy_id:
                                logger.warning(
                                    f"Failed to create/find policy '{policy_title}' for article {article_id}")
                                continue

                            # Process entity mentions for this policy
                            for entity_mention in entity_mentions:
                                # Check if entity_mention is a string (old format) or dict (new format)
                                if isinstance(entity_mention, dict):
                                    entity_name = entity_mention.get('en')
                                    entity_role = entity_mention.get(
                                        'role', 'MENTIONED')
                                else:
                                    entity_name = entity_mention
                                    entity_role = 'MENTIONED'  # Default for backward compatibility

                                if not entity_name:
                                    continue

                                # Find or create the entity - use ORGANIZATION as default type since we don't have specific type info
                                entity_id = db_client.find_or_create_entity(
                                    name=entity_name,
                                    entity_type="ORGANIZATION"  # Default type for entities mentioned in policies
                                )

                                if not entity_id:
                                    logger.warning(
                                        f"Failed to create/find entity '{entity_name}' for policy {policy_id}")
                                    continue

                                # Link the entity to the policy with the specified role
                                link_success = db_client.link_policy_entity(
                                    policy_id=policy_id,
                                    entity_id=entity_id,
                                    role=entity_role
                                )

                                if link_success:
                                    batch_policies += 1
                        except Exception as e:
                            logger.error(
                                f"Error processing policy mention in article {article_id}: {e}", exc_info=True)

                # ----------------------------------------------------
                # Process Relationship Contexts if present
                # ----------------------------------------------------
                rel_contexts = None
                if isinstance(extracted_data, dict) and 'rel_contexts' in extracted_data and isinstance(extracted_data['rel_contexts'], list):
                    rel_contexts = extracted_data['rel_contexts']
                    logger.debug(
                        f"Found {len(rel_contexts)} relationship contexts for article {article_id}")

                    for context in rel_contexts:
                        try:
                            # Extract relationship information
                            entity1_name = context.get('e1n')
                            entity2_name = context.get('e2n')
                            # Extract entity types if available in the updated format
                            # Default to ORGANIZATION if not specified
                            entity1_type = context.get('e1t', 'ORGANIZATION')
                            # Default to ORGANIZATION if not specified
                            entity2_type = context.get('e2t', 'ORGANIZATION')
                            context_type = context.get('ctx_ty')

                            # Handle both formats:
                            # - Old: 'evi' is a list of strings
                            # - New: 'evi' is a single string representing the best evidence
                            evidence = context.get('evi')
                            evidence_snippet = None

                            if isinstance(evidence, list) and evidence:
                                # Old format - take the first snippet from the list
                                evidence_snippet = evidence[0]
                            elif isinstance(evidence, str):
                                # New format - direct string
                                evidence_snippet = evidence

                            if not entity1_name or not entity2_name or not context_type:
                                logger.warning(
                                    f"Skipping relationship with missing entity names or context type in article {article_id}: {context}")
                                continue

                            # Find or create the entities with their types
                            entity_id_1 = db_client.find_or_create_entity(
                                name=entity1_name, entity_type=entity1_type)
                            entity_id_2 = db_client.find_or_create_entity(
                                name=entity2_name, entity_type=entity2_type)

                            if not entity_id_1 or not entity_id_2 or entity_id_1 == entity_id_2:
                                logger.warning(
                                    f"Failed to create valid entity pair '{entity1_name}' and '{entity2_name}' for relationship")
                                continue

                            # Record the relationship context
                            rel_success = db_client.record_relationship_context(
                                entity_id_1=entity_id_1,
                                entity_id_2=entity_id_2,
                                context_type=context_type,
                                article_id=article_id,
                                evidence_snippet=evidence_snippet
                            )

                            if rel_success:
                                batch_relationships += 1
                        except Exception as e:
                            logger.error(
                                f"Error processing relationship context in article {article_id}: {e}", exc_info=True)

                # ----------------------------------------------------
                # Continue with regular entity processing
                # ----------------------------------------------------
                if entity_list_to_process is not None:  # Proceed if we found a valid list of entities
                    # Check if the list is empty
                    if not entity_list_to_process:
                        logger.debug(
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
                            logger.debug(
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
        events_created += batch_events
        policies_created += batch_policies
        relationships_recorded += batch_relationships

        logger.info(
            f"Batch {batch_num}/{total_batches} complete: {batch_processed} processed, {batch_links} entity links, " +
            f"{batch_snippets} snippets, {batch_events} event links, {batch_policies} policy links, " +
            f"{batch_relationships} relationships, {batch_errors} errors")

    # Return overall summary
    return {
        "processed": processed_count,
        "links": entity_links_created,
        "snippets": snippets_stored,
        "errors": errors,
        "events": events_created,
        "policies": policies_created,
        "relationships": relationships_recorded
    }
