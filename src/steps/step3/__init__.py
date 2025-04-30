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
INTER_BATCH_DELAY_SECONDS = 40  # Delay between batches

# --- Length Scoring Function (New) ---


def calculate_length_score(length_chars: Optional[int],
                           min_viable: int = 2000,
                           optimal_start: int = 4000,
                           optimal_end: int = 18000,
                           max_reasonable: int = 35000,
                           long_article_min_score: float = 0.3) -> float:
    """
    Calculates a score (0.0-1.0) based on article character length.

    Args:
        length_chars: Character count of the article content.
        min_viable: Below this length, score is 0.
        optimal_start: Score ramps from 0 to 1 between min_viable and optimal_start.
        optimal_end: Score is 1.0 between optimal_start and optimal_end.
        max_reasonable: Score ramps from 1 down to long_article_min_score between optimal_end and max_reasonable.
        long_article_min_score: Minimum score assigned to articles longer than max_reasonable.

    Returns:
        Normalized length score.
    """
    if not isinstance(length_chars, int) or length_chars <= 0:
        # Handle cases where length wasn't fetched or content is empty/null
        return 0.0

    if length_chars < min_viable:
        return 0.0
    elif length_chars < optimal_start:
        # Ensure divisor is not zero
        if optimal_start == min_viable:
            return 1.0  # Or 0.0 depending on desired edge case handling
        # Linear increase from 0 to 1
        return (length_chars - min_viable) / (optimal_start - min_viable)
    elif length_chars <= optimal_end:
        return 1.0
    elif length_chars <= max_reasonable:
        # Ensure divisor is not zero
        if max_reasonable == optimal_end:
            return long_article_min_score
        # Linear decrease from 1 down to long_article_min_score
        progress = (length_chars - optimal_end) / \
            (max_reasonable - optimal_end)
        return 1.0 - (1.0 - long_article_min_score) * progress
    else:
        # Score for very long articles
        return long_article_min_score

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
    2. Fetches and prioritizes recent unprocessed articles
    3. Extracts entities using Gemini API with tier-based model selection using TaskManager for concurrency
    4. Stores entity results in the database
    5. Updates article processing status

    Returns:
        Dict[str, Any]: Status report containing metrics about the extraction process
    """
    start_time = time.monotonic()
    batch_index = 0
    processed_count = 0
    empty_passes = 0
    consecutive_failures = 0
    entity_counter = 0

    # Collect environment variables for configuration
    max_wait_seconds = float(os.getenv("GEMINI_MAX_WAIT_SECONDS", "60.0"))
    emergency_fallback_wait = float(
        os.getenv("EMERGENCY_FALLBACK_WAIT_SECONDS", "120.0"))
    # Update logging to show our wait time configuration
    logger.debug(f"Entity extraction configured with max_wait_seconds={max_wait_seconds}s, "
                 f"emergency_fallback_wait={emergency_fallback_wait}s")

    # Load model IDs from environment variables
    # Tier 0 (highest priority) models
    tier0_primary_model = os.getenv(
        "GEMINI_MODEL_PREF_1", "gemini-2.0-flash-exp")
    tier0_fallback_model = os.getenv("GEMINI_MODEL_PREF_2", "gemini-2.0-flash")

    # Tier 1 (medium priority) models
    tier1_primary_model = os.getenv("GEMINI_MODEL_PREF_2", "gemini-2.0-flash")
    tier1_fallback_model = os.getenv(
        "GEMINI_MODEL_PREF_3", "gemini-2.0-flash-lite")

    # Tier 2 (lowest priority) models
    tier2_primary_model = os.getenv(
        "GEMINI_MODEL_PREF_3", "gemini-2.0-flash-lite")
    tier2_fallback_model = os.getenv(
        "GEMINI_FALLBACK_MODEL_ID", "gemini-2.0-flash")

    try:
        # Initialize DB and API clients
        db_client = ReaderDBClient()
        gemini_client = GeminiClient()
        task_manager = TaskManager()

        # Configure the GeminiClient with wait time settings
        gemini_client.set_max_rate_limit_wait_seconds(max_wait_seconds)
        gemini_client.set_emergency_wait_seconds(emergency_fallback_wait)

        # Get domain goodness scores
        domain_scores = _get_domain_goodness_scores(db_client)

        # Prioritize articles based on domain and cluster scores
        articles = _prioritize_articles(db_client, domain_scores)
        if not articles:
            logger.warning("No articles needing entity extraction. Exiting.")
            return {
                "success": True,
                "processed": 0,
                "entity_links_created": 0,
                "runtime_seconds": 0
            }

        total_articles = len(articles)
        logger.info(
            f"Starting entity extraction for {total_articles} prioritized articles")

        # Calculate initial total articles and estimate total batches
        total_articles_to_process = len(articles)
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
        while articles:
            batch_start_time = time.time()  # Record batch start time
            batch_index += 1

            # --- Start: Dynamic Batch Sizing ---
            adjusted_batch_size = BATCH_SIZE  # Start with default
            if batch_index > 1 and gemini_client.rate_limiter:  # Check after the first batch if limiter exists
                rate_limited_models_count = 0
                total_models_checked = 0
                # Check current rate limit status for generation models
                for model_name in gemini_client.available_gen_models:
                    total_models_checked += 1
                    current_rpm = gemini_client.rate_limiter.get_current_rpm(
                        model_name)
                    limit = gemini_client.rate_limiter.model_rpm_limits.get(
                        model_name, 0)
                    available_slots = max(0, limit - current_rpm)
                    # Consider a model rate-limited if less than ~1/3rd of batch size slots are available
                    if available_slots < BATCH_SIZE / 3:
                        rate_limited_models_count += 1

                # If a significant portion of models are rate-limited, reduce batch size
                # E.g., if 50% or more models are limited
                if total_models_checked > 0 and rate_limited_models_count / total_models_checked >= 0.5:
                    previous_adjusted_batch_size = adjusted_batch_size
                    # Reduce batch size by half, minimum 3
                    adjusted_batch_size = max(3, BATCH_SIZE // 2)
                    if adjusted_batch_size < previous_adjusted_batch_size:
                        logger.info(
                            f"Reduced batch size from {previous_adjusted_batch_size} to {adjusted_batch_size} due to rate limit pressure ({rate_limited_models_count}/{total_models_checked} models limited).")
            # --- End: Dynamic Batch Sizing ---

            # Use adjusted_batch_size for composing the batch
            target_batch_size = adjusted_batch_size

            # Compose a balanced batch with the defined ratio (4/5/1) up to target_batch_size
            current_batch = []
            original_tier_counts = {0: 0, 1: 0, 2: 0}
            rebalanced_tier_counts = {0: 0, 1: 0, 2: 0}

            # --- Tier 0 ---
            tier0_target = int(target_batch_size * 0.5)  # Approx 50%
            tier0_added = 0
            temp_articles_tier0 = []
            remaining_articles_after_tier0 = []
            for article in articles:
                if article.get('processing_tier') == 0 and tier0_added < tier0_target:
                    temp_articles_tier0.append(article)
                    tier0_added += 1
                else:
                    remaining_articles_after_tier0.append(article)
            current_batch.extend(temp_articles_tier0)
            original_tier_counts[0] = tier0_added
            articles = remaining_articles_after_tier0  # Update remaining articles

            # --- Tier 1 ---
            tier1_target = int(target_batch_size * 0.4)  # Approx 40%
            tier1_added = 0
            temp_articles_tier1 = []
            remaining_articles_after_tier1 = []
            for article in articles:
                if article.get('processing_tier') == 1 and tier1_added < tier1_target:
                    temp_articles_tier1.append(article)
                    tier1_added += 1
                else:
                    remaining_articles_after_tier1.append(article)
            current_batch.extend(temp_articles_tier1)
            original_tier_counts[1] = tier1_added
            articles = remaining_articles_after_tier1  # Update remaining articles

            # --- Tier 2 ---
            tier2_target = target_batch_size - \
                len(current_batch)  # Fill remaining
            tier2_added = 0
            temp_articles_tier2 = []
            remaining_articles_after_tier2 = []
            for article in articles:
                if article.get('processing_tier') == 2 and tier2_added < tier2_target:
                    temp_articles_tier2.append(article)
                    tier2_added += 1
                else:
                    remaining_articles_after_tier2.append(article)
            current_batch.extend(temp_articles_tier2)
            original_tier_counts[2] = tier2_added
            articles = remaining_articles_after_tier2  # Update remaining articles

            # Smart rebalancing logic (fill remaining slots if batch is not full)
            remaining_slots = target_batch_size - len(current_batch)
            if remaining_slots > 0 and articles:
                logger.debug(
                    f"Rebalancing: {remaining_slots} slots left in batch {batch_index}. Filling from remaining {len(articles)} articles.")
                # Prioritize filling with remaining articles regardless of original tier (already sorted by priority)
                fill_count = min(remaining_slots, len(articles))
                articles_to_add_rebalance = articles[:fill_count]
                current_batch.extend(articles_to_add_rebalance)
                articles = articles[fill_count:]

                # Update rebalancing counts for logging
                for article in articles_to_add_rebalance:
                    # Default to tier 2 if missing
                    tier = article.get('processing_tier', 2)
                    rebalanced_tier_counts[tier] += 1

            # If no articles were added to the batch, we're done
            if not current_batch:
                break

            # Log detailed information about tier distribution
            was_rebalanced = any(
                count > 0 for count in rebalanced_tier_counts.values())

            if was_rebalanced:
                logger.debug(
                    f"Processing batch {batch_index} with {len(current_batch)} articles | "
                    f"Original distribution: Tier 0: {original_tier_counts[0]}, "
                    f"Tier 1: {original_tier_counts[1]}, Tier 2: {original_tier_counts[2]} | "
                    f"Added during rebalancing: Tier 0: {rebalanced_tier_counts[0]}, "
                    f"Tier 1: {rebalanced_tier_counts[1]}, Tier 2: {rebalanced_tier_counts[2]}"
                )
            else:
                logger.debug(
                    f"Processing batch {batch_index} with {len(current_batch)} articles | "
                    f"Standard distribution: Tier 0: {original_tier_counts[0]}, "
                    f"Tier 1: {original_tier_counts[1]}, Tier 2: {original_tier_counts[2]}"
                )

            # Assign appropriate models to each article based on tier
            for article in current_batch:
                tier = article.get('processing_tier')
                if tier == 0:
                    article['model_to_use'] = tier0_primary_model
                    article['fallback_model'] = tier0_fallback_model
                elif tier == 1:
                    article['model_to_use'] = tier1_primary_model
                    article['fallback_model'] = tier1_fallback_model
                elif tier == 2:
                    article['model_to_use'] = tier2_primary_model
                    article['fallback_model'] = tier2_fallback_model
                else:
                    # Default to tier 1 models if tier is missing or invalid
                    logger.warning(
                        f"Article {article.get('id')} has invalid tier {tier}. Defaulting to tier 1 models.")
                    article['model_to_use'] = tier1_primary_model
                    article['fallback_model'] = tier1_fallback_model

            # Heartbeat logging (modified)
            current_time = time.time()
            if current_time - last_heartbeat >= heartbeat_interval:
                processed_count = total_processed  # Use current total processed count
                remaining_articles = len(articles)
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

            # --- Start: Improved Inter-Batch Cooldown ---
            # Calculate max wait time needed across all models
            max_model_wait_time = 0.0
            if hasattr(gemini_client, 'rate_limiter') and gemini_client.rate_limiter:
                # Use available_gen_models from the client to check relevant models
                models_to_check = gemini_client.available_gen_models
                for model_name in models_to_check:
                    try:
                        wait_time = gemini_client.rate_limiter.get_wait_time(
                            model_name)
                        max_model_wait_time = max(
                            max_model_wait_time, wait_time)
                        logger.debug(
                            f"Cooldown Check: Model {model_name} needs {wait_time:.2f}s wait.")
                    except Exception as e:
                        logger.warning(
                            f"Error getting wait time for model {model_name}: {e}")

            # Determine cooldown duration
            cooldown_duration = INTER_BATCH_DELAY_SECONDS  # Default cooldown
            if max_model_wait_time > INTER_BATCH_DELAY_SECONDS:
                # If any model needs more time than the default, increase cooldown
                # Cap the dynamic cooldown to avoid excessively long waits (e.g., 30 seconds max)
                # Add 1s buffer, cap at 30s
                cooldown_duration = min(max_model_wait_time + 1.0, 30.0)
                logger.info(
                    f"Extended inter-batch cooldown to {cooldown_duration:.2f}s based on model rate limits (max needed: {max_model_wait_time:.2f}s).")
            else:
                logger.info(
                    f"Using standard inter-batch cooldown of {cooldown_duration}s.")

            # Apply the calculated cooldown
            if cooldown_duration > 0:
                await asyncio.sleep(cooldown_duration)
            # --- End: Improved Inter-Batch Cooldown ---

        # Prepare final status report
        status = {
            "success": True,
            "articles_found": len(articles),
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

    Calculates the Combined_Priority_Score = (0.375 * cluster_hotness_score) + (0.375 * domain_goodness_score) + (0.25 * length_score)
    and assigns articles to processing tiers based on their ranking:
    - Tier 0: Top ~40%
    - Tier 1: Next ~40%
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
            # Ensure the DB call returns 'content_length'
            articles = db_client.get_recent_day_unprocessed_articles()
            logger.info("Fetching articles published yesterday and today only")
        except Exception as e:
            logger.error(
                f"Error fetching articles from yesterday and today: {e}")
            # Fallback to recent unprocessed articles with days limit
            logger.info("Falling back to recent unprocessed articles")
            # Ensure fallback also returns 'content_length' (modify DB call if needed)
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

        # Calculate scores for each article using the updated weights
        CLUSTER_WEIGHT = 0.375  # Weight for cluster hotness (37.5%)
        DOMAIN_WEIGHT = 0.375   # Weight for domain goodness (37.5%)
        LENGTH_WEIGHT = 0.25    # Weight for article length (25%)

        for article in articles:
            # Get domain goodness score (default 0.5 if missing)
            domain = article.get('domain', '')
            domain_goodness_score = domain_scores.get(domain, 0.5)

            # Get cluster hotness score (default 0.0 if missing)
            cluster_id = article.get('cluster_id')
            cluster_hotness_score = cluster_hotness.get(
                cluster_id, 0.0) if cluster_id else 0.0

            # Get content length and calculate length score
            content_length = article.get('content_length')  # Fetched in Step 1
            length_score = calculate_length_score(
                content_length)  # Use function from Step 2

            # Calculate combined score with length included
            combined_score = (CLUSTER_WEIGHT * cluster_hotness_score +
                              DOMAIN_WEIGHT * domain_goodness_score +
                              LENGTH_WEIGHT * length_score)

            # Store scores in article dict for debugging and future use
            article['domain_goodness_score'] = domain_goodness_score
            article['cluster_hotness_score'] = cluster_hotness_score
            article['length_score'] = length_score  # Store the new score
            article['combined_score'] = combined_score

        # Sort articles by combined score (descending)
        sorted_articles = sorted(
            articles, key=lambda x: x.get('combined_score', 0.0), reverse=True)

        # Assign processing tiers based on percentages of the sorted list
        total_articles = len(sorted_articles)
        # Calculate tier cutoffs based on percentages
        tier0_cutoff = int(total_articles * 0.4)  # Top 40%
        # Next 40% (up to 80% cumulative)
        tier1_cutoff = int(total_articles * 0.8)

        for i, article in enumerate(sorted_articles):
            if i < tier0_cutoff:
                article['processing_tier'] = 0  # Top 40%
            elif i < tier1_cutoff:
                article['processing_tier'] = 1  # Next 40%
            else:
                article['processing_tier'] = 2  # Bottom 20%

        # Count articles in each tier for logging
        tier0_count = sum(
            1 for a in sorted_articles if a.get('processing_tier') == 0)
        tier1_count = sum(
            1 for a in sorted_articles if a.get('processing_tier') == 1)
        tier2_count = sum(
            1 for a in sorted_articles if a.get('processing_tier') == 2)

        logger.info(
            f"Prioritized {len(sorted_articles)} articles: {tier0_count} in tier 0 (40%), "
            f"{tier1_count} in tier 1 (40%), {tier2_count} in tier 2 (20%)")

        return sorted_articles

    except Exception as e:
        logger.error(f"Error prioritizing articles: {e}", exc_info=True)
        return []


async def _extract_entities_batch(gemini_client: GeminiClient, articles: List[Dict[str, Any]],
                                  task_manager: TaskManager) -> Dict[int, Any]:
    """
    Extracts entities from a batch of articles using the TaskManager and GeminiClient.
    Implements batch-level model pre-allocation based on rate limits.

    Args:
        gemini_client: The GeminiClient for API calls
        articles: List of article dictionaries
        task_manager: The TaskManager for concurrent API calls

    Returns:
        Dict[int, Any]: Dictionary mapping article IDs to their extraction results
    """
    logger.debug(
        f"Entering _extract_entities_batch for {len(articles)} articles.")
    if not articles:
        logger.warning("Empty batch passed to _extract_entities_batch")
        return {}

    # --- Start: Refactored Pre-allocation Logic (No explicit lock acquisition here) ---
    # Group articles by their assigned model_to_use
    model_groups = {}
    for article in articles:
        model = article.get('model_to_use')
        if model:
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(article)
        else:
            logger.warning(
                f"Article {article.get('id')} missing 'model_to_use' assignment. Skipping.")

    # Check if any articles have assigned models
    if not model_groups:
        logger.warning("No articles with assigned models in the batch.")
        return {}

    # Estimate available slots without holding lock for long periods
    revised_articles = []
    # Removed available_slots_cache

    for model_name, model_articles in model_groups.items():
        estimated_slots_available = 0
        limit = 0  # Initialize limit
        if gemini_client.rate_limiter:
            # Get limit for the model
            limit = gemini_client.rate_limiter.model_rpm_limits.get(
                model_name, 0)
            # Get current RPM (method handles its own brief lock)
            try:
                current_rpm = gemini_client.rate_limiter.get_current_rpm(
                    model_name)
                estimated_slots_available = max(0, limit - current_rpm)
            except Exception as e:
                logger.error(
                    f"Error getting current RPM for {model_name}: {e}. Assuming 0 slots.", exc_info=True)
                estimated_slots_available = 0  # Assume 0 if error occurs
        else:
            # No rate limiter, assume infinite slots (or handle as error)
            estimated_slots_available = len(model_articles)
            limit = estimated_slots_available  # Set limit for logging clarity
            logger.warning(
                "No rate limiter available, assuming all slots are open.")

        logger.debug(
            f"Model {model_name}: Limit={limit}, Est. Current RPM based estimate={current_rpm if gemini_client.rate_limiter else 'N/A'}, Est. Available Slots={estimated_slots_available}")

        articles_reassigned = 0
        for i, article in enumerate(model_articles):
            # Check based on ESTIMATED slots
            if i < estimated_slots_available:
                # Enough estimated slots, keep original model
                revised_articles.append(article)
            else:
                # Rate limit *likely* hit for the primary model based on estimate, try fallback or alternatives
                articles_reassigned += 1
                tier = article.get('processing_tier')
                fallback_model = article.get('fallback_model')
                reassigned = False

                # 1. Try Fallback Model
                if fallback_model and fallback_model != model_name:
                    # Estimate slots for fallback
                    fallback_slots = 0
                    if gemini_client.rate_limiter:
                        try:
                            fallback_limit = gemini_client.rate_limiter.model_rpm_limits.get(
                                fallback_model, 0)
                            fallback_rpm = gemini_client.rate_limiter.get_current_rpm(
                                fallback_model)
                            fallback_slots = max(
                                0, fallback_limit - fallback_rpm)
                        except Exception as e:
                            logger.error(
                                f"Error getting current RPM for fallback {fallback_model}: {e}. Assuming 0 slots.", exc_info=True)
                            fallback_slots = 0
                    else:
                        fallback_slots = 1  # Assume at least one slot if no limiter

                    # Use the estimate
                    if fallback_slots > 0:
                        article['model_to_use'] = fallback_model
                        revised_articles.append(article)
                        # Decrementing the estimate locally for this loop's check
                        estimated_slots_available -= 1  # Decrement original model slot estimate implicitly
                        # We don't decrement fallback_slots estimate as it's just a check
                        logger.info(
                            f"Article {article.get('id')} reassigned from {model_name} to fallback {fallback_model} due to ESTIMATED rate limit.")
                        reassigned = True

                # 2. Try Other Available Models (if fallback failed or wasn't applicable)
                if not reassigned and gemini_client.rate_limiter:
                    for alt_model, alt_limit in gemini_client.rate_limiter.model_rpm_limits.items():
                        # Skip the original and fallback models
                        if alt_model != model_name and alt_model != fallback_model:
                            alt_slots = 0
                            try:
                                alt_rpm = gemini_client.rate_limiter.get_current_rpm(
                                    alt_model)
                                alt_slots = max(0, alt_limit - alt_rpm)
                            except Exception as e:
                                logger.error(
                                    f"Error getting current RPM for alternative {alt_model}: {e}. Assuming 0 slots.", exc_info=True)
                                alt_slots = 0

                            # Use the estimate
                            if alt_slots > 0:
                                article['model_to_use'] = alt_model
                                revised_articles.append(article)
                                # Decrementing the estimate locally
                                estimated_slots_available -= 1  # Decrement original model slot estimate implicitly
                                # We don't decrement alt_slots estimate
                                logger.info(
                                    f"Article {article.get('id')} reassigned from {model_name} to alternative {alt_model} due to ESTIMATED rate limits.")
                                reassigned = True
                                break  # Found an alternative

                # 3. If still not reassigned, keep original but log warning
                if not reassigned:
                    logger.warning(
                        f"Article {article.get('id')} could not be reassigned from estimated rate-limited model {model_name}. Keeping original assignment, may face delays.")
                    # Keep original, task manager will handle wait
                    revised_articles.append(article)

        if articles_reassigned > 0:
            logger.info(
                f"Reassigned {articles_reassigned} articles originally intended for model {model_name} based on rate limit estimates.")

    # --- End: Refactored Pre-allocation Logic ---

    if not revised_articles:
        logger.warning(
            f"No articles remaining after pre-allocation checks. Skipping batch.")
        return {}

    logger.debug(
        f"Processing batch with {len(revised_articles)} articles after pre-allocation.")
    # Pass the list with potentially revised model assignments to the task manager
    logger.info(
        f"Calling task_manager.run_tasks for {len(revised_articles)} articles...")
    results = await task_manager.run_tasks(gemini_client, revised_articles)
    logger.info(
        f"task_manager.run_tasks completed for batch. Received {len(results)} results.")

    # Check for rate limiting errors (still useful for monitoring)
    rate_limit_errors = sum(1 for result in results.values()
                            if isinstance(result, dict) and
                            result.get('error', '').startswith(('API Rate Limit', 'Resource exhausted')))

    if rate_limit_errors > 0:
        logger.warning(
            f"Encountered {rate_limit_errors} rate limit errors during task execution in batch of {len(revised_articles)}")

    logger.debug(f"Exiting _extract_entities_batch.")
    return results


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
    logger.debug(
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
