#!/usr/bin/env python3
"""
influence.py - Entity Influence Score Calculation Module

This module implements advanced influence score calculation for entities by analyzing
mentions across articles, source quality, content context, and temporal factors.

Exported functions:
- calculate_entity_influence_score(conn, entity_id, recency_days): Calculates a comprehensive influence score
  - Returns float: Calculated influence score
- update_all_influence_scores(conn, entity_ids, recency_days): Updates influence scores for multiple entities
  - Returns Dict[str, Any]: Success and error counts

Related files:
- src/database/modules/entities.py: Basic entity operations
- src/database/reader_db_client.py: Client wrapper for these functions
- src/steps/step3/__init__.py: Uses these functions for entity processing
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import math

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder Weights - Adjust based on desired factor importance
# Removed context and type weights due to schema changes
WEIGHTS = {
    'base': 0.50,    # Base mention metrics (Increased weight)
    'quality': 0.25,  # Source quality (Increased weight)
    'temporal': 0.25  # Recency and trend (Increased weight)
}


def calculate_entity_influence_score(conn, entity_id: int, recency_days: int = 30) -> float:
    """
    Calculate comprehensive influence score for an entity based on multiple factors.

    Args:
        conn: Database connection
        entity_id: Entity ID to calculate score for
        recency_days: Number of days to prioritize for recency calculation

    Returns:
        Calculated influence score (0.0 if error or entity not found)
    """
    try:
        # Step 1: Collect raw data
        raw_data = _collect_entity_influence_data(
            conn, entity_id, recency_days)

        if raw_data is None:  # Handle case where entity data couldn't be collected
            logger.warning(
                f"Could not collect influence data for entity {entity_id}. Returning 0.")
            return 0.0

        # Step 2: Calculate component scores (each normalized to 0-1 range)
        base_score = _calculate_base_mention_score(raw_data)
        quality_score = _calculate_source_quality_score(raw_data)
        temporal_score = _calculate_temporal_score(raw_data)
        # Removed context_score calculation

        # Step 3: Apply weights to component scores
        # Using pre-defined WEIGHTS dictionary
        final_score = (
            WEIGHTS['base'] * base_score +
            WEIGHTS['quality'] * quality_score +
            WEIGHTS['temporal'] * temporal_score
        )

        # Step 4: Apply entity type modifier (REMOVED due to schema/complexity)
        # entity_type = raw_data.get('entity_type')
        # type_weight = _get_entity_type_weight(conn, entity_type)
        # final_score *= type_weight

        # Step 5: Store calculation factors for transparency (REMOVED DB storage)
        # Log the factors instead for now
        logger.debug(
            f"Influence factors for entity {entity_id}: base={base_score:.3f}, quality={quality_score:.3f}, temporal={temporal_score:.3f}, final={final_score:.3f}")
        # _store_influence_factors(conn, entity_id, { ... })

        # Step 6: Update the entity's influence score
        success = _update_entity_influence_score(conn, entity_id, final_score)
        if not success:
            logger.warning(
                f"Failed to update influence score for entity {entity_id} in the database.")
            # Decide if we should return 0.0 or the calculated score despite DB update failure
            # For now, return the calculated score but log the warning.

        return final_score

    except ValueError as ve:  # Catch specific error from data collection
        logger.error(
            f"Error calculating influence score for entity {entity_id}: {ve}")
        return 0.0
    except Exception as e:
        logger.error(
            f"Error calculating influence score for entity {entity_id}: {e}", exc_info=True)
        return 0.0


def update_all_influence_scores(conn, entity_ids: Optional[List[int]] = None,
                                recency_days: int = 30) -> Dict[str, Any]:
    """
    Update influence scores for multiple entities.

    Args:
        conn: Database connection
        entity_ids: List of entity IDs to update (if None, uses entities from recent articles)
        recency_days: Number of days to prioritize for recency calculation

    Returns:
        Dict with success count, error count, and total entities processed
    """
    target_entity_ids = []
    try:
        if entity_ids is None:
            # Get entities linked to articles published in the last N days
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT ae.entity_id
                FROM article_entities ae
                JOIN articles a ON ae.article_id = a.id
                WHERE a.pub_date >= (CURRENT_DATE - INTERVAL '%s DAYS')
            """, (recency_days,))
            target_entity_ids = [row[0] for row in cursor.fetchall()]
            cursor.close()
            logger.info(
                f"Found {len(target_entity_ids)} entities from recent articles ({recency_days} days) for influence update.")
        else:
            target_entity_ids = entity_ids
            logger.info(
                f"Updating influence for {len(target_entity_ids)} specified entities.")

        if not target_entity_ids:
            logger.info("No entities found to update influence scores.")
            return {
                "success_count": 0,
                "error_count": 0,
                "total_entities": 0
            }

        success_count = 0
        error_count = 0

        for entity_id in target_entity_ids:
            # Add a small delay or check to avoid overwhelming resources if needed
            try:
                # Recalculate score
                score = calculate_entity_influence_score(
                    conn, entity_id, recency_days)
                # The score is updated within calculate_entity_influence_score
                # We just log success/failure of the overall calculation here.
                # Note: calculate_entity_influence_score returns 0.0 on error.
                if score >= 0:  # Consider any non-negative score a success for the process
                    success_count += 1
                else:
                    # Should not happen if errors return 0.0, but keep for safety
                    error_count += 1
            except Exception as e:
                # Catch errors not handled within calculate_entity_influence_score
                logger.error(
                    f"Unexpected error during batch influence calculation for entity {entity_id}: {e}", exc_info=True)
                error_count += 1

        return {
            "success_count": success_count,
            "error_count": error_count,
            "total_entities": len(target_entity_ids)
        }

    except Exception as e:
        logger.error(
            f"Error in batch influence update setup: {e}", exc_info=True)
        return {
            "success_count": 0,
            "error_count": 0,
            # Report count even if setup failed
            "total_entities": len(target_entity_ids),
            "error": str(e)
        }


def _collect_entity_influence_data(conn, entity_id: int, recency_days: int) -> Optional[Dict[str, Any]]:
    """
    Collect all data needed for influence calculation.

    Args:
        conn: Database connection
        entity_id: Entity ID to collect data for
        recency_days: Number of days to consider for recency calculations

    Returns:
        Dictionary with all raw data needed for influence calculations, or None if entity not found.
    """
    try:
        cursor = conn.cursor()

        # Get basic entity info (using correct columns)
        cursor.execute("""
            SELECT name, entity_type, mentions
            FROM entities
            WHERE id = %s
        """, (entity_id,))
        entity_row = cursor.fetchone()

        if not entity_row:
            # Raise ValueError which will be caught by the calling function
            raise ValueError(f"Entity with ID {entity_id} not found")

        entity_name, entity_type, total_mentions = entity_row

        # 1. Get article count and total mention count from article_entities
        # Note: total_mentions from entities table might be slightly different if not perfectly synced,
        # using article_entities sum is more direct for this calculation.
        cursor.execute("""
            SELECT
                COUNT(DISTINCT article_id) as article_count,
                SUM(mention_count) as mention_sum
            FROM article_entities
            WHERE entity_id = %s
        """, (entity_id,))
        mention_stats = cursor.fetchone()
        article_count = mention_stats[0] if mention_stats else 0
        mention_sum_in_articles = mention_stats[1] if mention_stats and mention_stats[1] is not None else 0

        # 2. Get domain goodness scores for articles mentioning this entity
        cursor.execute("""
            SELECT cdg.domain_goodness_score
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            LEFT JOIN calculated_domain_goodness cdg ON a.domain = cdg.domain
            WHERE ae.entity_id = %s AND cdg.domain_goodness_score IS NOT NULL
        """, (entity_id,))
        # Use 0.5 as default if no score found
        domain_scores = [row[0] for row in cursor.fetchall()] or [0.5]

        # 3. Get cluster hotness for articles mentioning this entity
        cursor.execute("""
            SELECT c.hotness_score
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            JOIN clusters c ON a.cluster_id = c.id
            WHERE ae.entity_id = %s AND c.hotness_score IS NOT NULL AND a.cluster_id IS NOT NULL
        """, (entity_id,))
        # Use 0.0 as default if no score found
        cluster_scores = [row[0] for row in cursor.fetchall()] or [0.0]

        # 4. Get temporal data (mentions over time)
        recent_date_cutoff = datetime.now() - timedelta(days=recency_days)

        cursor.execute("""
            SELECT
                COUNT(DISTINCT article_id) as recent_articles,
                SUM(mention_count) as recent_mentions
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            WHERE ae.entity_id = %s AND a.pub_date >= %s
        """, (entity_id, recent_date_cutoff))
        recent_stats = cursor.fetchone()
        recent_articles = recent_stats[0] if recent_stats else 0
        recent_mentions = recent_stats[1] if recent_stats and recent_stats[1] is not None else 0

        # 5. Get processing tier distribution (REMOVED)
        # tier_distribution = {}

        # 6. Get trend data (comparing current period vs previous period)
        prev_period_start = recent_date_cutoff - timedelta(days=recency_days)

        cursor.execute("""
            SELECT SUM(mention_count) as prev_period_mentions
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            WHERE ae.entity_id = %s AND a.pub_date >= %s AND a.pub_date < %s
        """, (entity_id, prev_period_start, recent_date_cutoff))
        prev_mentions_row = cursor.fetchone()
        prev_mentions = prev_mentions_row[0] if prev_mentions_row and prev_mentions_row[0] is not None else 0

        cursor.close()

        # Compile all data
        return {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'entity_type': entity_type,
            'total_mentions_in_entities': total_mentions,  # From entities table
            'mention_sum_in_articles': mention_sum_in_articles,  # Sum from article_entities
            'article_count': article_count,
            'avg_domain_goodness': sum(domain_scores) / len(domain_scores) if domain_scores else 0.5,
            'avg_cluster_hotness': sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0,
            'recent_mentions': recent_mentions,
            'recent_articles': recent_articles,
            'prev_period_mentions': prev_mentions,
            'recency_days': recency_days
            # 'tier_distribution': tier_distribution, # Removed
            # 'influential_count': influential_count, # Removed
        }
    except Exception as e:
        logger.error(
            f"Error collecting influence data for entity {entity_id}: {e}", exc_info=True)
        if 'cursor' in locals() and cursor:
            cursor.close()
        return None  # Return None on error


def _calculate_base_mention_score(data: Dict[str, Any]) -> float:
    """
    Calculate score based on mention volume and reach (article count).
    Normalizes based on typical ranges (adjust based on data distribution).
    """
    mentions = data.get('mention_sum_in_articles', 0)
    articles = data.get('article_count', 0)

    # Avoid division by zero
    if articles == 0:
        return 0.0

    # Log-transform mentions to reduce impact of extreme outliers
    # Add 1 to avoid log(0)
    log_mentions = math.log10(mentions + 1)
    log_articles = math.log10(articles + 1)

    # Normalize (Example: assumes max log_mentions ~ 5 (100k), max log_articles ~ 4 (10k))
    # These max values should be tuned based on your data
    MAX_LOG_MENTIONS = 5.0
    MAX_LOG_ARTICLES = 4.0

    norm_mentions = min(log_mentions / MAX_LOG_MENTIONS, 1.0)
    norm_articles = min(log_articles / MAX_LOG_ARTICLES, 1.0)

    # Combine volume and reach (e.g., simple average)
    base_score = (norm_mentions + norm_articles) / 2.0
    return min(max(base_score, 0.0), 1.0)  # Clamp between 0 and 1


def _calculate_source_quality_score(data: Dict[str, Any]) -> float:
    """
    Calculate score based on average domain goodness and cluster hotness.
    Assumes goodness/hotness are already roughly 0-1 scales.
    """
    avg_goodness = data.get('avg_domain_goodness', 0.5)  # Default to neutral
    avg_hotness = data.get('avg_cluster_hotness', 0.0)

    # Combine (adjust weighting if needed)
    # Simple average assumes equal importance
    quality_score = (avg_goodness + avg_hotness) / 2.0

    return min(max(quality_score, 0.0), 1.0)  # Clamp between 0 and 1


def _calculate_temporal_score(data: Dict[str, Any]) -> float:
    """
    Calculate score based on recency and trend.
    """
    recent_mentions = data.get('recent_mentions', 0)
    total_mentions = data.get('mention_sum_in_articles', 0)
    prev_mentions = data.get('prev_period_mentions', 0)

    # Recency component: Proportion of mentions that are recent
    recency_factor = 0.0
    if total_mentions > 0:
        recency_factor = recent_mentions / total_mentions

    # Trend component: Growth compared to previous period
    # Use smoothing (add 1) to avoid extreme values with low counts
    trend_factor = 0.0
    if prev_mentions + 1 > 0:
        # Calculate growth ratio
        growth_ratio = (recent_mentions + 1) / (prev_mentions + 1)
        # Map growth ratio to a 0-1 scale (e.g., using tanh or log)
        # Simple mapping: Clamp growth > 1, penalize decline < 1
        if growth_ratio >= 1:
            # Normalize growth (e.g., 1 = no change, 2 = doubled -> maps to 0.5)
            # This mapping needs tuning based on expected growth rates.
            # Example: arctan mapping to keep it bounded
            trend_factor = (math.atan(growth_ratio - 1) /
                            (math.pi / 2)) * 0.5 + 0.5
        else:
            # Penalize decline (e.g., ratio 0.5 -> maps to 0.25)
            trend_factor = growth_ratio * 0.5

    # Combine recency and trend (e.g., weighted average)
    temporal_score = 0.6 * recency_factor + 0.4 * trend_factor

    return min(max(temporal_score, 0.0), 1.0)  # Clamp between 0 and 1


def _get_entity_type(conn, entity_id: int) -> Optional[str]:
    """
    Retrieve the type of a given entity.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT entity_type FROM entities WHERE id = %s", (entity_id,))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting entity type for {entity_id}: {e}")
        return None


def _update_entity_influence_score(conn, entity_id: int, final_score: float) -> bool:
    """
    Update the influence score for an entity in the database.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE entities
            SET influence_score = %s
            WHERE id = %s
            RETURNING id;
        """, (final_score, entity_id))
        success = cursor.fetchone() is not None
        conn.commit()
        cursor.close()
        if not success:
            logger.warning(
                f"Attempted to update influence score for non-existent entity ID: {entity_id}")
        return success
    except Exception as e:
        logger.error(
            f"Error updating influence score for entity {entity_id}: {e}", exc_info=True)
        conn.rollback()
        return False
