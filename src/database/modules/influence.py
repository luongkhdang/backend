#!/usr/bin/env python3
"""
influence.py - Entity Influence Score Calculation Module

This module implements advanced influence score calculation for entities by analyzing
mentions across articles, source quality, content context, and temporal factors.
It utilizes weights based on entity types and stores calculation factors for transparency.

Exported functions:
- calculate_entity_influence_score(conn: Any, entity_id: int, recency_days: int) -> float: Calculates a comprehensive influence score.
  - Parameters:
    - conn: Database connection object.
    - entity_id (int): ID of the entity to calculate score for.
    - recency_days (int): Number of days to prioritize for recency (default: 30).
  - Returns (float): Calculated influence score (0.0 if error or entity not found).
- update_all_influence_scores(conn: Any, entity_ids: Optional[List[int]], recency_days: int) -> Dict[str, Any]: Updates influence scores for multiple entities.
  - Parameters:
    - conn: Database connection object.
    - entity_ids (Optional[List[int]]): List of entity IDs to update (if None, uses entities from recent articles).
    - recency_days (int): Number of days to prioritize for recency (default: 30).
  - Returns (Dict[str, Any]): Dictionary with success count, error count, and total entities processed.

Related files:
- src/database/modules/entities.py: Basic entity operations.
- src/database/modules/schema.py: Defines `entity_type_weights` and `entity_influence_factors` tables.
- src/database/reader_db_client.py: Client wrapper for these functions.
- src/steps/step3/__init__.py: Uses these functions for entity processing.
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

# Weights for combining component scores - adjust based on desired factor importance
# These determine the relative contribution of each factor before entity type modification.
COMPONENT_WEIGHTS = {
    'base': 0.40,     # Base mention metrics (reach)
    'quality': 0.20,  # Source quality (domain goodness)
    'temporal': 0.20,  # Recency and trend
    'context': 0.20   # Influential context proportion
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
        # Step 1: Collect raw data (now includes entity_type)
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
        context_score = _calculate_context_score(raw_data)

        # Step 3: Apply weights to component scores
        weighted_score = (
            COMPONENT_WEIGHTS['base'] * base_score +
            COMPONENT_WEIGHTS['quality'] * quality_score +
            COMPONENT_WEIGHTS['temporal'] * temporal_score +
            COMPONENT_WEIGHTS['context'] * context_score
        )

        # Step 4: Apply entity type modifier
        entity_type = raw_data.get('entity_type')
        type_weight = _get_entity_type_weight(conn, entity_type)
        final_score = weighted_score * type_weight

        # Ensure score stays within a reasonable range (e.g., 0 to potentially > 1 if weights allow)
        final_score = max(0, final_score)

        # Step 5: Store calculation factors for transparency
        factors_data = {
            "base_mention_score": base_score,
            "source_quality_score": quality_score,
            "content_context_score": context_score,
            "temporal_score": temporal_score,
            "entity_type_weight_used": type_weight,
            "final_weighted_score_pre_type": weighted_score,
            "final_influence_score": final_score,
            "raw_data": raw_data  # Include the raw collected data
        }
        factors_stored = _store_influence_factors(
            conn, entity_id, factors_data)
        if not factors_stored:
            logger.warning(
                f"Failed to store influence factors for entity {entity_id}.")

        # Step 6: Update the entity's influence score and timestamp
        success = _update_entity_influence_score(
            conn, entity_id, final_score)
        if not success:
            logger.warning(
                f"Failed to update influence score for entity {entity_id} in the database.")
            # Return calculated score despite DB update failure, log warning.

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

        # Get basic entity info (including entity_type)
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
        # Also get sum of mentions where is_influential_context is TRUE
        cursor.execute("""
            SELECT
                COUNT(DISTINCT article_id) as article_count,
                SUM(mention_count) as mention_sum,
                SUM(CASE WHEN is_influential_context THEN mention_count ELSE 0 END) as influential_mention_sum
            FROM article_entities
            WHERE entity_id = %s
        """, (entity_id,))
        mention_stats = cursor.fetchone()
        article_count = mention_stats[0] if mention_stats else 0
        mention_sum_in_articles = mention_stats[1] if mention_stats and mention_stats[1] is not None else 0
        influential_mention_sum = mention_stats[2] if mention_stats and mention_stats[2] is not None else 0

        # 2. Get domain goodness scores for articles mentioning this entity
        cursor.execute("""
            SELECT ds.goodness_score
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            LEFT JOIN domain_statistics ds ON a.domain = ds.domain
            WHERE ae.entity_id = %s AND ds.goodness_score IS NOT NULL
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

        # 4. Get publication dates for recent mentions
        cutoff_date = datetime.now() - timedelta(days=recency_days)
        cursor.execute("""
            SELECT a.pub_date
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            WHERE ae.entity_id = %s AND a.pub_date IS NOT NULL AND a.pub_date >= %s
        """, (entity_id, cutoff_date))
        recent_pub_dates = [row[0] for row in cursor.fetchall()]

        # 5. Get publication dates for all mentions (for trend analysis if needed later)
        cursor.execute("""
            SELECT a.pub_date
            FROM article_entities ae
            JOIN articles a ON ae.article_id = a.id
            WHERE ae.entity_id = %s AND a.pub_date IS NOT NULL
            ORDER BY a.pub_date DESC
        """, (entity_id,))
        all_pub_dates = [row[0] for row in cursor.fetchall()]

        cursor.close()

        return {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "entity_type": entity_type,  # Added entity_type
            "total_global_mentions": total_mentions,
            "linked_article_count": article_count,
            "mention_sum_in_articles": mention_sum_in_articles,
            "influential_mention_sum": influential_mention_sum,
            "domain_scores": domain_scores,
            "cluster_hotness_scores": cluster_scores,
            "recent_pub_dates": recent_pub_dates,
            "all_pub_dates": all_pub_dates
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
    articles = data.get('linked_article_count', 0)

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
    """Calculate score based on average domain goodness of mentioning articles."""
    domain_scores = data.get('domain_scores', [])
    if not domain_scores:
        return 0.5  # Default score if no domains or scores found

    # Simple average for now, could be weighted later
    avg_score = sum(domain_scores) / len(domain_scores)
    # Normalize? Domain scores should ideally be 0-1, but let's clamp just in case.
    return max(0.0, min(1.0, avg_score))


def _calculate_temporal_score(data: Dict[str, Any]) -> float:
    """
    Calculate score based on recency and trend.
    """
    recent_mentions = data.get('recent_pub_dates', [])
    total_mentions = data.get('mention_sum_in_articles', 0)
    prev_mentions = data.get('all_pub_dates', [])

    # Recency component: Proportion of mentions that are recent
    recency_factor = 0.0
    if total_mentions > 0:
        recency_factor = len(recent_mentions) / total_mentions

    # Trend component: Growth compared to previous period
    # Use smoothing (add 1) to avoid extreme values with low counts
    trend_factor = 0.0
    if len(prev_mentions) > 0:
        # Calculate growth ratio
        growth_ratio = (len(recent_mentions) + 1) / (len(prev_mentions) + 1)
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


def _calculate_context_score(data: Dict[str, Any]) -> float:
    """Calculate score based on the proportion of mentions in influential contexts."""
    mention_sum = data.get('mention_sum_in_articles', 0)
    influential_sum = data.get('influential_mention_sum', 0)

    if mention_sum <= 0:
        return 0.0  # No mentions, no influential context score

    proportion = influential_sum / mention_sum
    # Apply logistic function to amplify significance of having *any* influential context
    # k affects steepness, midpoint is 0.5
    k = 5
    midpoint = 0.2  # Lower midpoint: even a small proportion is significant
    score = 1 / (1 + math.exp(-k * (proportion - midpoint)))

    return max(0.0, min(1.0, score))


def _get_entity_type_weight(conn, entity_type: Optional[str]) -> float:
    """
    Fetch the weight multiplier for a given entity type from the database.
    Defaults to 1.0 if type is None or not found.
    """
    if not entity_type:
        return 1.0

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT weight FROM entity_type_weights WHERE entity_type = %s
        """, (entity_type,))
        result = cursor.fetchone()
        cursor.close()
        if result:
            return float(result[0])
        else:
            logger.warning(
                f"No weight found for entity type '{entity_type}'. Defaulting to 1.0.")
            return 1.0
    except Exception as e:
        logger.error(
            f"Error fetching weight for entity type '{entity_type}': {e}", exc_info=True)
        # Default to 1.0 on error to avoid calculation failure
        return 1.0


def _store_influence_factors(conn, entity_id: int, factors_data: Dict[str, Any]) -> bool:
    """
    Store the calculated influence factors and raw data in the entity_influence_factors table.
    """
    try:
        cursor = conn.cursor()
        # Serialize the raw data part to JSONB
        raw_data_json = json.dumps(factors_data.get("raw_data", {}))

        cursor.execute("""
            INSERT INTO entity_influence_factors (
                entity_id, calculation_timestamp,
                base_mention_score, source_quality_score, content_context_score, temporal_score,
                raw_data
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entity_id, calculation_timestamp) DO UPDATE SET
                base_mention_score = EXCLUDED.base_mention_score,
                source_quality_score = EXCLUDED.source_quality_score,
                content_context_score = EXCLUDED.content_context_score,
                temporal_score = EXCLUDED.temporal_score,
                raw_data = EXCLUDED.raw_data;
        """, (
            entity_id, datetime.now(),  # Use current timestamp for calculation_timestamp
            factors_data.get("base_mention_score"),
            factors_data.get("source_quality_score"),
            factors_data.get("content_context_score"),
            factors_data.get("temporal_score"),
            raw_data_json
        ))
        conn.commit()
        cursor.close()
        logger.debug(
            f"Successfully stored influence factors for entity {entity_id}")
        return True
    except Exception as e:
        conn.rollback()  # Rollback on error
        logger.error(
            f"Error storing influence factors for entity {entity_id}: {e}", exc_info=True)
        if 'cursor' in locals() and cursor:
            cursor.close()
        return False


def _update_entity_influence_score(conn, entity_id: int, final_score: float) -> bool:
    """Update the influence score and timestamp in the entities table."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE entities
            SET influence_score = %s,
                influence_calculated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (final_score, entity_id))
        updated_rows = cursor.rowcount
        conn.commit()
        cursor.close()
        return updated_rows > 0
    except Exception as e:
        conn.rollback()  # Rollback on error
        logger.error(
            f"Error updating final influence score for entity {entity_id}: {e}", exc_info=True)
        if 'cursor' in locals() and cursor:
            cursor.close()
        return False
