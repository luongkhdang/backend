"""
hotness.py - Cluster hotness calculation

This module handles the calculation of which clusters should be marked as "hot"
based on a weighted score including size, recency, entity influence, and topic relevance.
It uses a Top N selection approach to mark the highest scoring clusters as hot.

Exported functions:
- calculate_hotness_factors(cluster_data: Dict[int, List[int]], article_ids: List[int], 
                           pub_dates: List[Optional[datetime]], reader_client: ReaderDBClient, 
                           nlp: Optional[Any] = None) -> Tuple[Dict[int, bool], Dict[int, float]]
  Determines which clusters should be marked as hot based on multiple weighted factors and Top N selection
- get_weight_params() -> Dict[str, float]: Returns the weight parameters for hotness calculation

Related files:
- src/steps/step2/core.py: Uses these functions for hotness determination
- src/steps/step2/database.py: Uses hotness data when storing clusters
- src/steps/step2/interpretation.py: Used for keyword extraction for topic relevance
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import json
import math

# Handle numpy import gracefully
try:
    import numpy as np
except ImportError:
    logging.warning("numpy not available; some functionality may be limited")
    np = None  # Not needed in this module's core functionality

from src.database.reader_db_client import ReaderDBClient
from src.steps.step2.interpretation import get_cluster_keywords
from .config import load_step2_config

# Configure logging
logger = logging.getLogger(__name__)


def calculate_hotness_factors(
    cluster_data: Dict[int, List[int]],
    article_ids: List[int],
    pub_dates: List[Optional[datetime]],
    reader_client: ReaderDBClient,
    nlp: Optional[Any] = None
) -> Tuple[Dict[int, bool], Dict[int, float]]:
    """
    Calculate which clusters should be marked as hot based on a weighted score of four factors
    and a Top N selection approach:
    1. Size Factor: Logarithmic count, normalized
    2. Recency Factor: Proportion of recent articles, normalized
    3. Entity Influence Factor: Average influence score of linked entities, normalized
    4. Topic Relevance Factor: Binary score based on matching keywords against predefined core topics

    Instead of using a threshold, this function selects the top N clusters by score to mark as hot.

    Args:
        cluster_data: Dictionary mapping cluster labels to indices of articles in that cluster
        article_ids: List of article database IDs (same order as indices in cluster_data)
        pub_dates: List of article publication dates (same order as article_ids)
        reader_client: Database client for retrieving entity influence and article titles
        nlp: Optional spaCy model for keyword extraction (if topic relevance is enabled)

    Returns:
        Tuple containing:
        - Dict[int, bool]: Mapping cluster labels to boolean hot status
        - Dict[int, float]: Mapping cluster labels to numerical hotness scores
    """
    # Get core topic keywords from environment or use default list
    default_core_topics = (
        "china,us,united states,vietnam,europe,germany,war,trade,exports,tariffs,"
        "geopolitics,geopolitical,political economy,influence,lobbying,narrative,framing,"
        "disinformation,misinformation,ai,artificial intelligence,election,campaign,"
        "pentagon,defense,state department,diplomacy,itc,international trade commission"
    )
    core_topic_keywords_str = os.getenv(
        "CORE_TOPIC_KEYWORDS", default_core_topics)
    core_topic_keywords = {kw.strip().lower()
                           for kw in core_topic_keywords_str.split(',')}

    # Get configuration parameters from environment
    recency_days = int(os.getenv("RECENCY_DAYS", "3"))
    cluster_sample_size = int(os.getenv("CLUSTER_SAMPLE_SIZE", "10"))
    calculate_topic_relevance = os.getenv(
        "CALCULATE_TOPIC_RELEVANCE", "true").lower() == "true"

    # Get the target number of hot clusters from environment
    target_hot_clusters = int(os.getenv("TARGET_HOT_CLUSTERS", "15"))

    # Calculate recency cutoff date
    recency_cutoff = datetime.now() - timedelta(days=recency_days)

    # Dictionary to store raw and normalized factor scores
    raw_scores = {
        "size": {},
        "recency": {},
        "influence": {},
        "relevance": {}
    }

    # Dictionary to map cluster labels to DB article IDs
    cluster_article_ids = {}

    # Process each cluster to calculate raw scores
    for label, indices in cluster_data.items():
        if label < 0:  # Skip noise points
            continue

        # Get DB article IDs for this cluster
        article_db_ids = [article_ids[i] for i in indices]
        cluster_article_ids[label] = article_db_ids

        # 1. Size Factor (Raw): Calculate log(count + 1)
        article_count = len(article_db_ids)
        size_score = math.log(article_count + 1)
        raw_scores["size"][label] = size_score

        # 2. Recency Factor (Raw): Calculate proportion of recent articles
        recent_count = 0
        valid_pub_dates = 0
        for i in indices:
            pub_date = pub_dates[i]
            if pub_date:
                valid_pub_dates += 1
                if pub_date >= recency_cutoff:
                    recent_count += 1

        recency_score = recent_count / valid_pub_dates if valid_pub_dates > 0 else 0
        raw_scores["recency"][label] = recency_score

        # 3. Entity Influence Factor (Raw): Calculate average influence score
        influence_scores = reader_client.get_entity_influence_for_articles(
            article_db_ids)

        # Filter out None values before calculating the average
        valid_scores = [
            score for score in influence_scores.values() if score is not None]
        avg_influence = sum(valid_scores) / \
            len(valid_scores) if valid_scores else 0

        raw_scores["influence"][label] = avg_influence

        # 4. Topic Relevance Factor (Raw): Binary score based on keyword matching
        relevance_score = 0
        if calculate_topic_relevance and nlp:
            # Extract keywords from article titles
            keywords = get_cluster_keywords(
                reader_client, article_db_ids, nlp, cluster_sample_size)

            # Check if any keyword matches core topics
            for keyword in keywords:
                if any(core_topic in keyword or keyword in core_topic for core_topic in core_topic_keywords):
                    relevance_score = 1  # Binary score: either 0 or 1
                    break

        raw_scores["relevance"][label] = relevance_score

    # Normalize scores using min-max scaling
    norm_scores = {
        "size": {},
        "recency": {},
        "influence": {},
        # No need to normalize relevance as it's already 0 or 1
        "relevance": raw_scores["relevance"]
    }

    # Function to normalize scores using min-max scaling
    def normalize_scores(scores):
        if not scores:
            return {}
        min_val = min(scores.values())
        max_val = max(scores.values())
        range_val = max_val - min_val

        if range_val == 0:  # All scores are the same
            return {k: 1.0 if v > 0 else 0.0 for k, v in scores.items()}

        return {k: (v - min_val) / range_val for k, v in scores.items()}

    # Normalize size, recency, and influence scores
    for factor in ["size", "recency", "influence"]:
        if raw_scores[factor]:  # Only normalize if we have scores
            norm_scores[factor] = normalize_scores(raw_scores[factor])

    # Get weight parameters
    weight_params = get_weight_params()
    w_size = weight_params["size"]
    w_recency = weight_params["recency"]
    w_influence = weight_params["influence"]
    w_relevance = weight_params["relevance"]

    # Calculate final hotness scores for all clusters
    hotness_scores = {}

    for label in cluster_data.keys():
        if label < 0:  # Skip noise points
            continue

        # Calculate weighted hotness score
        final_score = (
            w_size * norm_scores["size"].get(label, 0) +
            w_recency * norm_scores["recency"].get(label, 0) +
            w_influence * norm_scores["influence"].get(label, 0) +
            w_relevance * norm_scores["relevance"].get(label, 0)
        )

        hotness_scores[label] = final_score

    # Rank clusters by hotness score and select top N
    ranked_clusters = sorted(
        [(label, score) for label, score in hotness_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Create hotness map (Top N are hot, rest are not)
    hotness_map = {}
    for i, (label, _) in enumerate(ranked_clusters):
        hotness_map[label] = i < target_hot_clusters

    # For clusters not in the ranking (should not happen in normal operation)
    for label in cluster_data.keys():
        if label >= 0 and label not in hotness_map:
            hotness_map[label] = False

    # Get the actual number of hot clusters (may be less than target if not enough clusters)
    hot_clusters = sum(1 for is_hot in hotness_map.values() if is_hot)

    # Enhanced logging with detailed score information
    # Log statistics for up to 20 clusters or all if fewer
    log_clusters = min(20, len(ranked_clusters))

    # Convert numpy types to native Python types for JSON serialization
    log_data = {
        "total_clusters": int(len(cluster_data)),
        "hot_clusters": int(hot_clusters),
        "target_hot_clusters": int(target_hot_clusters),
        "weight_params": {k: float(v) for k, v in weight_params.items()},
        "factor_statistics": {
            "size": {
                "min": float(min(raw_scores["size"].values())) if raw_scores["size"] else 0,
                "max": float(max(raw_scores["size"].values())) if raw_scores["size"] else 0,
                "avg": float(sum(raw_scores["size"].values()) / len(raw_scores["size"])) if raw_scores["size"] else 0
            },
            "recency": {
                "min": float(min(raw_scores["recency"].values())) if raw_scores["recency"] else 0,
                "max": float(max(raw_scores["recency"].values())) if raw_scores["recency"] else 0,
                "avg": float(sum(raw_scores["recency"].values()) / len(raw_scores["recency"])) if raw_scores["recency"] else 0
            },
            "influence": {
                "min": float(min(raw_scores["influence"].values())) if raw_scores["influence"] else 0,
                "max": float(max(raw_scores["influence"].values())) if raw_scores["influence"] else 0,
                "avg": float(sum(raw_scores["influence"].values()) / len(raw_scores["influence"])) if raw_scores["influence"] else 0
            },
            "relevance": {
                "count_relevant": int(sum(1 for v in raw_scores["relevance"].values() if v > 0)),
                "percent_relevant": float(sum(1 for v in raw_scores["relevance"].values() if v > 0) * 100 / len(raw_scores["relevance"])) if raw_scores["relevance"] else 0
            }
        },
        "top_clusters": {
            int(label): {
                "rank": i+1,
                "score": float(score),
                "is_hot": bool(hotness_map[label]),
                "article_count": int(len(cluster_data[label])),
                "factors": {
                    "size": {
                        "raw": float(raw_scores["size"].get(label, 0)),
                        "normalized": float(norm_scores["size"].get(label, 0)),
                        "weighted": float(w_size * norm_scores["size"].get(label, 0))
                    },
                    "recency": {
                        "raw": float(raw_scores["recency"].get(label, 0)),
                        "normalized": float(norm_scores["recency"].get(label, 0)),
                        "weighted": float(w_recency * norm_scores["recency"].get(label, 0))
                    },
                    "influence": {
                        "raw": float(raw_scores["influence"].get(label, 0)),
                        "normalized": float(norm_scores["influence"].get(label, 0)),
                        "weighted": float(w_influence * norm_scores["influence"].get(label, 0))
                    },
                    "relevance": {
                        "raw": float(raw_scores["relevance"].get(label, 0)),
                        "weighted": float(w_relevance * raw_scores["relevance"].get(label, 0))
                    }
                }
            } for i, (label, score) in enumerate(ranked_clusters[:log_clusters])
        }
    }

    logger.info(
        f"Hotness calculation complete: {json.dumps(log_data, default=str)}")

    return hotness_map, hotness_scores


def get_weight_params() -> Dict[str, float]:
    """
    Get weight parameters for hotness calculation from environment or defaults.

    Returns:
        Dict[str, float]: Dictionary with weight parameters for different factors
    """
    return {
        "size": float(os.getenv("W_SIZE", "0.15")),
        "recency": float(os.getenv("W_RECENCY", "0.30")),
        "influence": float(os.getenv("W_INFLUENCE", "0.30")),
        "relevance": float(os.getenv("W_RELEVANCE", "0.25"))
    }


def calculate_hotness_scores(db_client, cluster_article_map, nlp=None) -> Dict[int, float]:
    """
    Calculates hotness scores for clusters based on size, recency, entity influence, and topic relevance.

    Args:
        db_client: Instance of ReaderDBClient.
        cluster_article_map: Dictionary mapping cluster_id to list of article_ids.
        nlp: Optional loaded spaCy model for keyword extraction (topic relevance).

    Returns:
        Dictionary mapping cluster_id to hotness score.
    """
    if not cluster_article_map:
        return {}

    cfg = load_step2_config()
    all_article_ids = [aid for articles in cluster_article_map.values()
                       for aid in articles]

    logger.info(
        f"Calculating hotness for {len(cluster_article_map)} clusters using {len(all_article_ids)} articles.")

    # Fetch necessary data in bulk
    logger.debug("Fetching publication dates...")
    pub_dates = db_client.get_publication_dates_for_articles(all_article_ids)
    logger.debug("Fetching entity influence scores...")
    influence_scores = db_client.get_entity_influence_for_articles(
        all_article_ids)

    # Topic Relevance Calculation (now runs unconditionally if nlp model provided)
    topic_relevance_scores = defaultdict(float)  # Default to 0
    if nlp:
        logger.debug("Calculating topic relevance...")
        try:
            topic_relevance_scores = calculate_article_topic_relevance(
                db_client, all_article_ids, nlp, cfg.core_topic_keywords
            )
        except Exception as e:
            logger.error(
                f"Error calculating topic relevance: {e}", exc_info=True)
            # Continue without relevance scores if calculation fails
    else:
        logger.info(
            "Skipping topic relevance calculation (spaCy model not available).")

    hotness_scores = {}
    current_time = datetime.now(timezone.utc)
    max_articles_in_cluster = max(len(
        articles) for articles in cluster_article_map.values()) if cluster_article_map else 1

    # Precompute factors if possible (e.g., max influence score)
    max_influence = max(influence_scores.values()) if influence_scores else 1.0
    max_relevance = max(topic_relevance_scores.values()
                        ) if topic_relevance_scores else 1.0

    for cluster_id, article_ids in cluster_article_map.items():
        if cluster_id == -1 or not article_ids:  # Skip noise cluster or empty clusters
            continue

        cluster_size = len(article_ids)

        # --- Calculate individual components ---

        # Size Score (normalized)
        size_score = cluster_size / \
            max_articles_in_cluster if max_articles_in_cluster > 0 else 0

        # Recency Score (average recency)
        total_recency_score = 0
        valid_recency_count = 0
        for article_id in article_ids:
            pub_date = pub_dates.get(article_id)
            if pub_date:
                # Ensure pub_date is offset-aware for comparison
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(
                        tzinfo=timezone.utc)  # Assume UTC if naive
                days_old = (current_time - pub_date).days
                # Simple decay: score = 1 / (days_old + 1), capped by cfg.recency_days
                recency_factor = 1.0 / (min(days_old, cfg.recency_days) + 1)
                total_recency_score += recency_factor
                valid_recency_count += 1
        recency_score = total_recency_score / \
            valid_recency_count if valid_recency_count > 0 else 0

        # Influence Score (average normalized influence)
        total_influence_score = 0
        for article_id in article_ids:
            score = influence_scores.get(article_id, 0)
            # Normalize influence: score / max_influence (handle max_influence=0)
            total_influence_score += score / max_influence if max_influence > 0 else 0
        influence_score = total_influence_score / \
            cluster_size if cluster_size > 0 else 0

        # Topic Relevance Score (average normalized relevance)
        total_relevance_score = 0
        # Only calculate if nlp was available and scores were generated
        if topic_relevance_scores:
            for article_id in article_ids:
                score = topic_relevance_scores.get(article_id, 0)
                # Normalize relevance: score / max_relevance (handle max_relevance=0)
                total_relevance_score += score / max_relevance if max_relevance > 0 else 0
            relevance_score = total_relevance_score / \
                cluster_size if cluster_size > 0 else 0
        else:
            relevance_score = 0  # Set to 0 if relevance couldn't be calculated

        # Weighted Sum
        hotness = (
            (cfg.w_size * size_score) +
            (cfg.w_recency * recency_score) +
            (cfg.w_influence * influence_score) +
            (cfg.w_relevance * relevance_score)
        )

        hotness_scores[cluster_id] = hotness
        logger.debug(
            f"Cluster {cluster_id}: Size={size_score:.3f}, Recency={recency_score:.3f}, Influence={influence_score:.3f}, Relevance={relevance_score:.3f} -> Hotness={hotness:.3f}")

    return hotness_scores


def calculate_article_topic_relevance(db_client, article_ids, nlp, core_keywords) -> Dict[int, float]:
    """
    Calculates topic relevance score for articles based on keyword overlap.

    Args:
        db_client: Instance of ReaderDBClient.
        article_ids: List of article IDs.
        nlp: Loaded spaCy model for keyword extraction.
        core_keywords: Set of core topic keywords.

    Returns:
        Dictionary mapping article ID to topic relevance score.
    """
    topic_relevance_scores = {}
    for article_id in article_ids:
        relevance_score = 0
        if nlp:
            # Extract keywords from article titles
            keywords = get_cluster_keywords(
                db_client, [article_id], nlp, 10)

            # Check if any keyword matches core topics
            for keyword in keywords:
                if any(core_topic in keyword or keyword in core_topic for core_topic in core_keywords):
                    relevance_score = 1  # Binary score: either 0 or 1
                    break

        topic_relevance_scores[article_id] = relevance_score
    return topic_relevance_scores
