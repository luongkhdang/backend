"""
hotness.py - Cluster hotness calculation

This module handles the calculation of which clusters should be marked as "hot"
based on entity influence, topic relevance, and article recency.

Exported functions:
- calculate_hotness_factors(article_data: List[Tuple[int, np.ndarray]], labels: np.ndarray, 
                          pub_dates: List[Optional[datetime]], entity_weights: Optional[Dict[str, float]]) -> Dict[int, bool]
  Determines which clusters should be marked as hot based on multiple factors
- get_hotness_threshold() -> float: Returns the configured hotness threshold from environment or default
- get_weight_params() -> Dict[str, float]: Returns the weight parameters for hotness calculation

Related files:
- src/steps/step2/core.py: Uses these functions for hotness determination
- src/steps/step2/database.py: Uses hotness data when storing clusters
"""

import logging
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
import json

# Configure logging
logger = logging.getLogger(__name__)


def calculate_hotness_factors(
    article_data: List[Tuple[int, np.ndarray]],
    labels: np.ndarray,
    pub_dates: List[Optional[datetime]],
    entity_weights: Optional[Dict[str, float]] = None
) -> Dict[int, bool]:
    """
    Calculate which clusters should be marked as hot based on multiple factors:
    1. Recency of articles in the cluster
    2. Entity influence (using provided entity weights)
    3. Number of articles in the cluster

    Args:
        article_data: List of tuples (article_id, embedding)
        labels: HDBSCAN cluster labels array
        pub_dates: List of article publication dates (same order as article_data)
        entity_weights: Optional dictionary mapping entity IDs to their importance scores

    Returns:
        Dictionary mapping cluster labels to boolean hot status
    """
    # Skip noise points (-1)
    unique_labels = [l for l in set(labels) if l >= 0]

    # Default values from environment or hardcoded
    min_cluster_size = int(os.getenv("MIN_HOT_CLUSTER_SIZE", "5"))
    recency_days = int(os.getenv("HOT_RECENCY_DAYS", "14"))
    hotness_threshold = get_hotness_threshold()

    # Calculate recency cutoff date
    recency_cutoff = datetime.now() - timedelta(days=recency_days)

    # Initialize results dictionary and cluster data
    hotness_map = {}
    cluster_scores = {}
    cluster_stats = {}

    # Group articles by cluster
    cluster_articles = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:  # Skip noise points
            article_id = article_data[i][0]
            pub_date = pub_dates[i]
            cluster_articles[label].append((article_id, pub_date))

    # Get weight parameters for hotness calculation
    weight_params = get_weight_params()
    recency_weight = weight_params["recency"]
    influence_weight = weight_params["influence"]
    size_weight = weight_params["size"]

    # Calculate hotness scores for each cluster
    for label in unique_labels:
        # Skip clusters that are too small
        if len(cluster_articles[label]) < min_cluster_size:
            hotness_map[label] = False
            continue

        # Calculate recency score (% of articles published within recency window)
        recent_count = 0
        for _, pub_date in cluster_articles[label]:
            if pub_date and pub_date >= recency_cutoff:
                recent_count += 1

        recency_score = recent_count / \
            len(cluster_articles[label]) if cluster_articles[label] else 0

        # Size factor (larger clusters get a small bonus)
        # Caps at 50 articles
        size_factor = min(1.0, len(cluster_articles[label]) / 50)

        # Entity influence score
        influence_score = 0.0
        if entity_weights:
            # For MVP, this can be simplified by assuming entity_weights
            # carries article_id -> score mappings
            # In future iterations, this would analyze entity presence in articles
            article_ids = [aid for aid, _ in cluster_articles[label]]
            matched_weights = [entity_weights.get(
                str(aid), 0.0) for aid in article_ids]
            influence_score = sum(matched_weights) / \
                len(article_ids) if article_ids else 0

        # Calculate final hotness score with weighted components
        hotness_score = (recency_weight * recency_score) + \
            (influence_weight * influence_score) + \
            (size_weight * size_factor)

        # Store scores and mark as hot if above threshold
        cluster_scores[label] = hotness_score
        hotness_map[label] = hotness_score >= hotness_threshold

        # Store statistics for logging
        cluster_stats[label] = {
            "size": len(cluster_articles[label]),
            "recent_articles": recent_count,
            "recency_score": recency_score,
            "influence_score": influence_score,
            "size_factor": size_factor,
            "hotness_score": hotness_score,
            "is_hot": hotness_map[label]
        }

    # Log statistics for top clusters
    top_clusters = sorted(
        [(label, stats["hotness_score"])
         for label, stats in cluster_stats.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    log_data = {
        "total_clusters": len(unique_labels),
        "hot_clusters": sum(1 for is_hot in hotness_map.values() if is_hot),
        "top_clusters": {label: cluster_stats[label] for label, _ in top_clusters}
    }

    logger.info(
        f"Hotness calculation complete: {json.dumps(log_data, default=str)}")

    return hotness_map


def get_hotness_threshold() -> float:
    """
    Get the configured hotness threshold from environment or default value.

    Returns:
        float: Hotness threshold score required for a cluster to be marked as "hot"
    """
    return float(os.getenv("HOTNESS_SCORE_THRESHOLD", "0.7"))


def get_weight_params() -> Dict[str, float]:
    """
    Get weight parameters for hotness calculation from environment or defaults.

    Returns:
        Dict[str, float]: Dictionary with weight parameters for different factors
    """
    return {
        "recency": float(os.getenv("HOTNESS_RECENCY_WEIGHT", "0.5")),
        "influence": float(os.getenv("HOTNESS_INFLUENCE_WEIGHT", "0.3")),
        "size": float(os.getenv("HOTNESS_SIZE_WEIGHT", "0.2"))
    }
