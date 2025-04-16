"""
metadata_generation.py - Cluster metadata enhancement

This module provides functions for generating enhanced metadata for clusters,
including extracting keywords, top domains, key entities, date ranges, and
identifying representative articles.

Exported functions:
- generate_and_update_cluster_metadata(cluster_id, article_ids, embeddings_map, 
                                     cluster_centroid, hotness_score, reader_db_client) -> None
  Calculates enhanced metadata for a cluster and updates the database

Related files:
- src/steps/step2/core.py: Calls the metadata enhancement functions
- src/steps/step2/database.py: Provides functions for updating clusters
- src/database/reader_db_client.py: Database client used for operations
- src/localnlp/localnlp_client.py: Used for NLP tasks if available
"""

import logging
import os
import random
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import Counter
from datetime import datetime
import re
import json
import numpy as np
from scipy.spatial.distance import cosine

# Import database client
from src.database.reader_db_client import ReaderDBClient

# Configure logging
logger = logging.getLogger(__name__)

# Common stop words to exclude from keywords
STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
    'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
    'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'by', 'with',
    'at', 'be', 'was', 'were', 'are', 'said', 'says', 'say', 'has', 'have',
    'had', 'not', 'no', 'more', 'most', 'some', 'new', 'also', 'its'
}


def generate_and_update_cluster_metadata(
    cluster_id: int,
    article_ids: List[int],
    embeddings_map: Dict[int, List[float]],
    cluster_centroid: Optional[List[float]],
    reader_db_client: ReaderDBClient
) -> None:
    """
    Calculates enhanced metadata for a cluster and updates the database.

    Args:
        cluster_id: The ID of the cluster to process.
        article_ids: List of article IDs belonging to this cluster.
        embeddings_map: Dictionary mapping article IDs to their embeddings.
        cluster_centroid: The calculated centroid for this cluster.
        reader_db_client: Instance of ReaderDBClient for DB operations.
    """
    if not article_ids:
        logger.warning(f"No articles to process for cluster {cluster_id}")
        return

    try:
        logger.info(
            f"Generating enhanced metadata for cluster {cluster_id} with {len(article_ids)} articles")

        # Sample limit for titles and entities processing
        max_sample_size = min(len(article_ids), int(
            os.getenv("CLUSTER_SAMPLE_SIZE", "50")))

        # Fetch article titles for keyword extraction
        titles = _get_article_titles(
            reader_db_client, article_ids, max_sample_size)

        # Fetch article publication dates for date range
        pub_dates = _get_publication_dates(reader_db_client, article_ids)

        # Fetch article domains
        domains = _get_article_domains(reader_db_client, article_ids)

        # Fetch article entities
        entities = _get_article_entities(reader_db_client, article_ids)

        # Find representative article (closest to centroid)
        representative_article_id = _find_representative_article(
            article_ids, embeddings_map, cluster_centroid)

        # Generate enhanced metadata
        enhanced_metadata = {
            "top_keywords": _extract_keywords_from_titles(titles),
            "date_range": _calculate_date_range(pub_dates),
            "top_domains": _analyze_domains(domains),
            "key_entities": _analyze_entities(entities),
            "representative_article_id": representative_article_id,
            "last_metadata_update": datetime.now().isoformat()
        }

        # Update the cluster with enhanced metadata
        success = reader_db_client.update_cluster_metadata(
            cluster_id, enhanced_metadata)

        if success:
            logger.info(
                f"Successfully updated metadata for cluster {cluster_id}")
        else:
            logger.warning(
                f"Failed to update metadata for cluster {cluster_id}")

    except Exception as e:
        logger.error(
            f"Error generating metadata for cluster {cluster_id}: {e}", exc_info=True)


def _get_article_titles(reader_db_client: ReaderDBClient, article_ids: List[int], max_sample_size: int) -> List[str]:
    """
    Get titles for the specified articles, limited to a maximum sample size.

    Args:
        reader_db_client: Database client
        article_ids: List of article IDs
        max_sample_size: Maximum number of titles to fetch

    Returns:
        List of article titles
    """
    try:
        # Use existing method to get titles
        sample_size = min(len(article_ids), max_sample_size)
        sample_ids = article_ids if sample_size == len(
            article_ids) else random.sample(article_ids, sample_size)
        return reader_db_client.get_sample_titles_for_articles(sample_ids, sample_size)
    except Exception as e:
        logger.error(f"Error fetching article titles: {e}")
        return []


def _get_publication_dates(reader_db_client: ReaderDBClient, article_ids: List[int]) -> Dict[int, Optional[datetime]]:
    """
    Get publication dates for the specified articles.

    Args:
        reader_db_client: Database client
        article_ids: List of article IDs

    Returns:
        Dictionary mapping article IDs to publication dates
    """
    try:
        # Use existing method to get publication dates
        return reader_db_client.get_publication_dates_for_articles(article_ids)
    except Exception as e:
        logger.error(f"Error fetching publication dates: {e}")
        return {}


def _get_article_domains(reader_db_client: ReaderDBClient, article_ids: List[int]) -> Dict[int, str]:
    """
    Get domains for the specified articles.

    Args:
        reader_db_client: Database client
        article_ids: List of article IDs

    Returns:
        Dictionary mapping article IDs to domains
    """
    domains = {}
    conn = None

    try:
        conn = reader_db_client.get_connection()
        if conn:
            cursor = conn.cursor()

            # Create a query with parameter placeholders for the article IDs
            placeholders = ', '.join(['%s'] * len(article_ids))
            query = f"""
                SELECT id, domain
                FROM articles
                WHERE id IN ({placeholders})
            """

            cursor.execute(query, article_ids)

            for row in cursor.fetchall():
                article_id = row[0]
                domain = row[1] if row[1] else "unknown"
                domains[article_id] = domain

            cursor.close()

    except Exception as e:
        logger.error(f"Error fetching article domains: {e}")
    finally:
        if conn:
            reader_db_client.release_connection(conn)

    return domains


def _get_article_entities(reader_db_client: ReaderDBClient, article_ids: List[int]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Get entities for the specified articles.

    Args:
        reader_db_client: Database client
        article_ids: List of article IDs

    Returns:
        Dictionary mapping article IDs to lists of entity information
    """
    entities = {}
    conn = None

    try:
        conn = reader_db_client.get_connection()
        if conn:
            cursor = conn.cursor()

            # Create a query with parameter placeholders for the article IDs
            placeholders = ', '.join(['%s'] * len(article_ids))
            query = f"""
                SELECT ae.article_id, e.id, e.name, e.entity_type, ae.mention_count, e.influence_score
                FROM article_entities ae
                JOIN entities e ON ae.entity_id = e.id
                WHERE ae.article_id IN ({placeholders})
            """

            cursor.execute(query, article_ids)

            for row in cursor.fetchall():
                article_id = row[0]
                entity_info = {
                    "id": row[1],
                    "name": row[2],
                    "type": row[3],
                    "mention_count": row[4],
                    "influence_score": row[5]
                }

                if article_id not in entities:
                    entities[article_id] = []

                entities[article_id].append(entity_info)

            cursor.close()

    except Exception as e:
        logger.error(f"Error fetching article entities: {e}")
    finally:
        if conn:
            reader_db_client.release_connection(conn)

    return entities


def _extract_keywords_from_titles(titles: List[str], max_keywords: int = 10) -> List[str]:
    """
    Extract the most frequent keywords from article titles.

    Args:
        titles: List of article titles
        max_keywords: Maximum number of keywords to return

    Returns:
        List of top keywords
    """
    if not titles:
        return []

    try:
        # Preprocess titles to extract words
        all_words = []
        for title in titles:
            if not title:
                continue

            # Convert to lowercase
            title = title.lower()

            # Remove non-alphanumeric characters and split by whitespace
            words = re.findall(r'\b[a-z0-9][a-z0-9-]*\b', title)

            # Remove stop words and very short words
            words = [
                word for word in words if word not in STOP_WORDS and len(word) > 2]

            all_words.extend(words)

        # Count word frequency
        word_counts = Counter(all_words)

        # Get the most common words
        top_keywords = [word for word,
                        _ in word_counts.most_common(max_keywords)]

        return top_keywords

    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []


def _calculate_date_range(pub_dates: Dict[int, Optional[datetime]]) -> Dict[str, str]:
    """
    Calculate the date range (earliest and latest) from publication dates.

    Args:
        pub_dates: Dictionary mapping article IDs to publication dates

    Returns:
        Dictionary with min_date and max_date as ISO strings
    """
    if not pub_dates:
        return {"min_date": None, "max_date": None}

    try:
        # Filter out None dates
        valid_dates = [date for date in pub_dates.values() if date is not None]

        if not valid_dates:
            return {"min_date": None, "max_date": None}

        min_date = min(valid_dates)
        max_date = max(valid_dates)

        return {
            "min_date": min_date.isoformat() if min_date else None,
            "max_date": max_date.isoformat() if max_date else None
        }

    except Exception as e:
        logger.error(f"Error calculating date range: {e}")
        return {"min_date": None, "max_date": None}


def _analyze_domains(domains: Dict[int, str], max_domains: int = 5) -> List[Dict[str, Any]]:
    """
    Analyze the frequency of domains in the cluster.

    Args:
        domains: Dictionary mapping article IDs to domains
        max_domains: Maximum number of top domains to return

    Returns:
        List of dictionaries with domain information
    """
    if not domains:
        return []

    try:
        # Count domain frequency
        domain_counts = Counter(domains.values())
        total_articles = len(domains)

        # Get the most common domains with count and percentage
        top_domains = []
        for domain, count in domain_counts.most_common(max_domains):
            percentage = (count / total_articles) * \
                100 if total_articles > 0 else 0
            top_domains.append({
                "domain": domain,
                "count": count,
                "percentage": round(percentage, 2)
            })

        return top_domains

    except Exception as e:
        logger.error(f"Error analyzing domains: {e}")
        return []


def _analyze_entities(entities: Dict[int, List[Dict[str, Any]]], max_entities: int = 5) -> List[Dict[str, Any]]:
    """
    Analyze the most prominent entities across all articles in the cluster.

    Args:
        entities: Dictionary mapping article IDs to lists of entity information
        max_entities: Maximum number of top entities to return

    Returns:
        List of dictionaries with entity information
    """
    if not entities:
        return []

    try:
        # Count entity frequency and accumulate influence
        entity_data = {}

        for article_id, article_entities in entities.items():
            for entity in article_entities:
                entity_id = entity["id"]

                if entity_id not in entity_data:
                    entity_data[entity_id] = {
                        "id": entity_id,
                        "name": entity["name"],
                        "type": entity["type"],
                        "article_count": 0,
                        "total_mentions": 0,
                        "influence_score": entity["influence_score"] if entity["influence_score"] else 0
                    }

                entity_data[entity_id]["article_count"] += 1
                entity_data[entity_id]["total_mentions"] += entity["mention_count"] if entity["mention_count"] else 1

        # Sort by article count (primary) and total mentions (secondary)
        sorted_entities = sorted(
            entity_data.values(),
            key=lambda e: (e["article_count"],
                           e["total_mentions"], e["influence_score"]),
            reverse=True
        )

        # Take the top entities
        return sorted_entities[:max_entities]

    except Exception as e:
        logger.error(f"Error analyzing entities: {e}")
        return []


def _find_representative_article(article_ids: List[int], embeddings_map: Dict[int, List[float]],
                                 cluster_centroid: Optional[List[float]]) -> Optional[int]:
    """
    Find the article closest to the cluster centroid (most representative).

    Args:
        article_ids: List of article IDs in the cluster
        embeddings_map: Dictionary mapping article IDs to embeddings
        cluster_centroid: The centroid of the cluster

    Returns:
        ID of the most representative article, or None if not found
    """
    if not cluster_centroid or not article_ids or not embeddings_map:
        return None

    try:
        # Find the article with minimum distance to centroid
        min_distance = float('inf')
        representative_id = None

        for article_id in article_ids:
            if article_id not in embeddings_map:
                continue

            embedding = embeddings_map[article_id]

            # Calculate cosine distance
            distance = cosine(embedding, cluster_centroid)

            if distance < min_distance:
                min_distance = distance
                representative_id = article_id

        return representative_id

    except Exception as e:
        logger.error(f"Error finding representative article: {e}")
        return None
