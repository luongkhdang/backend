"""
interpretation.py - Cluster interpretation and keyword extraction

This module handles the extraction of keywords and topics from clusters,
providing meaningful interpretations for clusters of articles.

Exported functions:
- extract_cluster_keywords(db_client: Any, article_ids: List[int], model: Optional[Any] = None) -> Dict[str, float]
  Extracts keywords from article texts for a given cluster
- interpret_clusters(db_client: Any, cluster_article_map: Dict[int, List[int]], model: Optional[Any] = None) -> Dict[int, Dict[str, Any]]
  Generates full interpretations for multiple clusters
- get_cluster_keywords(db_client: Any, cluster_id: int, model: Optional[Any] = None) -> Dict[str, float]
  Gets keywords for a specific cluster by ID
- interpret_cluster(reader_client: ReaderDBClient, cluster_id: int, nlp: Any) -> None
  Updates metadata for a specific cluster with interpretation data

Related files:
- src/steps/step2/core.py: Uses these functions for cluster interpretation
- src/steps/step2/clustering.py: Provides clustering results for interpretation
- src/database/reader_db_client.py: Used for database operations
"""

import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import re
import json

# Handle optional imports gracefully
try:
    import numpy as np
except ImportError:
    logging.warning("numpy not available; some functionality may be limited")
    # Create a minimal substitute for np.mean if needed

    class NumpySubstitute:
        @staticmethod
        def mean(arr, axis=0):
            if axis == 0 and isinstance(arr, list) and all(isinstance(x, list) for x in arr):
                # Simple column-wise mean for 2D arrays
                result = [sum(col)/len(col) for col in zip(*arr)]
                return result
            # Simple mean for 1D arrays
            return sum(arr) / len(arr)
    np = NumpySubstitute()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    logging.warning(
        "sklearn not available; keyword extraction will be limited")
    TfidfVectorizer = None

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logging.warning("spaCy not available; keyword extraction will be limited")
    spacy = None
    SPACY_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    logging.warning("tqdm not available; using simple progress reporting")
    # Simple substitute for tqdm

    def tqdm(iterable, **kwargs):
        return iterable

try:
    from psycopg2.extras import RealDictCursor
except ImportError:
    RealDictCursor = None
    logger = logging.getLogger(__name__)
    logger.warning("psycopg2.extras not available. Cannot use RealDictCursor.")

# Configure logging
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.

    Args:
        text: Raw text string to clean

    Returns:
        Cleaned text string
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove special characters but keep alphanumeric, spaces, and some punctuation
    text = re.sub(r'[^\w\s\.\,\?\!]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_cluster_keywords(
    db_client: Any,
    article_ids: List[int],
    model: Optional[Any] = None,
    top_n: int = 15
) -> Dict[str, float]:
    """
    Extract keywords from articles in a cluster using TF-IDF and named entity recognition.

    Args:
        db_client: Database client to fetch article texts
        article_ids: List of article IDs in the cluster
        model: Optional spaCy model for named entity recognition
        top_n: Number of top keywords to return

    Returns:
        Dictionary mapping keywords to their importance scores
    """
    if not article_ids:
        return {}

    # Load spaCy model if not provided
    if model is None:
        try:
            model = spacy.load("en_core_web_lg")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            model = None

    # Fetch article texts from database
    article_texts = []
    for article_id in article_ids:
        try:
            # Query should be adjusted based on actual DB schema
            query = "SELECT content FROM articles WHERE id = %s"
            result = db_client.fetch_one(query, (article_id,))
            if result and result.get('content'):
                article_texts.append(clean_text(result['content']))
        except Exception as e:
            logger.error(f"Error fetching article {article_id}: {e}")

    if not article_texts:
        return {}

    # Combine TF-IDF with entity recognition for better keyword extraction
    keywords = {}

    # 1. TF-IDF for term importance
    try:
        tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        tfidf_matrix = tfidf.fit_transform(article_texts)
        feature_names = tfidf.get_feature_names_out()

        # Get average TF-IDF scores for each term
        avg_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

        # Add top TF-IDF terms
        for term, score in sorted(
            zip(feature_names, avg_tfidf_scores),
            key=lambda x: x[1],
            reverse=True
        )[:50]:  # Consider more terms initially
            keywords[term] = float(score)

    except Exception as e:
        logger.error(f"Error in TF-IDF extraction: {e}")

    # 2. Named Entity Recognition for important entities
    if model:
        all_entities = []
        # Limit processing for efficiency
        for text in article_texts[:min(len(article_texts), 20)]:
            try:
                # Limit text length for processing efficiency
                doc = model(text[:10000])
                for ent in doc.ents:
                    if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT", "EVENT"]:
                        all_entities.append(ent.text.lower())
            except Exception as e:
                logger.error(f"Error in entity extraction: {e}")

        # Count entity frequencies
        entity_counter = Counter(all_entities)

        # Add entity scores (normalized by document count)
        doc_count = len(article_texts)
        for entity, count in entity_counter.most_common(30):
            normalized_score = min(
                1.0, count / (doc_count * 0.5))  # Cap at 1.0
            if entity in keywords:
                # Boost existing keyword if it's also an entity
                keywords[entity] *= (1.0 + normalized_score)
            else:
                keywords[entity] = normalized_score

    # Return top keywords
    return {
        k: float(v)
        for k, v in sorted(
            keywords.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
    }


def interpret_clusters(
    db_client: Any,
    cluster_article_map: Dict[int, List[int]],
    model: Optional[Any] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Generate interpretations for multiple clusters.

    Args:
        db_client: Database client for fetching article data
        cluster_article_map: Mapping of cluster IDs to lists of article IDs
        model: Optional spaCy model for NLP processing

    Returns:
        Dictionary mapping cluster IDs to their interpretation data
    """
    # Load spaCy model once if not provided
    if not model and any(cluster_article_map.values()):
        try:
            model = spacy.load("en_core_web_lg")
            logger.info("Loaded spaCy model for entity recognition")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            model = None

    interpretations = {}

    # Process each cluster
    for cluster_id, article_ids in tqdm(cluster_article_map.items(), desc="Interpreting clusters"):
        if not article_ids:
            continue

        # Extract keywords for the cluster
        keywords = extract_cluster_keywords(db_client, article_ids, model)

        # Get publication timeline data
        pub_dates = []
        try:
            query = "SELECT published_at FROM articles WHERE id IN %s"
            results = db_client.fetch_all(query, (tuple(article_ids),))
            for result in results:
                if result and result.get('published_at'):
                    pub_dates.append(result['published_at'])
        except Exception as e:
            logger.error(
                f"Error fetching publication dates for cluster {cluster_id}: {e}")

        # Store cluster interpretation
        interpretations[cluster_id] = {
            "cluster_id": cluster_id,
            "article_count": len(article_ids),
            "keywords": keywords,
            "date_range": {
                "earliest": min(pub_dates) if pub_dates else None,
                "latest": max(pub_dates) if pub_dates else None,
                "count": len(pub_dates)
            }
        }

    return interpretations


def get_cluster_keywords(
    reader_client: Any,
    article_db_ids: List[int],
    nlp: Any,
    sample_size: int = 10
) -> List[str]:
    """
    Extract keywords from article titles for a specific cluster.
    This function is focused on extracting only the most relevant keywords
    for topic relevance calculation.

    Args:
        reader_client: Database client to fetch article titles
        article_db_ids: List of article IDs in the cluster
        nlp: Loaded spaCy model for NLP processing
        sample_size: Maximum number of titles to analyze

    Returns:
        List[str]: List of extracted keywords as strings
    """
    if not article_db_ids or not nlp:
        return []

    # Get sample of article titles
    titles = reader_client.get_sample_titles_for_articles(
        article_db_ids, sample_size)
    if not titles:
        return []

    # Combine titles into a single text for processing
    combined_text = " ".join(titles)

    # Process text with spaCy
    doc = nlp(combined_text)

    # Extract noun chunks as they often represent topics better than single entities
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

    # Count frequency of noun chunks
    from collections import Counter
    chunk_counter = Counter(noun_chunks)

    # Get the top keywords (most frequent noun chunks)
    top_n = 5  # Return top 5 keywords
    top_keywords = [keyword for keyword, _ in chunk_counter.most_common(top_n)]

    # If we don't have enough noun chunks, supplement with named entities
    if len(top_keywords) < top_n:
        entities = [ent.text.lower() for ent in doc.ents
                    if ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT"]]
        entity_counter = Counter(entities)

        # Add top entities not already in keywords
        for entity, _ in entity_counter.most_common(top_n - len(top_keywords)):
            if entity not in top_keywords:
                top_keywords.append(entity)

    return top_keywords


def interpret_cluster(reader_client: Any, cluster_id: int, nlp: Any) -> None:
    """
    Perform basic interpretation of a cluster using spaCy.

    Args:
        reader_client: Initialized ReaderDBClient
        cluster_id: Database ID of the cluster
        nlp: Loaded spaCy model
    """
    conn = None
    try:
        conn = reader_client.get_connection()
        if not conn:
            logger.error("Could not get DB connection for interpretation")
            return

        # Use RealDictCursor to ensure results are dictionaries
        cursor_factory = RealDictCursor if RealDictCursor else None
        cursor = conn.cursor(cursor_factory=cursor_factory)

        sample_size = int(os.getenv("CLUSTER_SAMPLE_SIZE", "10"))
        query = """
        SELECT id, title, content
        FROM articles
        WHERE cluster_id = %s
        ORDER BY pub_date DESC
        LIMIT %s
        """

        cursor.execute(query, (cluster_id, sample_size))
        # fetchall() will now return a list of dictionaries if RealDictCursor is used
        articles = cursor.fetchall()

        if not articles:
            logger.warning(f"No articles found for cluster {cluster_id}")
            cursor.close()
            reader_client.release_connection(conn)
            return

        # Now we can reliably access results as dictionaries
        combined_text = " ".join(
            [article['title'] for article in articles if article.get('title')])
        article_ids = [article['id'] for article in articles]

        # Process with spaCy
        doc = nlp(combined_text)

        # Extract top entities
        entities = defaultdict(int)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT"]:
                entities[ent.text] += 1

        # Get top 5 entities
        top_entities = sorted(
            [(e, c) for e, c in entities.items()], key=lambda x: x[1], reverse=True)[:5]

        # Extract noun chunks as topics
        noun_chunks = defaultdict(int)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to short phrases
                noun_chunks[chunk.text] += 1

        # Get top 5 noun chunks
        top_chunks = sorted(
            [(c, n) for c, n in noun_chunks.items()], key=lambda x: x[1], reverse=True)[:5]

        # Create metadata
        metadata = {
            "entities": [{"text": e, "count": c} for e, c in top_entities],
            "topics": [{"text": t, "count": c} for t, c in top_chunks],
            "sample_size": len(articles),
            "sample_ids": article_ids
        }

        metadata_json = json.dumps(metadata)

        # First check if the metadata column exists in the clusters table
        try:
            check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'clusters' AND column_name = 'metadata'
            )
            """
            # Use a standard cursor for schema check if RealDictCursor causes issues
            schema_cursor = conn.cursor()
            schema_cursor.execute(check_query)
            column_exists = schema_cursor.fetchone()[0]
            schema_cursor.close()

            if column_exists:
                # Update cluster metadata in database
                update_query = """
                UPDATE clusters
                SET metadata = %s
                WHERE id = %s
                """
                cursor.execute(update_query, (metadata_json, cluster_id))
                conn.commit()
                logger.info(
                    f"Updated metadata for cluster {cluster_id}: {len(top_entities)} entities, {len(top_chunks)} topics")
            else:
                # Try to add the column if it doesn't exist
                try:
                    logger.info(
                        f"Metadata column doesn't exist in clusters table. Attempting to add it.")
                    alter_query = """
                    ALTER TABLE clusters 
                    ADD COLUMN IF NOT EXISTS metadata JSONB;
                    """
                    schema_cursor = conn.cursor()
                    schema_cursor.execute(alter_query)
                    conn.commit()
                    schema_cursor.close()

                    # Now try the update again
                    update_query = """
                    UPDATE clusters
                    SET metadata = %s
                    WHERE id = %s
                    """
                    cursor.execute(update_query, (metadata_json, cluster_id))
                    conn.commit()
                    logger.info(
                        f"Added metadata column and updated cluster {cluster_id}: {len(top_entities)} entities, {len(top_chunks)} topics")
                except Exception as alter_error:
                    # If we can't alter the table, just log the metadata instead
                    logger.warning(
                        f"Unable to add metadata column: {alter_error}")
                    logger.info(
                        f"Cluster {cluster_id} interpretation (not stored in DB): {len(top_entities)} entities, {len(top_chunks)} topics")
                    logger.debug(f"Metadata content: {metadata_json[:100]}...")
        except Exception as schema_error:
            logger.warning(f"Error checking schema: {schema_error}")
            # Just log the interpretation results without storing
            logger.info(
                f"Cluster {cluster_id} interpretation (not stored): {len(top_entities)} entities, {len(top_chunks)} topics")

        cursor.close()

    except Exception as e:
        logger.error(
            f"Error interpreting cluster {cluster_id}: {e}", exc_info=True)
    finally:
        # Ensure connection is always released
        if conn:
            reader_client.release_connection(conn)
