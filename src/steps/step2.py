#!/usr/bin/env python3
"""
step2.py - Article Clustering Module

This module implements Step 2 of the data refinery pipeline: clustering articles based on their 
embeddings. It uses HDBSCAN for clustering and updates the database with cluster assignments
and centroids.

Exported functions:
- run(): Main function that orchestrates the clustering process
  - Returns Dict[str, Any]: Status report of clustering operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for embeddings and clusters
"""

import logging
import os
import time
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Tuple, Optional, Set
import importlib.util
from collections import defaultdict

# Import database client
from src.database.reader_db_client import ReaderDBClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check if spaCy is available for cluster interpretation
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None


def run() -> Dict[str, Any]:
    """
    Main function to run the article clustering process.

    This function:
    1. Retrieves all article embeddings from the database
    2. Uses HDBSCAN to cluster the embeddings
    3. Calculates centroids for each cluster
    4. Stores clusters and updates article cluster assignments in the database
    5. Optionally interprets clusters using spaCy

    Returns:
        Dict[str, Any]: Status report containing metrics about the clustering process
    """
    start_time = time.time()
    status = {
        "embeddings_processed": 0,
        "clusters_found": 0,
        "noise_points": 0,
        "articles_assigned": 0,
        "db_update_success": 0,
        "db_update_failure": 0,
        "runtime_seconds": 0,
        "success": False,
        "error": None
    }

    reader_client = None

    try:
        # Step 1: Initialize database client
        logger.info("Step 2.1: Initializing database connection")
        reader_client = ReaderDBClient()

        # Step 2: Retrieve all embeddings from the database
        logger.info("Step 2.2: Fetching article embeddings from database")
        embeddings_data = get_all_embeddings(reader_client)

        if not embeddings_data:
            logger.warning(
                "No embeddings found in database. Clustering skipped.")
            status["error"] = "No embeddings found"
            return status

        article_ids, embeddings = zip(*embeddings_data)
        status["embeddings_processed"] = len(embeddings)
        logger.info(
            f"Retrieved {len(embeddings)} article embeddings for clustering")

        # Step 3: Prepare data for clustering
        logger.info("Step 2.3: Preparing data for clustering")
        # Convert list of embeddings to numpy array
        X = np.array(embeddings)

        # Step 4: Run HDBSCAN clustering
        logger.info("Step 2.4: Running HDBSCAN clustering algorithm")
        min_cluster_size = int(os.getenv("MIN_CLUSTER_SIZE", "10"))
        logger.info(f"Using min_cluster_size={min_cluster_size}")

        # Initialize and run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            metric='cosine',
            min_cluster_size=min_cluster_size,
            prediction_data=True,
            core_dist_n_jobs=-1  # Use all available cores
        )
        labels = clusterer.fit_predict(X)

        # Analyze clustering results
        unique_labels = set(labels)
        num_clusters = len([l for l in unique_labels if l >= 0])
        num_noise = list(labels).count(-1)

        status["clusters_found"] = num_clusters
        status["noise_points"] = num_noise

        logger.info(
            f"Clustering complete: {num_clusters} clusters found, {num_noise} noise points")

        if num_clusters == 0:
            logger.warning(
                "No clusters found. Check min_cluster_size parameter.")
            status["error"] = "No clusters found"
            return status

        # Step 5: Calculate centroids
        logger.info("Step 2.5: Calculating cluster centroids")
        cluster_data = calculate_centroids(X, labels)

        # Step 6: Update database with clusters
        logger.info("Step 2.6: Inserting clusters into database")
        hdbscan_to_db_id_map = insert_clusters(reader_client, cluster_data)

        if not hdbscan_to_db_id_map:
            logger.error("Failed to insert any clusters into database")
            status["error"] = "Cluster insertion failed"
            return status

        # Step 7: Update article cluster assignments
        logger.info("Step 2.7: Updating article cluster assignments")
        assignments = []
        for i, (article_id, label) in enumerate(zip(article_ids, labels)):
            # Map HDBSCAN label to database cluster_id
            # If label is -1 (noise) or not in map, cluster_id will be None
            cluster_id = hdbscan_to_db_id_map.get(label)
            assignments.append((article_id, cluster_id))

        status["articles_assigned"] = len(assignments)
        success_count, fail_count = batch_update_article_cluster_assignments(
            reader_client, assignments)
        status["db_update_success"] = success_count
        status["db_update_failure"] = fail_count

        # Step 8 (Optional): Basic cluster interpretation with spaCy
        if SPACY_AVAILABLE and os.getenv("INTERPRET_CLUSTERS", "false").lower() == "true":
            logger.info("Step 2.8: Performing basic cluster interpretation")
            try:
                import spacy
                nlp = spacy.load("en_core_web_lg")

                # Get cluster interpretations for a subset of clusters (limit to save time)
                max_clusters_to_interpret = min(
                    int(os.getenv("MAX_CLUSTERS_TO_INTERPRET", "10")), num_clusters)

                # Get the largest clusters by article count
                largest_clusters = sorted(
                    [(label, data["count"])
                     for label, data in cluster_data.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:max_clusters_to_interpret]

                for label, count in largest_clusters:
                    db_cluster_id = hdbscan_to_db_id_map.get(label)
                    if db_cluster_id:
                        interpret_cluster(reader_client, db_cluster_id, nlp)
            except Exception as e:
                logger.warning(f"Cluster interpretation failed: {e}")
                # Don't fail the whole process if interpretation fails

        status["success"] = True

    except Exception as e:
        logger.error(f"Error in clustering process: {e}", exc_info=True)
        status["error"] = str(e)
    finally:
        # Close database connection
        if reader_client:
            reader_client.close()

    # Calculate runtime
    end_time = time.time()
    runtime_seconds = round(end_time - start_time, 2)
    status["runtime_seconds"] = runtime_seconds
    logger.info(f"Clustering completed in {runtime_seconds} seconds")

    return status


def get_all_embeddings(reader_client: ReaderDBClient) -> List[Tuple[int, List[float]]]:
    """
    Retrieve all article embeddings from the database.

    Args:
        reader_client: Initialized ReaderDBClient

    Returns:
        List of tuples (article_id, embedding)
    """
    try:
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        # Query to fetch all embeddings
        query = """
        SELECT e.article_id, e.embedding
        FROM embeddings e
        WHERE e.embedding IS NOT NULL
        """

        cursor.execute(query)
        results = cursor.fetchall()

        # Process results
        embeddings_data = []
        for article_id, embedding in results:
            embeddings_data.append((article_id, embedding))

        cursor.close()
        reader_client.release_connection(conn)

        return embeddings_data
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}", exc_info=True)
        return []


def calculate_centroids(X: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
    """
    Calculate centroids for each cluster.

    Args:
        X: Array of embeddings
        labels: Cluster labels from HDBSCAN

    Returns:
        Dictionary mapping cluster labels to cluster data (centroid, count, indices)
    """
    # Initialize dictionary to store cluster data
    cluster_data = {}

    # Group indices by cluster label
    for i, label in enumerate(labels):
        if label >= 0:  # Skip noise points (-1)
            if label not in cluster_data:
                cluster_data[label] = {
                    "indices": [],
                    "centroid": None,
                    "count": 0
                }
            cluster_data[label]["indices"].append(i)

    # Calculate centroid for each cluster
    for label, data in cluster_data.items():
        indices = data["indices"]
        cluster_embeddings = X[indices]

        # Calculate mean embedding vector
        centroid = np.mean(cluster_embeddings, axis=0)

        # Normalize centroid to unit length
        normalized_centroid = normalize(centroid.reshape(1, -1), norm='l2')[0]

        # Store data
        data["centroid"] = normalized_centroid.tolist()
        data["count"] = len(indices)

    return cluster_data


def insert_clusters(reader_client: ReaderDBClient, cluster_data: Dict[int, Dict[str, Any]]) -> Dict[int, int]:
    """
    Insert clusters into the database.

    Args:
        reader_client: Initialized ReaderDBClient
        cluster_data: Dictionary of cluster data from calculate_centroids

    Returns:
        Dictionary mapping HDBSCAN labels to database cluster IDs
    """
    hdbscan_to_db_id_map = {}

    for label, data in cluster_data.items():
        centroid = data["centroid"]
        article_count = data["count"]

        # Use existing insert_cluster function, which returns cluster ID
        cluster_id = reader_client.insert_cluster(
            centroid=centroid,
            is_hot=(article_count >= int(
                os.getenv("HOT_CLUSTER_THRESHOLD", "20")))
        )

        if cluster_id:
            hdbscan_to_db_id_map[label] = cluster_id
            logger.debug(
                f"Inserted cluster {label} with {article_count} articles as DB ID {cluster_id}")

    logger.info(f"Inserted {len(hdbscan_to_db_id_map)} clusters into database")
    return hdbscan_to_db_id_map


def batch_update_article_cluster_assignments(
    reader_client: ReaderDBClient,
    assignments: List[Tuple[int, Optional[int]]]
) -> Tuple[int, int]:
    """
    Update cluster assignments for multiple articles in batch.

    Args:
        reader_client: Initialized ReaderDBClient
        assignments: List of tuples (article_id, cluster_id)

    Returns:
        Tuple of (success_count, failure_count)
    """
    if not assignments:
        return 0, 0

    try:
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        # Split assignments into batches
        batch_size = 1000
        success_count = 0
        failure_count = 0

        for i in range(0, len(assignments), batch_size):
            batch = assignments[i:i+batch_size]

            try:
                # Convert batch to SQL-friendly format and handle NULL properly
                values = []
                for article_id, cluster_id in batch:
                    if cluster_id is None:
                        values.append(f"({article_id}, NULL)")
                    else:
                        values.append(f"({article_id}, {cluster_id})")

                values_str = ", ".join(values)

                # SQL query using temporary table for efficient update
                query = f"""
                UPDATE articles
                SET cluster_id = temp.cluster_id
                FROM (VALUES {values_str}) AS temp(article_id, cluster_id)
                WHERE articles.id = temp.article_id
                """

                cursor.execute(query)
                success_count += len(batch)
            except Exception as e:
                logger.error(f"Error updating batch {i//batch_size}: {e}")
                failure_count += len(batch)
                # Continue with next batch rather than failing completely

        conn.commit()
        cursor.close()
        reader_client.release_connection(conn)

        logger.info(
            f"Updated {success_count} article cluster assignments (failed: {failure_count})")
        return success_count, failure_count

    except Exception as e:
        logger.error(
            f"Failed to update cluster assignments: {e}", exc_info=True)
        return 0, len(assignments)


def interpret_cluster(reader_client: ReaderDBClient, cluster_id: int, nlp):
    """
    Perform basic interpretation of a cluster using spaCy.

    Args:
        reader_client: Initialized ReaderDBClient
        cluster_id: Database ID of the cluster
        nlp: Loaded spaCy model
    """
    try:
        # Get a sample of articles from this cluster
        conn = reader_client.get_connection()
        cursor = conn.cursor()

        sample_size = int(os.getenv("CLUSTER_SAMPLE_SIZE", "10"))
        query = """
        SELECT id, title, content
        FROM articles
        WHERE cluster_id = %s
        ORDER BY published_at DESC
        LIMIT %s
        """

        cursor.execute(query, (cluster_id, sample_size))
        articles = cursor.fetchall()

        if not articles:
            logger.warning(f"No articles found for cluster {cluster_id}")
            return

        # Combine titles for text analysis
        combined_text = " ".join([title for _, title, _ in articles if title])

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
            "sample_ids": [article_id for article_id, _, _ in articles]
        }

        # Update cluster metadata in database
        update_query = """
        UPDATE clusters
        SET metadata = %s
        WHERE id = %s
        """

        import json
        cursor.execute(update_query, (json.dumps(metadata), cluster_id))
        conn.commit()

        logger.info(
            f"Updated metadata for cluster {cluster_id}: {len(top_entities)} entities, {len(top_chunks)} topics")

        cursor.close()
        reader_client.release_connection(conn)

    except Exception as e:
        logger.error(
            f"Error interpreting cluster {cluster_id}: {e}", exc_info=True)


if __name__ == "__main__":
    # When run directly, execute the clustering process
    status = run()
    print(f"Clustering status: {'Success' if status['success'] else 'Failed'}")
    print(f"Processed {status['embeddings_processed']} embeddings")
    print(
        f"Found {status['clusters_found']} clusters ({status['noise_points']} noise points)")
    print(f"Runtime: {status['runtime_seconds']} seconds")
