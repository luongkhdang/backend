"""
core.py - Core implementation of the clustering process

This module contains the main run function that implements the entire
clustering process flow, from fetching data to inserting clusters and
updating article assignments.

Exported functions:
- run() -> Dict[str, Any]: Main function orchestrating the entire clustering process,
  returns status report of the operation

Related files:
- src/steps/step2/__init__.py: Imports and re-exports run() function
- src/steps/step2/data_fetching.py: Used to retrieve embeddings
- src/steps/step2/clustering.py: Used for centroid calculation and cluster evaluation
- src/steps/step2/hotness.py: Used for cluster hotness determination
- src/steps/step2/database.py: Used to store clusters and update articles
- src/steps/step2/interpretation.py: Used for cluster interpretation
- src/database/reader_db_client.py: Database operations
"""

import logging
import os
import time
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from typing import List, Dict, Any, Tuple, Optional, Set
import importlib.util
import random

# Import database client
from src.database.reader_db_client import ReaderDBClient

# Import from local modules
from .data_fetching import get_all_embeddings
from .clustering import calculate_centroids
from .hotness import calculate_hotness_factors, get_hotness_threshold, get_weight_params
from .database import insert_clusters, batch_update_article_cluster_assignments
from .interpretation import interpret_cluster, get_cluster_keywords
from .utils import initialize_nlp, SPACY_AVAILABLE

# Configure logging
logger = logging.getLogger(__name__)


def run() -> Dict[str, Any]:
    """
    Main function to run the article clustering process.

    This function:
    1. Retrieves all article embeddings from the database
    2. Uses HDBSCAN to cluster the embeddings
    3. Calculates centroids for each cluster
    4. Determines "hotness" of clusters based on multiple factors
    5. Stores clusters and updates article cluster assignments in the database
    6. Optionally interprets clusters using spaCy

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
    nlp = None

    try:
        # Step 1: Initialize database client
        logger.info("Step 2.1: Initializing database connection")
        reader_client = ReaderDBClient()

        # Load spaCy model if needed for topic relevance calculation or interpretation
        calculate_topic_relevance = os.getenv(
            "CALCULATE_TOPIC_RELEVANCE", "true").lower() == "true"
        interpret_clusters = os.getenv(
            "INTERPRET_CLUSTERS", "false").lower() == "true"

        if (calculate_topic_relevance or interpret_clusters) and SPACY_AVAILABLE:
            try:
                nlp = initialize_nlp()
                logger.info(
                    "Successfully loaded spaCy model for keyword extraction")
            except Exception as e:
                logger.warning(
                    f"Failed to load spaCy model: {e}. Topic relevance calculation may be limited.")
                nlp = None

        # Step 2: Retrieve all embeddings and metadata from the database
        logger.info(
            "Step 2.2: Fetching article embeddings and metadata from database")
        embeddings_data = get_all_embeddings(reader_client)

        if not embeddings_data:
            logger.warning(
                "No embeddings found in database. Clustering skipped.")
            status["error"] = "No embeddings found"
            return status

        article_ids, embeddings, pub_dates = zip(*embeddings_data)
        status["embeddings_processed"] = len(embeddings)
        logger.info(
            f"Retrieved {len(embeddings)} article embeddings for clustering")

        # Step 3: Prepare data for clustering
        logger.info("Step 2.3: Preparing data for clustering")

        # Add logging for embeddings list before conversion
        if embeddings:
            logger.info(
                f"Shape of embeddings list before conversion: ({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})")
            logger.info(
                f"Type of 'embeddings': {type(embeddings)}, Type of first element: {type(embeddings[0]) if embeddings else 'N/A'}")

            # Log the first few values of the first embedding for debugging
            if embeddings and len(embeddings) > 0:
                first_embedding = embeddings[0]
                sample_values = first_embedding[:5] if len(
                    first_embedding) >= 5 else first_embedding
                logger.info(
                    f"Sample values from first embedding: {sample_values}")
                logger.info(
                    f"Types of sample values: {[type(v) for v in sample_values]}")
        else:
            logger.warning("Embeddings list is empty before conversion.")

        # Convert list of embeddings to numpy array
        X = np.array(embeddings)
        logger.info(f"Shape of X after np.array conversion: {X.shape}")
        logger.info(f"Data type of X array: {X.dtype}")

        # Check if we have string or object dtype and convert if needed
        if X.dtype == 'O' or np.issubdtype(X.dtype, np.string_) or np.issubdtype(X.dtype, np.unicode_):
            logger.warning(
                f"Non-numeric dtype detected: {X.dtype}. Attempting to convert to float.")
            try:
                # Try to convert to float array
                X = X.astype(np.float64)
                logger.info(
                    f"Successfully converted to float64. New dtype: {X.dtype}")
            except ValueError as e:
                logger.error(f"Failed to convert to float: {e}")
                status["error"] = f"Failed to convert embeddings to float: {e}"
                return status

        # Try to fix dimension issues if we have a 1D array
        if X.ndim == 1 and len(X) > 0:
            logger.warning(
                f"Got 1D array with length {len(X)}, attempting to reshape...")
            try:
                # If each embedding is itself a list/array, X might be an array of arrays
                # which numpy couldn't properly reshape automatically
                if isinstance(embeddings[0], (list, np.ndarray)):
                    # Manually create a 2D array - probably needed if embeddings are different lengths
                    embedding_length = len(embeddings[0])
                    logger.info(
                        f"First embedding has length {embedding_length}, creating uniform 2D array")
                    # Create a new 2D array with the right dimensions
                    new_X = np.zeros((len(embeddings), embedding_length))
                    for i, emb in enumerate(embeddings):
                        if len(emb) == embedding_length:  # Skip malformed embeddings
                            new_X[i] = emb
                    X = new_X
                    logger.info(f"Successfully reshaped to {X.shape}")
                else:
                    # If X is a flat array, reshape it into a 2D array with 1 feature
                    X = X.reshape(-1, 1)
                    logger.info(f"Reshaped flat array to {X.shape}")
            except Exception as reshape_error:
                logger.error(f"Reshaping failed: {reshape_error}")
                # Continue with validation which will catch the issue

        # Add validation before HDBSCAN
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            error_msg = f"Invalid shape for HDBSCAN input: {X.shape}. Expected 2D array."
            logger.error(error_msg)
            status["error"] = error_msg
            if reader_client:
                reader_client.close()
            return status
        logger.info(f"Data shape {X.shape} is valid for HDBSCAN.")

        # Step 4: Run HDBSCAN clustering
        logger.info("Step 2.4: Running HDBSCAN clustering algorithm")
        min_cluster_size = int(os.getenv("MIN_CLUSTER_SIZE", "10"))
        logger.info(f"Using min_cluster_size={min_cluster_size}")

        # Compute cosine distance matrix explicitly
        logger.info("Computing cosine distance matrix for clustering")
        try:
            # Normalize the vectors to unit length for cosine distance calculation
            X_normalized = normalize(X, norm='l2', axis=1)

            # Calculate cosine distances - this returns a matrix where
            # each entry (i,j) is the cosine distance between vectors i and j
            distance_matrix = cosine_distances(X_normalized)

            logger.info(f"Distance matrix shape: {distance_matrix.shape}")

            # Initialize HDBSCAN with precomputed distance matrix
            # Note: prediction_data is not supported with precomputed distances
            clusterer = hdbscan.HDBSCAN(
                metric='precomputed',  # Tell HDBSCAN we're giving it precomputed distances
                min_cluster_size=min_cluster_size,
                core_dist_n_jobs=-1  # Use all available cores
            )

            # Fit using the distance matrix
            labels = clusterer.fit_predict(distance_matrix)

        except Exception as e:
            logger.error(f"Error computing distances or clustering: {e}")

            # Fallback to euclidean metric which is always supported
            # With direct vector inputs, we can use prediction_data
            logger.info("Falling back to euclidean metric")
            clusterer = hdbscan.HDBSCAN(
                metric='euclidean',
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

        # Step 5.5: Calculate hotness factors
        logger.info("Step 2.5.5: Calculating cluster hotness factors")

        # Create list of article data in the expected format (article_id, embedding)
        article_data_for_hotness = [(article_ids[i], X[i])
                                    for i in range(len(article_ids))]

        # Call with the correct number of arguments
        cluster_hotness_map = calculate_hotness_factors(
            article_data_for_hotness,
            labels,
            pub_dates
        )

        # Step 6: Update database with clusters
        logger.info("Step 2.6: Inserting clusters into database")
        hdbscan_to_db_id_map = insert_clusters(
            reader_client, cluster_data, cluster_hotness_map)

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
        if SPACY_AVAILABLE and os.getenv("INTERPRET_CLUSTERS", "false").lower() == "true" and nlp is not None:
            logger.info("Step 2.8: Performing basic cluster interpretation")
            try:
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
