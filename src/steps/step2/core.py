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
import importlib.util
import random
from typing import List, Dict, Any, Tuple, Optional, Set

# Handle optional dependencies with try/except blocks
try:
    import numpy as np
except ImportError:
    logging.error("numpy is required but not available")
    raise

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    logging.warning("hdbscan not available; clustering will not be possible")
    hdbscan = None
    HDBSCAN_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning(
        "sklearn not available; some functionality will be limited")
    cosine_distances = None
    normalize = None
    SKLEARN_AVAILABLE = False

# Import database client
from src.database.reader_db_client import ReaderDBClient

# Import from local modules
from .data_fetching import get_all_embeddings
from .clustering import calculate_centroids, cluster_articles
from .hotness import calculate_hotness_factors, get_weight_params
from .database import insert_clusters, batch_update_article_cluster_assignments
from .interpretation import interpret_cluster, get_cluster_keywords
from .utils import initialize_nlp, SPACY_AVAILABLE
from .metadata_generation import generate_and_update_cluster_metadata

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
            logger.debug(
                f"Shape of embeddings list before conversion: ({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})")
            logger.debug(
                f"Type of 'embeddings': {type(embeddings)}, Type of first element: {type(embeddings[0]) if embeddings else 'N/A'}")

            # Log the first few values of the first embedding for debugging
            if embeddings and len(embeddings) > 0:
                first_embedding = embeddings[0]
                sample_values = first_embedding[:5] if len(
                    first_embedding) >= 5 else first_embedding
                logger.debug(
                    f"Sample values from first embedding: {sample_values}")
                logger.debug(
                    f"Types of sample values: {[type(v) for v in sample_values]}")
        else:
            logger.warning("Embeddings list is empty before conversion.")

        # Convert list of embeddings to numpy array
        X = np.array(embeddings)
        logger.debug(f"Shape of X after np.array conversion: {X.shape}")
        logger.debug(f"Data type of X array: {X.dtype}")

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
                    logger.debug(
                        f"First embedding has length {embedding_length}, creating uniform 2D array")
                    # Create a new 2D array with the right dimensions
                    new_X = np.zeros((len(embeddings), embedding_length))
                    for i, emb in enumerate(embeddings):
                        if len(emb) == embedding_length:  # Skip malformed embeddings
                            new_X[i] = emb
                    X = new_X
                    logger.debug(f"Successfully reshaped to {X.shape}")
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

        try:
            # Normalize the vectors to unit length for cosine distance calculation
            X_normalized = normalize(X, norm='l2', axis=1)

            # Calculate cosine distances - this returns a matrix where
            # each entry (i,j) is the cosine distance between vectors i and j
            distance_matrix = cosine_distances(X_normalized)

            logger.debug(f"Distance matrix shape: {distance_matrix.shape}")

            # Initialize HDBSCAN with precomputed distance matrix
            # Note: HDBSCAN does not support GPU acceleration, all computation is CPU-based
            # regardless of available hardware. The precomputed matrix approach can be more efficient
            # for cosine distances with high-dimensional data.
            clusterer = hdbscan.HDBSCAN(
                metric='precomputed',  # Tell HDBSCAN we're giving it precomputed distances
                min_cluster_size=min_cluster_size,
                core_dist_n_jobs=-1  # Use all available cores
            )

            # Fit using the distance matrix
            labels = clusterer.fit_predict(distance_matrix)

        except Exception as e:
            logger.error(f"Error computing distances or clustering: {e}")

            # Fallback to using the cluster_articles function with euclidean metric
            logger.info(
                "Falling back to direct clustering with cosine metric")
            labels, _ = cluster_articles(
                embeddings=X,
                min_cluster_size=min_cluster_size,
                metric='cosine'  # Use cosine metric for high-dimensional data
            )

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

        # Step 5: Calculate cluster centroids
        logger.info(
            "Step 2.5: Calculating cluster centroids and organizing cluster data")
        cluster_data, centroids = calculate_centroids(labels, X)
        logger.info(f"Calculated centroids for {len(centroids)} clusters")

        # Step 6: Determine hotness of clusters
        logger.info(
            "Step 2.6: Determining cluster hotness using weighted factors")
        try:
            cluster_hotness_map, cluster_hotness_scores = calculate_hotness_factors(
                cluster_data=cluster_data,
                article_ids=article_ids,
                pub_dates=pub_dates,
                reader_client=reader_client,
                nlp=nlp
            )
            hot_clusters = sum(
                1 for is_hot in cluster_hotness_map.values() if is_hot)
            logger.info(
                f"Determined {hot_clusters} hot clusters based on weighted factor analysis")
        except Exception as e:
            logger.error(
                f"Error calculating hotness factors: {e}", exc_info=True)
            if "numpy.int64" in str(e):
                logger.info(
                    "Attempting to convert numpy types to native Python types")
                # If we got a numpy.int64 error, we can try again with converted types
                # This is a fallback in case our fixes in other modules didn't catch all cases
                status["error"] = f"Error in hotness calculation: {e}"
                return status
            raise

        # Step 7: Delete existing clusters from today before creating new ones
        logger.info(
            "Step 2.7: Deleting existing clusters from today (Pacific Time)")
        deleted_count = reader_client.delete_todays_clusters()
        logger.info(f"Deleted {deleted_count} existing clusters from today")

        # Step 8: Store clusters in the database
        logger.info("Step 2.8: Storing clusters in database")

        # Calculate cluster article counts
        cluster_article_counts = {label: len(
            indices) for label, indices in cluster_data.items()}

        cluster_db_map = insert_clusters(
            reader_client, centroids, cluster_hotness_map, cluster_hotness_scores, cluster_article_counts)
        logger.info(f"Inserted {len(cluster_db_map)} clusters into database")

        # Step 9: Update article cluster assignments
        logger.info("Step 2.9: Updating article cluster assignments")
        update_stats = batch_update_article_cluster_assignments(
            reader_client, article_ids, labels, cluster_db_map, cluster_hotness_map)

        # Access the dictionary values correctly
        if isinstance(update_stats, dict):
            status["articles_assigned"] = update_stats.get("success", 0)
            status["db_update_success"] = update_stats.get("success", 0)
            status["db_update_failure"] = update_stats.get("failure", 0)
        else:
            # Fallback in case update_stats is not a dict (backward compatibility)
            logger.warning("update_stats is not a dictionary as expected")
            if isinstance(update_stats, tuple) and len(update_stats) == 2:
                status["articles_assigned"] = update_stats[0]
                status["db_update_success"] = update_stats[0]
                status["db_update_failure"] = update_stats[1]

        # Step 10: Enhanced cluster metadata generation
        logger.info("Step 2.10: Generating enhanced cluster metadata")
        try:
            # Create a mapping of cluster_label -> list of article_ids
            cluster_article_map = {}
            for article_index, label in enumerate(labels):
                if label >= 0:  # Skip noise points (label -1)
                    if label not in cluster_article_map:
                        cluster_article_map[label] = []
                    cluster_article_map[label].append(
                        article_ids[article_index])

            # Create a mapping of article_id -> embedding for efficient lookups
            embeddings_map = {article_ids[i]: embeddings[i]
                              for i in range(len(article_ids))}

            # Process each cluster
            for label, cluster_articles in cluster_article_map.items():
                db_cluster_id = cluster_db_map.get(label)
                if db_cluster_id and cluster_articles:
                    # Get the centroid for this cluster
                    cluster_centroid = centroids.get(label)

                    # Generate enhanced metadata for this cluster
                    generate_and_update_cluster_metadata(
                        cluster_id=db_cluster_id,
                        article_ids=cluster_articles,
                        embeddings_map=embeddings_map,
                        cluster_centroid=cluster_centroid,
                        reader_db_client=reader_client
                    )

            logger.info(
                f"Successfully generated enhanced metadata for {len(cluster_article_map)} clusters")
        except Exception as e:
            logger.warning(
                f"Error during enhanced metadata generation: {e}", exc_info=True)
            # Don't fail the whole process if metadata generation fails

        # Step 11 (Optional): Basic cluster interpretation with spaCy
        if SPACY_AVAILABLE and os.getenv("INTERPRET_CLUSTERS", "false").lower() == "true" and nlp is not None:
            logger.info("Step 2.11: Performing basic cluster interpretation")
            try:
                # Get cluster interpretations for a subset of clusters (limit to save time)
                max_clusters_to_interpret = min(
                    int(os.getenv("MAX_CLUSTERS_TO_INTERPRET", "10")), num_clusters)

                # Get the largest clusters by article count
                largest_clusters = sorted(
                    [(label, len(data))  # Use len(data) to get the count
                     for label, data in cluster_data.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:max_clusters_to_interpret]

                for label, count in largest_clusters:
                    db_cluster_id = cluster_db_map.get(label)
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
