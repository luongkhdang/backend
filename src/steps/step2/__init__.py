"""
step2 package - Article Clustering Module

This package implements Step 2 of the data refinery pipeline: clustering articles based on their 
embeddings. It uses HDBSCAN for clustering and updates the database with cluster assignments
and centroids, including sophisticated hotness determination.

Exported functions:
- run(): Main function that orchestrates the clustering process
  - Returns Dict[str, Any]: Status report of clustering operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for embeddings and clusters
"""

import logging
from typing import Dict, Any

# Import necessary submodules
from .data_fetching import get_all_embeddings
from .clustering import calculate_centroids
from .hotness import calculate_hotness_factors
from .database import insert_clusters, batch_update_article_cluster_assignments
from .interpretation import interpret_cluster, get_cluster_keywords
from .utils import initialize_nlp

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run() -> Dict[str, Any]:
    """
    Main function to run the article clustering process.

    This function orchestrates the entire clustering process:
    1. Retrieves all article embeddings and metadata from the database
    2. Uses HDBSCAN to cluster the embeddings
    3. Calculates centroids for each cluster
    4. Determines "hotness" of clusters using a weighted combination of factors
    5. Stores clusters and updates article cluster assignments in the database
    6. Optionally interprets clusters and updates metadata

    Returns:
        Dict[str, Any]: Status report containing metrics about the clustering process
    """
    # Import the run function from the core module
    from .core import run as core_run

    # Call and return the result from the core module's run function
    return core_run()
