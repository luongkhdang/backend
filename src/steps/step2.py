#!/usr/bin/env python3
"""
step2.py - Article Clustering Module

This module implements Step 2 of the data refinery pipeline: clustering articles based on their 
embeddings. It uses HDBSCAN for clustering and updates the database with cluster assignments
and centroids.

This file has been refactored into a proper module structure in the src/steps/step2/ directory.
It now serves as a compatibility layer for existing code, redirecting to the modularized implementation.

Exported functions:
- run(): Main function that orchestrates the clustering process
  - Returns Dict[str, Any]: Status report of clustering operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for embeddings and clusters
- src/steps/step2/__init__.py: New modularized implementation
"""

import logging
from typing import Dict, Any

# Import from the modularized implementation
from src.steps.step2 import run as modular_run

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    # Simply delegate to the modularized implementation
    logger.info("Redirecting to modularized implementation in src/steps/step2/")
    return modular_run()


if __name__ == "__main__":
    # When run directly, execute the clustering process
    status = run()
    print(f"Clustering status: {'Success' if status['success'] else 'Failed'}")
    print(f"Processed {status['embeddings_processed']} embeddings")
    print(
        f"Found {status['clusters_found']} clusters ({status['noise_points']} noise points)")
    print(f"Runtime: {status['runtime_seconds']} seconds")
