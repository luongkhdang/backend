"""
data_fetching.py - Functions for retrieving data from database

This module contains functions for fetching embeddings and associated metadata
from the database for the clustering process.

Exported functions:
- get_all_embeddings(reader_client: ReaderDBClient) -> List[Tuple[int, List[float], Optional[datetime]]]
  Retrieves all article embeddings and publication dates from the database

Related files:
- src/steps/step2/core.py: Uses this module to fetch data for clustering
- src/database/reader_db_client.py: Database client used for data retrieval
"""

import logging
from typing import List, Tuple, Any, Dict, Optional
from datetime import datetime

from src.database.reader_db_client import ReaderDBClient

# Configure logging
logger = logging.getLogger(__name__)


def get_all_embeddings(reader_client: ReaderDBClient) -> List[Tuple[int, List[float], Optional[datetime]]]:
    """
    Retrieve all article embeddings and relevant metadata from the database.

    Args:
        reader_client: Initialized ReaderDBClient

    Returns:
        List of tuples (article_id, embedding, pub_date)
    """
    try:
        # Use ReaderDBClient's method to fetch embeddings with publication dates
        all_embeddings = reader_client.get_all_embeddings_with_pub_date()

        # Process results - ensure embeddings are converted to float values
        embeddings_data = []
        for embedding_data in all_embeddings:
            article_id = embedding_data.get('article_id')
            embedding = embedding_data.get('embedding')
            pub_date = embedding_data.get('pub_date')

            # Skip if missing critical data
            if article_id is None or embedding is None:
                continue

            # Process embedding to ensure it's a list of floats
            if isinstance(embedding, list):
                # Convert each element to float if needed
                float_embedding = [float(val) if not isinstance(
                    val, float) else val for val in embedding]
                embeddings_data.append((article_id, float_embedding, pub_date))
            elif isinstance(embedding, str):
                # If embedding is a string (like a JSON array), parse it
                try:
                    import json
                    float_embedding = [float(val)
                                       for val in json.loads(embedding)]
                    embeddings_data.append(
                        (article_id, float_embedding, pub_date))
                except Exception as e:
                    logger.warning(
                        f"Failed to parse embedding string for article_id {article_id}: {e}")
                    # Skip this embedding
            else:
                logger.warning(
                    f"Unexpected embedding type for article_id {article_id}: {type(embedding)}")
                # Skip this embedding

        # Log some debug info
        if embeddings_data:
            logger.info(f"First embedding type: {type(embeddings_data[0][1])}")
            logger.info(
                f"First embedding element type: {type(embeddings_data[0][1][0]) if embeddings_data[0][1] else 'N/A'}")
            logger.info(
                f"Retrieved {len(embeddings_data)} embeddings from the database")

        return embeddings_data
    except Exception as e:
        logger.error(f"Error fetching embeddings: {e}", exc_info=True)
        return []
