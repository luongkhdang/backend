#!/usr/bin/env python3
"""
step4.py - Article Export Module

This module implements Step 4 of the data refinery pipeline: exporting recently processed
articles along with their metadata and top entities to JSON files. It retrieves articles 
published yesterday or today, fetches their associated domain goodness scores and top 
entities by influence score, and saves the consolidated data to a timestamped JSON file.

Exported functions:
- run(): Main function that orchestrates the article export process
  - Returns Dict[str, Any]: Status report of the export operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for articles and entities
- src/database/modules/articles.py: Article retrieval operations
- src/database/modules/entities.py: Entity retrieval operations
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.database.reader_db_client import ReaderDBClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run() -> Dict[str, Any]:
    """
    Main function to run the article export process.

    This function:
    1. Retrieves recently processed articles (published yesterday or today)
    2. Fetches domain goodness scores and top entities for each article
    3. Exports the consolidated data to a timestamped JSON file in src/output/

    Returns:
        Dict[str, Any]: Status report containing metrics about the export process
    """
    start_time = datetime.now()

    # Initialize the status dictionary with default values
    status = {
        "success": False,
        "articles_processed": 0,
        "output_file": None,
        "error": None,
        "runtime_seconds": 0
    }

    # Initialize database client
    db_client = None

    try:
        # Initialize database client
        db_client = ReaderDBClient()

        # Step 1: Retrieve recently processed articles with their details
        logger.info(
            "Fetching recently processed articles (published yesterday or today)...")
        articles = db_client.get_recent_day_processed_articles_with_details()

        if not articles:
            logger.warning("No recently processed articles found.")
            status["articles_processed"] = 0
            status["success"] = True
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        logger.info(f"Found {len(articles)} articles to export")

        # Step 2: Initialize output data list
        output_data = []

        # Step 3: Process each article
        for article in articles:
            article_id = article.get('article_id')
            if not article_id:
                logger.warning(f"Skipping article with missing ID: {article}")
                continue

            # Get top entities for the article
            top_entities = db_client.get_top_entities_for_article(
                article_id, limit=10)

            # Structure the data for this article
            article_data = {
                'article_id': article_id,
                'title': article.get('title', ''),
                'domain': article.get('domain', ''),
                'goodness_score': article.get('goodness_score', 0.5),
                'pub_date': article.get('pub_date'),
                'cluster_id': article.get('cluster_id'),
                'top_entities': top_entities
            }

            output_data.append(article_data)

        # Step 4: Create output directory if it doesn't exist
        output_dir = "src/output/"
        os.makedirs(output_dir, exist_ok=True)

        # Step 5: Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"step4_output_{timestamp}.json"
        output_path = os.path.join(output_dir, output_filename)

        # Step 6: Write data to JSON file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(
            f"Successfully exported {len(output_data)} articles to {output_path}")

        # Set success status
        status["success"] = True
        status["articles_processed"] = len(output_data)
        status["output_file"] = output_path
        status["runtime_seconds"] = (
            datetime.now() - start_time).total_seconds()

        return status

    except Exception as e:
        logger.error(f"Error in Step 4 (Article Export): {e}", exc_info=True)
        status["error"] = str(e)
        status["runtime_seconds"] = (
            datetime.now() - start_time).total_seconds()
        return status

    finally:
        # Ensure the database connection is properly closed
        if db_client:
            db_client.close()
            logger.debug("Database connection closed")


if __name__ == "__main__":
    # When run directly, execute the article export process
    status = run()
    print(
        f"Article export status: {'Success' if status['success'] else 'Failed'}")
    print(f"Processed {status.get('articles_processed', 0)} articles")
    if status.get('output_file'):
        print(f"Output saved to: {status.get('output_file')}")
    if status.get('error'):
        print(f"Error: {status.get('error')}")
    print(f"Runtime: {status.get('runtime_seconds', 0):.2f} seconds")
