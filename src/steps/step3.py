#!/usr/bin/env python3
"""
step3.py - Entity Extraction Module

This module implements Step 3 of the data refinery pipeline: processing recent articles to extract
entities, store entity relationships, and update basic entity statistics. It prioritizes articles
based on combined domain goodness and cluster hotness scores, then calls Gemini API for entity
extraction with tier-based model selection.

This file has been refactored into a proper module structure in the src/steps/step3/ directory.
It now serves as a compatibility layer for existing code, redirecting to the modularized implementation.

Exported functions:
- run(): Main function that orchestrates the entity extraction process
  - Returns Dict[str, Any]: Status report of entity extraction operation

Related files:
- src/main.py: Calls this module as part of the pipeline
- src/database/reader_db_client.py: Database operations for articles and entities
- src/gemini/gemini_client.py: Used for entity extraction API calls
- src/steps/step3/__init__.py: New modularized implementation
- src/utils/task_manager.py: Manages concurrent execution of API calls
"""

import logging
import asyncio
from typing import Dict, Any

# Import from the modularized implementation
from src.steps.step3 import run as modular_run

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run() -> Dict[str, Any]:
    """
    Main function to run the entity extraction process.

    This function:
    1. Retrieves domain goodness scores
    2. Fetches and prioritizes recent unprocessed articles
    3. Extracts entities using Gemini API with tier-based model selection using TaskManager for concurrency
    4. Stores entity results in the database
    5. Updates article processing status

    Returns:
        Dict[str, Any]: Status report containing metrics about the extraction process
    """
    # Simply delegate to the modularized implementation, which is now async
    logger.debug(
        "Redirecting to modularized async implementation in src/steps/step3/")
    try:
        return await modular_run()
    except Exception as e:
        logger.critical(
            f"Critical error in step3 processing: {e}", exc_info=True)
        # Return error status
        return {
            "success": False,
            "error": f"Critical error: {str(e)}",
            "processed": 0,
            "entity_links_created": 0,
            "runtime_seconds": 0
        }


# For backward compatibility with synchronous code
def run_sync() -> Dict[str, Any]:
    """
    Synchronous wrapper around the async run function.
    This is provided for backward compatibility with code that can't use async/await.
    It's recommended to use the async run() function directly when possible.

    Returns:
        Dict[str, Any]: Status report containing metrics about the extraction process
    """
    logger.warning(
        "Using run_sync() - consider migrating to async run() for better performance")
    try:
        return asyncio.run(run())
    except RuntimeError as e:
        logger.error(f"Error running async code in sync context: {e}")
        # Return failure status
        return {
            "success": False,
            "error": f"Error running async code: {str(e)}",
            "processed": 0,
            "entity_links_created": 0,
            "runtime_seconds": 0
        }


if __name__ == "__main__":
    # When run directly, execute the entity extraction process using asyncio.run
    try:
        status = asyncio.run(run())
        print(
            f"Entity extraction status: {'Success' if status['success'] else 'Failed'}")
        print(f"Processed {status.get('processed', 0)} articles")
        print(f"Created {status.get('entity_links_created', 0)} entity links")
        print(f"Runtime: {status.get('runtime_seconds', 0):.2f} seconds")
    except Exception as e:
        print(f"Entity extraction failed with error: {e}")
