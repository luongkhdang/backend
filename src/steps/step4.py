#!/usr/bin/env python3
"""
step4.py - Article Analysis and Grouping Module

This module implements Step 4 of the data refinery pipeline:
1. Retrieves recently processed articles.
2. Prepares metadata for analysis.
3. Calls the enhanced GeminiClient with a specific prompt (`src/prompts/step4.txt`) to perform thematic grouping based on geopolitical/economic relevance.
4. Saves the structured analysis output (article groups with rationales) from Gemini to a JSON file.

Exported functions:
- run(): Main async function that orchestrates the article analysis process.
  - Returns Dict[str, Any]: Status report of the analysis and export operation.

Related files:
- src/main.py: Calls this module as part of the pipeline.
- src/database/reader_db_client.py: Database operations for articles and entities.
- src/gemini/gemini_client.py: Enhanced Gemini client used for analysis.
- src/gemini/modules/generator.py: Contains the actual implementation of analyze_articles_with_prompt.
- src/prompts/step4.txt: Prompt defining the analysis task for Gemini.
"""

import os
import json
import logging
import asyncio  # Added for async run
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.database.reader_db_client import ReaderDBClient
# Use the enhanced GeminiClient instead of Generator
from src.gemini.gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model to use for analysis
ANALYSIS_MODEL = "models/123123123"
# Prompt file path
PROMPT_FILE = "src/prompts/step4.txt"


async def run() -> Dict[str, Any]:  # Changed to async def
    """
    Main async function to run the article analysis and grouping process.

    This function:
    1. Retrieves recently processed articles.
    2. Prepares necessary metadata for analysis.
    3. Calls the enhanced GeminiClient for thematic grouping.
    4. Validates and saves the analysis result to a timestamped JSON file.

    Returns:
        Dict[str, Any]: Status report containing metrics about the analysis process.
    """
    start_time = datetime.now()

    # Initialize status dictionary
    status = {
        "success": False,
        "input_articles_count": 0,
        "groups_generated": 0,
        "output_file": None,
        "error": None,
        "runtime_seconds": 0,
        "text_length": 0
    }

    # Initialize clients
    db_client = None
    gemini_client = None

    try:
        # Initialize database client
        db_client = ReaderDBClient()
        # Initialize the GeminiClient (API key loaded from env within GeminiClient)
        gemini_client = GeminiClient()

        # Step 1: Retrieve recently processed articles
        logger.info(
            "Fetching recently processed articles (published yesterday or today)...")
        # Note: get_recent_day_processed_articles_with_details includes domain goodness
        # which we don't need for the input, but it's the most convenient function.
        articles_raw = db_client.get_recent_day_processed_articles_with_details()

        if not articles_raw:
            logger.warning(
                "No recently processed articles found for analysis.")
            # No articles is not an error for this step
            status["success"] = True
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        status["input_articles_count"] = len(articles_raw)
        logger.info(
            f"Found {len(articles_raw)} articles to prepare for analysis.")

        # Step 2: Prepare data for Gemini analysis input
        articles_for_analysis = []
        for article in articles_raw:
            article_id = article.get('article_id')
            if not article_id:
                logger.warning(
                    f"Skipping article with missing ID in raw data: {article}")
                continue

            # Fetch detailed top entities required by the prompt
            top_entities_detail = db_client.get_top_entities_with_influence_flag(
                article_id, limit=5)  # Fetch details needed for prompt

            # Format entities for the prompt input with snippets
            entities_for_prompt = []
            for entity in top_entities_detail:
                # Only include influential entities
                if entity.get('is_influential_context', False):
                    entity_id = entity.get('entity_id')

                    # Get snippets for this entity in this article
                    entity_snippets = db_client.get_article_entity_snippets(
                        article_id, entity_id)

                    # Format the entity with its snippets
                    entity_data = {
                        "name": entity.get('name'),
                        "entity_type": entity.get('entity_type'),
                        "snippets": [snippet.get('snippet') for snippet in entity_snippets]
                    }
                    entities_for_prompt.append(entity_data)

            # Construct the dictionary with fields required by the prompt
            analysis_input_item = {
                'article_id': article_id,
                'title': article.get('title', ''),
                'domain': article.get('domain', ''),
                # Convert pub_date to ISO format string if it's a datetime object
                'pub_date': article.get('pub_date').date().isoformat() if isinstance(article.get('pub_date'), datetime) else article.get('pub_date'),
                'cluster_id': article.get('cluster_id'),
                'frame_phrases': article.get('frame_phrases', []),
                'top_entities': entities_for_prompt
            }
            articles_for_analysis.append(analysis_input_item)

        if not articles_for_analysis:
            logger.error(
                "Failed to prepare any articles for analysis (all might have been skipped).")
            status["error"] = "No articles could be prepared for analysis."
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        # Step 3: Call the GeminiClient for analysis
        # Note: The actual implementation of analyze_articles_with_prompt is in src/gemini/modules/generator.py
        logger.info(
            f"Sending {len(articles_for_analysis)} prepared articles to Gemini for analysis...")
        # Convert the articles list to a JSON string

        # Save a copy of the input data for debugging
        output_dir = "src/output/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"step4_input_data_{timestamp}.json"
        debug_path = os.path.join(output_dir, debug_filename)

        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(articles_for_analysis))
            logger.info(f"Saved input data for debugging to {debug_path}")
        except IOError as e:
            logger.warning(f"Failed to write debug input file: {e}")

        analysis_result = await gemini_client.analyze_articles_with_prompt(
            articles_data=articles_for_analysis,
            prompt_file_path=PROMPT_FILE,
            model_name=ANALYSIS_MODEL
            # Using default system instruction and other params from GeminiClient method
        )

        # Step 4: Process and validate the response
        if analysis_result is None:
            logger.error("Gemini analysis failed after retries.")
            status["error"] = "Gemini analysis call failed."
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        # The result is now a plain text string
        text_length = len(analysis_result)
        logger.info(
            f"Successfully received text analysis result from Gemini ({text_length} characters).")
        # No longer counting groups as that was JSON-specific
        status["text_length"] = text_length

        # Step 5: Save the analysis results as text
        output_dir = "src/output/"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Changed to .txt
        output_filename = f"step4_analysis_output_{timestamp}.txt"
        output_path = os.path.join(output_dir, output_filename)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result)  # Direct write, no JSON dump
            logger.info(
                f"Successfully saved Gemini analysis text to {output_path}")
        except IOError as e:
            logger.error(
                f"Failed to write analysis output to {output_path}: {e}")
            status["error"] = f"Failed to write output file: {e}"
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        # Step 6: Final success status update
        status["success"] = True
        status["output_file"] = output_path
        status["runtime_seconds"] = (
            datetime.now() - start_time).total_seconds()

        logger.info(
            f"Step 4 (Analysis) completed successfully in {status['runtime_seconds']:.2f} seconds.")
        return status

    except Exception as e:
        logger.error(
            f"Critical error in Step 4 (Analysis): {e}", exc_info=True)
        status["error"] = str(e)
        status["runtime_seconds"] = (
            datetime.now() - start_time).total_seconds()
        return status

    finally:
        # Ensure the database connection is properly closed
        if db_client:
            db_client.close()
            logger.debug("Database connection closed.")

# Updated main execution block for async
if __name__ == "__main__":
    try:
        status = asyncio.run(run())
        print(
            f"Step 4 Analysis Status: {'Success' if status['success'] else 'Failed'}")
        print(f"Input Articles: {status.get('input_articles_count', 0)}")
        if status.get('success'):
            print(f"Text Length: {status.get('text_length', 0)} characters")
            print(f"Output saved to: {status.get('output_file')}")
        if status.get('error'):
            print(f"Error: {status.get('error')}")
        print(f"Runtime: {status.get('runtime_seconds', 0):.2f} seconds")
    except KeyboardInterrupt:
        print("\nStep 4 execution interrupted by user.")
    except Exception as e:
        print(f"\nCritical error during Step 4 execution: {e}")
