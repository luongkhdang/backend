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
ANALYSIS_MODEL = "models/gemini-2.5-flash-preview-04-17"
# Remove PROMPT_FILE constant, load directly
# PROMPT_FILE = "src/prompts/step4.txt"
PROMPT_FILE_PATH = "src/prompts/step4.txt"  # Keep path for loading


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
            article_id = article.get('id')
            if not article_id:
                logger.warning(
                    f"Skipping article with missing ID in raw data: {article}")
                continue

            # Fetch detailed top entities required by the prompt
            top_entities_detail = db_client.get_top_entities_with_influence_flag(
                article_id, limit=5)  # Fetch details needed for prompt

            # Format entities for the prompt input with snippets
            entities_for_prompt = []
            influential_count = 0  # Counter for influential entities

            # Check if this article is hot
            is_hot = article.get('is_hot', False)

            for entity in top_entities_detail:
                # Only include influential entities and limit to top 2
                if entity.get('is_influential_context', False) and influential_count < 2:
                    entity_id = entity.get('entity_id')

                    # Format the entity data
                    entity_data = [
                        entity.get('name'),
                        entity.get('entity_type')
                    ]

                    # Only get snippets if the article is hot
                    if is_hot:
                        # Get snippets for this entity in this article
                        entity_snippets = db_client.get_article_entity_snippets(
                            article_id, entity_id)
                        # Add snippets as the third element
                        entity_data.append([snippet.get('snippet')
                                           for snippet in entity_snippets])
                    else:
                        # For non-hot articles, add an empty array for snippets
                        entity_data.append([])

                    entities_for_prompt.append(entity_data)
                    influential_count += 1  # Increment counter after processing an influential entity

            # Convert pub_date to ISO format string if it's a datetime object
            pub_date = article.get('pub_date')
            if isinstance(pub_date, datetime):
                pub_date = pub_date.date().isoformat()
            else:
                pub_date = article.get('pub_date', '')

            # Construct the list item with fields in positional order
            analysis_input_item = {
                'article_id': article_id,
                'title': article.get('title', ''),
                'domain': article.get('domain', ''),
                'content_length': article.get('content_length', 0),
                'pub_date': pub_date,
                'cluster_id': article.get('cluster_id', ''),
                'frame_phrases': article.get('frame_phrases', []),
                'main_entities': entities_for_prompt,
            }
            articles_for_analysis.append(analysis_input_item)

        if not articles_for_analysis:
            logger.error(
                "Failed to prepare any articles for analysis (all might have been skipped).")
            status["error"] = "No articles could be prepared for analysis."
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        # --- Step 3 Preparation: Load prompt, inject date, prepare final text --- #
        logger.info(f"Loading prompt template from {PROMPT_FILE_PATH}")
        try:
            with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            if not prompt_template:
                raise ValueError("Prompt template file is empty.")
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Failed to load prompt template: {e}")
            status["error"] = f"Failed to load prompt template: {e}"
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        # Inject today's date
        today_date_str = datetime.now().strftime("%Y-%m-%d")
        prompt_template = prompt_template.replace(
            "{TODAY_DATE}", today_date_str)
        logger.debug("Injected today's date into prompt template.")

        # Prepare article data JSON
        try:
            articles_json = json.dumps(
                articles_for_analysis, separators=(',', ':'), ensure_ascii=True)
        except (TypeError, OverflowError) as e:
            logger.error(f"Failed to serialize article data to JSON: {e}")
            status["error"] = f"Failed to serialize article data: {e}"
            status["runtime_seconds"] = (
                datetime.now() - start_time).total_seconds()
            return status

        # Inject article data JSON
        full_prompt_text = prompt_template.replace(
            "{INPUT_DATA_JSON}", articles_json)
        logger.debug(
            f"Prepared full prompt text (length: {len(full_prompt_text)} chars).")

        # Extract System Instruction/Persona from the loaded prompt template
        # (Assuming it's defined between "## Persona:" and "Input Data Format:")
        try:
            persona_start = prompt_template.find("## Persona:")
            persona_end = prompt_template.find("Input Data Format:")
            if persona_start != -1 and persona_end != -1 and persona_end > persona_start:
                step4_system_instruction = prompt_template[persona_start + len(
                    "## Persona:"):persona_end].strip()
                logger.debug("Extracted Step 4 system instruction/persona.")
            else:
                logger.warning(
                    "Could not extract specific system instruction for Step 4 from prompt file. Using default.")
                step4_system_instruction = None  # Let GeminiClient use its default
        except Exception as e:
            logger.warning(
                f"Error extracting system instruction: {e}. Using default.")
            step4_system_instruction = None

        # Save a copy of the final prompt for debugging
        output_dir = "src/output/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_filename = f"step4_final_prompt_{timestamp}.txt"
        debug_path = os.path.join(output_dir, debug_filename)
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(full_prompt_text)
            logger.info(
                f"Saved final prompt text for debugging to {debug_path}")
        except IOError as e:
            logger.warning(f"Failed to write debug prompt file: {e}")

        # --- Step 4: Call the GeminiClient for analysis using the prepared full prompt --- #
        logger.info(
            f"Sending final prompt (length: {len(full_prompt_text)}) to Gemini for analysis...")

        # Use generate_essay_from_prompt as it accepts a full prompt string
        # Pass the extracted system instruction
        analysis_result = await gemini_client.generate_essay_from_prompt(
            full_prompt_text=full_prompt_text,
            model_name=ANALYSIS_MODEL,
            system_instruction=step4_system_instruction,  # Pass specific instruction
            temperature=0.2,  # Set temperature appropriate for analysis
            save_debug_info=False  # Already saved the prompt above
        )

        # Step 5: Process and validate the response
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

        # Step 6: Final success status update
        status["success"] = True
        status["output_file"] = debug_path
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
