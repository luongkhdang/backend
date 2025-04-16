"""
task_manager.py - Async Task Manager for Concurrent API Calls

This module provides a TaskManager class for managing the concurrent execution of 
asynchronous tasks, particularly for Gemini API calls in the entity extraction process.

Exported classes:
- TaskManager: Manages concurrent execution of asynchronous tasks
  - run_tasks(gemini_client, tasks_definitions): Runs multiple tasks concurrently
  
Related files:
- src/steps/step3/__init__.py: Uses TaskManager for concurrent entity extraction
- src/gemini/gemini_client.py: Provides the API client used by TaskManager
"""

import logging
import asyncio
import json
from typing import List, Dict, Any, Tuple

# For type hints
from src.gemini.gemini_client import GeminiClient

# Configure logging
logger = logging.getLogger(__name__)


class TaskManager:
    """
    TaskManager handles concurrent execution of asynchronous tasks,
    particularly for Gemini API calls in the entity extraction process.
    """

    def __init__(self):
        """Initialize the TaskManager."""
        self.logger = logging.getLogger(__name__)
        self.logger.debug("TaskManager initialized")

    async def _run_single_task(self, gemini_client: GeminiClient, task_data: Dict[str, Any]) -> Tuple[int, Any]:
        """
        Execute a single Gemini API call for one article.

        Args:
            gemini_client: The GeminiClient instance to use for the API call
            task_data: Dictionary containing task information (article_id, content, tier, model_to_use)

        Returns:
            Tuple[int, Any]: Article ID and the result (parsed entities or error dictionary)
        """
        # Extract necessary data
        article_id = task_data.get('article_id')
        content = task_data.get('content')
        tier = task_data.get('processing_tier')
        model_to_use = task_data.get('model_to_use')

        try:
            # Use GeminiClient's async method
            self.logger.debug(
                f"Starting API call for article {article_id} using tier {tier} model: {model_to_use}")
            extraction_result = await gemini_client.generate_text_with_prompt_async(
                article_content=content,
                processing_tier=tier,
                model_override=model_to_use
            )

            if extraction_result:
                # Attempt to parse the JSON string response
                try:
                    parsed_entities = json.loads(extraction_result)
                    self.logger.debug(
                        f"Successfully extracted entities for article {article_id}")
                    return (article_id, parsed_entities)
                except json.JSONDecodeError as json_err:
                    self.logger.warning(
                        f"Failed to parse JSON from Gemini for article {article_id}: {json_err}")
                    return (article_id, {
                        "error": "Invalid JSON response",
                        "raw_response": extraction_result
                    })
            else:
                self.logger.warning(
                    f"No entity extraction result from Gemini for article {article_id}")
                return (article_id, {
                    "error": "No response from API"
                })

        except Exception as api_err:
            # Log the full exception for debugging
            self.logger.error(
                f"Error calling Gemini API for article {article_id}: {api_err}", exc_info=True)
            return (article_id, {
                "error": f"API call failed: {str(api_err)}"
            })

    async def run_tasks(self, gemini_client: GeminiClient, tasks_definitions: List[Dict[str, Any]]) -> Dict[int, Any]:
        """
        Run multiple entity extraction tasks concurrently.

        Args:
            gemini_client: The GeminiClient instance to use
            tasks_definitions: List of task definitions, each a dict with article info

        Returns:
            Dict[int, Any]: Dictionary mapping article IDs to their results
        """
        if not tasks_definitions:
            self.logger.warning("No tasks provided to run_tasks")
            return {}

        self.logger.info(
            f"Preparing to run {len(tasks_definitions)} tasks concurrently")

        # Create list of coroutines
        awaitables = []
        for task_data in tasks_definitions:
            if not task_data.get('article_id'):
                self.logger.warning("Skipping task without article_id")
                continue

            # Create coroutine for each task
            awaitables.append(self._run_single_task(gemini_client, task_data))

        if not awaitables:
            self.logger.warning("No valid tasks to run")
            return {}

        # Run all tasks concurrently
        self.logger.info(f"Running {len(awaitables)} tasks concurrently")
        results = await asyncio.gather(*awaitables, return_exceptions=True)
        self.logger.info("Concurrent task execution completed")

        # Process results
        final_results = {}
        for result in results:
            # Handle exceptions from gather if return_exceptions=True
            if isinstance(result, Exception):
                self.logger.error(f"Task failed with exception: {result}")
                continue

            # Process normal result
            if isinstance(result, tuple) and len(result) == 2:
                article_id, data = result
                final_results[article_id] = data
            else:
                self.logger.error(f"Unexpected result format: {result}")

        self.logger.info(
            f"Processed {len(final_results)} results out of {len(awaitables)} tasks")
        return final_results
