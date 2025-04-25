"""
task_manager.py - Async Task Manager for Concurrent API Calls

This module provides a TaskManager class for managing the concurrent execution of 
asynchronous tasks, particularly for Gemini API calls in the entity extraction process.
It relies on the passed GeminiClient instance to handle API specifics like model selection,
fallback, retries, and rate limiting.

Exported classes:
- TaskManager: Manages concurrent execution of asynchronous tasks
  - run_tasks(gemini_client, tasks_definitions): Runs multiple tasks concurrently using asyncio.gather
  
Related files:
- src/steps/step3/__init__.py: Uses TaskManager for concurrent entity extraction
- src/gemini/gemini_client.py: Provides the API client used by TaskManager
"""

import logging
import asyncio
import json
import time
from typing import List, Dict, Any, Tuple

# Import google exceptions for error handling if needed
import google.api_core.exceptions

# For type hints
from src.gemini.gemini_client import GeminiClient

# Configure logging
logger = logging.getLogger(__name__)


class TaskManager:
    """
    TaskManager handles concurrent execution of asynchronous tasks,
    particularly for Gemini API calls in the entity extraction process.
    It now relies on the GeminiClient to handle model selection, fallback, and rate limiting.
    """

    def __init__(self):
        """Initialize the TaskManager."""
        self.logger = logging.getLogger(__name__)
        self.logger.debug("TaskManager initialized")

    async def _run_single_task(self, gemini_client: GeminiClient, task_data: Dict[str, Any]) -> Tuple[int, Any]:
        """
        Execute a single Gemini API call for one article using the refactored GeminiClient.

        Args:
            gemini_client: The GeminiClient instance to use for the API call
            task_data: Dictionary containing task information (article_id, content, tier, model_to_use [optional override])

        Returns:
            Tuple[int, Any]: Article ID and the result (parsed entities or error dictionary)
        """
        article_id = task_data.get('article_id')
        content = task_data.get('content', '')
        tier = task_data.get('processing_tier', 0)
        # Model override is optional; client selects if not provided
        model_override = task_data.get('model_to_use')

        content_length = len(content) if content else 0
        log_model = model_override if model_override else "client-selected"
        self.logger.debug(
            f"Task for article {article_id}: Using {log_model} model (tier {tier}), content length {content_length}")

        if not article_id:
            self.logger.error("Task missing article_id")
            # Return a placeholder or raise an error? Returning error dict for now.
            return (-1, {"error": "Task missing article_id"})

        if not content or content_length == 0:
            self.logger.warning(f"Empty content for article {article_id}")
            return (article_id, {"error": "Empty content"})

        try:
            # Call the GeminiClient's async method directly.
            # It handles model selection, fallback, retries, and rate limiting internally.
            self.logger.debug(
                f"Starting API call via GeminiClient for article {article_id}")

            # Add timeout for the entire client call (including its internal retries)
            extraction_result = await asyncio.wait_for(
                gemini_client.generate_text_with_prompt_async(
                    article_content=content,
                    processing_tier=tier,
                    model_override=model_override
                    # Removed fallback_model - client handles this
                ),
                timeout=180  # 3 minute timeout for the overall task attempt
            )

            # Removed internal rate limit check/cooldown logic

            if extraction_result:
                # Client returns the already parsed dictionary
                entity_count = len(extraction_result.get('entities', []))
                self.logger.debug(
                    f"Successfully processed article {article_id}, found {entity_count} entities.")
                return (article_id, extraction_result)
            else:
                # This case might indicate an issue within the client or an unexpected empty response
                self.logger.warning(
                    f"GeminiClient returned None or empty result for article {article_id}")
                # Check client logs for more details if this occurs
                return (article_id, {
                    "error": "Client returned no result (check GeminiClient logs)"
                })

        except asyncio.TimeoutError:
            self.logger.error(
                f"Overall task timeout for article {article_id} after 180 seconds")
            return (article_id, {
                "error": f"Task timeout after 180 seconds"
            })
        # Catch specific Google API errors if needed for distinct handling
        except google.api_core.exceptions.ResourceExhausted as rate_limit_err:
            self.logger.error(
                f"Gemini API rate limit/quota error for article {article_id}: {rate_limit_err}", exc_info=False)
            return (article_id, {
                "error": f"API Rate Limit/Quota Error: {str(rate_limit_err)}"
            })
        except google.api_core.exceptions.GoogleAPIError as api_err:
            self.logger.error(
                f"Google API error for article {article_id}: {api_err}", exc_info=True)
            return (article_id, {
                "error": f"Google API Error: {str(api_err)}"
            })
        except Exception as e:
            # Catch any other unexpected errors during the client call
            self.logger.error(
                f"Unexpected error processing article {article_id}: {e}", exc_info=True)
            return (article_id, {
                "error": f"Unexpected task error: {str(e)}"
            })

    async def run_tasks(self, gemini_client: GeminiClient, tasks_definitions: List[Dict[str, Any]]) -> Dict[int, Any]:
        """
        Run multiple entity extraction tasks concurrently using asyncio.gather.

        Args:
            gemini_client: The GeminiClient instance to use
            tasks_definitions: List of task definitions, each a dict with article info

        Returns:
            Dict[int, Any]: Dictionary mapping article IDs to their results (or error dicts)
        """
        if not tasks_definitions:
            self.logger.warning("No tasks provided to run_tasks")
            return {}

        self.logger.info(
            f"Preparing to run {len(tasks_definitions)} tasks concurrently using asyncio.gather")

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

        # Run tasks concurrently using asyncio.gather
        # return_exceptions=True allows us to capture errors from individual tasks
        # without stopping the entire batch.
        results = await asyncio.gather(*awaitables, return_exceptions=True)

        # Process results and handle potential exceptions from gather
        final_results = {}
        task_idx = 0
        valid_task_defs = [t for t in tasks_definitions if t.get(
            'article_id')]  # Keep track of original task defs

        for result in results:
            original_task_data = valid_task_defs[task_idx]
            article_id = original_task_data.get(
                'article_id', -1)  # Should always have ID here

            if isinstance(result, Exception):
                # An unexpected exception occurred within asyncio.gather or the task itself
                self.logger.error(
                    f"Task for article {article_id} failed with exception in gather: {result}", exc_info=result)
                final_results[article_id] = {
                    "error": f"Task failed in gather: {str(result)}"}
            elif isinstance(result, tuple) and len(result) == 2:
                # Normal result format: (article_id, result_data)
                res_article_id, res_data = result
                if res_article_id != article_id and article_id != -1:
                    # Log mismatch, but use ID from result if valid
                    self.logger.warning(
                        f"Article ID mismatch! Task def ID: {article_id}, Result ID: {res_article_id}. Using result ID.")
                    final_results[res_article_id] = res_data
                elif res_article_id == -1 and article_id != -1:
                    # Result indicates error before ID was confirmed, use task def ID
                    final_results[article_id] = res_data
                else:  # IDs match or task def ID was missing
                    final_results[res_article_id] = res_data
            else:
                # Unexpected result format from _run_single_task
                self.logger.error(
                    f"Unexpected result format for article {article_id}: {result}")
                final_results[article_id] = {
                    "error": "Unexpected result format from task"}

            task_idx += 1

        self.logger.info(f"Finished processing {len(results)} tasks.")
        # Log summary of model usage from GeminiClient's perspective if possible/needed
        # (Requires GeminiClient to expose this info or log it effectively)

        return final_results
