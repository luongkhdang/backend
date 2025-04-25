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
        article_id = task_data.get('id')
        content = task_data.get('content', '')
        tier = task_data.get('processing_tier', 0)
        model_override = task_data.get('model_to_use')

        # Log entry point with essential details
        self.logger.debug(
            f"[_run_single_task Entry] article_id={article_id}, tier={tier}, model_override='{model_override}'")

        # --- Enhanced Check for article_id ---
        if not article_id:
            self.logger.error(
                f"[_run_single_task Error] Task missing article ID (key 'id'). Task data: {task_data}")
            # Log exit point for error case
            self.logger.debug(
                f"[_run_single_task Exit - Error] article_id=None, returning error dict")
            return (-1, {"error": "Task missing article ID (key 'id')"})
        # --- End Enhanced Check ---

        content_length = len(content) if content else 0
        if not content or content_length == 0:
            self.logger.warning(
                f"[_run_single_task Warning] Empty content for article {article_id}. Returning error dict.")
            # Log exit point for warning/error case
            self.logger.debug(
                f"[_run_single_task Exit - Error] article_id={article_id}, returning error dict")
            return (article_id, {"error": "Empty content"})

        try:
            # Log before the actual API call attempt
            log_model = model_override if model_override else "client-selected"
            self.logger.info(
                f"[_run_single_task] Attempting API call for article {article_id} (model: {log_model}, tier: {tier}) via GeminiClient")

            start_time = time.monotonic()

            # Add timeout for the entire client call (including its internal retries)
            extraction_result = await asyncio.wait_for(
                gemini_client.generate_text_with_prompt_async(
                    article_content=content,
                    processing_tier=tier,
                    model_override=model_override
                ),
                timeout=180  # 3 minute timeout for the overall task attempt
            )

            end_time = time.monotonic()
            duration = end_time - start_time
            self.logger.info(
                f"[_run_single_task] API call for article {article_id} completed in {duration:.2f}s.")

            if extraction_result:
                entity_count = len(extraction_result.get('entities', []))
                self.logger.debug(
                    f"[_run_single_task Success] Processed article {article_id}, found {entity_count} entities.")
                # Log exit point for success case
                self.logger.debug(
                    f"[_run_single_task Exit - Success] article_id={article_id}, returning data")
                return (article_id, extraction_result)
            else:
                self.logger.warning(
                    f"[_run_single_task Warning] GeminiClient returned None/empty result for article {article_id}")
                # Log exit point for warning/error case
                self.logger.debug(
                    f"[_run_single_task Exit - Error] article_id={article_id}, returning error dict")
                return (article_id, {
                    "error": "Client returned no result (check GeminiClient logs)"
                })

        except asyncio.TimeoutError:
            elapsed_time = time.monotonic() - start_time
            self.logger.error(
                f"[_run_single_task Error] Overall task timeout for article {article_id} after {elapsed_time:.2f}s (limit 180s)")
            # Log exit point for timeout error
            self.logger.debug(
                f"[_run_single_task Exit - Error] article_id={article_id}, returning error dict")
            return (article_id, {
                "error": f"Task timeout after 180 seconds"
            })
        except google.api_core.exceptions.ResourceExhausted as rate_limit_err:
            elapsed_time = time.monotonic() - start_time
            self.logger.error(
                f"[_run_single_task Error] Gemini API rate limit/quota error for article {article_id} after {elapsed_time:.2f}s: {rate_limit_err}", exc_info=False)
            # Log exit point for rate limit error
            self.logger.debug(
                f"[_run_single_task Exit - Error] article_id={article_id}, returning error dict")
            return (article_id, {
                "error": f"API Rate Limit/Quota Error: {str(rate_limit_err)}"
            })
        except google.api_core.exceptions.GoogleAPIError as api_err:
            elapsed_time = time.monotonic() - start_time
            self.logger.error(
                f"[_run_single_task Error] Google API error for article {article_id} after {elapsed_time:.2f}s: {api_err}", exc_info=True)
            # Log exit point for Google API error
            self.logger.debug(
                f"[_run_single_task Exit - Error] article_id={article_id}, returning error dict")
            return (article_id, {
                "error": f"Google API Error: {str(api_err)}"
            })
        except Exception as e:
            elapsed_time = time.monotonic() - start_time
            # Catch any other unexpected errors
            self.logger.error(
                f"[_run_single_task Error] Unexpected error processing article {article_id} after {elapsed_time:.2f}s: {e}", exc_info=True)
            # Log exit point for unexpected error
            self.logger.debug(
                f"[_run_single_task Exit - Error] article_id={article_id}, returning error dict")
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
        num_tasks_received = len(tasks_definitions)
        self.logger.debug(
            f"[run_tasks Entry] Received {num_tasks_received} task definitions.")

        if not tasks_definitions:
            self.logger.warning(
                "[run_tasks] No tasks provided. Returning empty dict.")
            return {}

        # Create list of coroutines
        awaitables = []
        valid_task_indices = []  # Keep track of original indices of valid tasks
        skipped_tasks_count = 0

        self.logger.debug(
            f"[run_tasks] Preparing awaitables for {num_tasks_received} tasks.")
        for i, task_data in enumerate(tasks_definitions):
            article_id_check = task_data.get('id')
            if not article_id_check:
                self.logger.warning(
                    f"[run_tasks] Skipping task at index {i} due to missing/falsy ID (key 'id'). Data: {task_data}")
                skipped_tasks_count += 1
                continue

            # Create coroutine for each valid task
            awaitables.append(self._run_single_task(gemini_client, task_data))
            valid_task_indices.append(i)  # Store the original index

        num_awaitables = len(awaitables)
        if not awaitables:
            self.logger.warning(
                f"[run_tasks] No valid tasks to run after checking {num_tasks_received} definitions (skipped {skipped_tasks_count}). Returning empty dict.")
            return {}

        self.logger.info(
            f"[run_tasks] Starting asyncio.gather for {num_awaitables} tasks (skipped {skipped_tasks_count}).")
        start_gather_time = time.monotonic()

        # Run tasks concurrently using asyncio.gather
        results = await asyncio.gather(*awaitables, return_exceptions=True)

        end_gather_time = time.monotonic()
        gather_duration = end_gather_time - start_gather_time
        num_results = len(results)
        self.logger.info(
            f"[run_tasks] asyncio.gather completed in {gather_duration:.2f} seconds.")

        # Process results and handle potential exceptions from gather
        final_results = {}
        exceptions_count = 0
        successful_count = 0
        error_results_count = 0

        self.logger.debug(
            f"[run_tasks] Processing {num_results} results from asyncio.gather.")
        for i, gather_result in enumerate(results):
            original_task_index = valid_task_indices[i]
            original_task_data = tasks_definitions[original_task_index]
            # Use a default (-1 or similar) if ID is somehow missing again, though it was checked before
            article_id_from_original = original_task_data.get(
                'id', f'MISSING_ID_at_index_{original_task_index}')

            if isinstance(gather_result, Exception):
                # This exception was raised within _run_single_task OR asyncio.gather itself
                exceptions_count += 1
                self.logger.error(
                    # Log stack trace if it's an actual exception object
                    f"[run_tasks] Task for article_id '{article_id_from_original}' (original index {original_task_index}) resulted in an exception caught by gather: {gather_result}", exc_info=(isinstance(gather_result, Exception)))
                final_results[article_id_from_original] = {
                    "error": f"Task failed with exception: {str(gather_result)}"}
            else:
                # gather_result should be the tuple (article_id, result_dict) from _run_single_task
                task_article_id, task_result = gather_result
                # Verify the article ID matches if possible (handle potential -1 case)
                if task_article_id != -1 and task_article_id != article_id_from_original:
                    self.logger.warning(
                        f"[run_tasks] Mismatch between original article ID ({article_id_from_original}) and task result ID ({task_article_id}). Using original ID for key.")
                elif task_article_id == -1:
                    self.logger.warning(
                        f"[run_tasks] Task result for original article ID ({article_id_from_original}) indicates an internal error (ID -1). Storing error under original ID.")

                # Check if the result dictionary contains an 'error' key
                if isinstance(task_result, dict) and 'error' in task_result:
                    error_results_count += 1
                    self.logger.warning(
                        f"[run_tasks] Task for article_id '{article_id_from_original}' completed with an error state: {task_result['error']}")
                else:
                    successful_count += 1
                    # self.logger.debug(f"[run_tasks] Task for article_id '{article_id_from_original}' completed successfully.") # Optional: too verbose?

                # Store result using original ID as key
                final_results[article_id_from_original] = task_result

        self.logger.info(
            f"[run_tasks Exit] Processed {num_results} results: {successful_count} successful, {error_results_count} with errors, {exceptions_count} exceptions caught by gather.")
        return final_results
