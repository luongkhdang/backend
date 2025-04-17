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
import time
from typing import List, Dict, Any, Tuple, Set

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

        # Model coordination - Track model usage across tasks
        self.model_usage_counter = {
            'models/gemini-2.0-flash-thinking-exp-01-21': 0,
            'models/gemini-2.0-flash-exp': 0,
            'models/gemini-2.0-flash': 0,
            'models/gemini-2.0-flash-lite': 0
        }

        # Track which models are in cooldown due to rate limits
        self.models_in_cooldown: Set[str] = set()

        # Add lock to prevent race conditions when updating model statuses
        self.model_lock = asyncio.Lock()

    async def _remove_from_cooldown(self, model_name: str, wait_time: float) -> None:
        """
        Asynchronously remove a model from cooldown after the wait time has elapsed.

        Args:
            model_name: Name of the model to remove from cooldown
            wait_time: Time in seconds to wait before removing from cooldown
        """
        await asyncio.sleep(wait_time)
        async with self.model_lock:
            if model_name in self.models_in_cooldown:
                self.models_in_cooldown.remove(model_name)
                self.logger.info(
                    f"Model {model_name} removed from cooldown after {wait_time:.2f}s wait")

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
        content = task_data.get('content', '')
        tier = task_data.get('processing_tier', 0)
        primary_model = task_data.get('model_to_use', 'No model specified')
        fallback_model = task_data.get(
            'fallback_model', gemini_client.FALLBACK_MODEL)

        # Smart model selection - Check if primary model is in cooldown
        async with self.model_lock:
            if primary_model in self.models_in_cooldown:
                # Check wait time for primary model
                wait_time = 0
                if hasattr(gemini_client, 'rate_limiter') and gemini_client.rate_limiter:
                    wait_time = await gemini_client.rate_limiter.get_wait_time_async(primary_model)

                # If wait time is ≤ 40 seconds, wait and use primary model
                if 0 < wait_time <= 40:
                    self.logger.info(
                        f"Primary model {primary_model} in cooldown but wait time is only {wait_time:.2f}s. Waiting to use better model for article {article_id}")
                    await asyncio.sleep(wait_time)
                    # Remove from cooldown since we're waiting
                    self.models_in_cooldown.remove(primary_model)
                    model_to_use = primary_model
                else:
                    self.logger.info(
                        f"Primary model {primary_model} in cooldown with wait time > 40s, using fallback model {fallback_model} for article {article_id}")
                    model_to_use = fallback_model
            else:
                model_to_use = primary_model

        content_length = len(content) if content else 0
        self.logger.debug(
            f"Task for article {article_id}: Using {model_to_use}, tier {tier}, content length {content_length}")

        if not content or content_length == 0:
            self.logger.warning(f"Empty content for article {article_id}")
            return (article_id, {"error": "Empty content"})

        try:
            # Use GeminiClient's async method
            self.logger.debug(
                f"Starting API call for article {article_id} using tier {tier} model: {model_to_use}")

            # Add timeout to each individual API call
            try:
                # Handle rate limit hits by adding to cooldown set
                try:
                    # Already has timeout inside but let's add another layer of safety
                    extraction_result = await asyncio.wait_for(
                        gemini_client.generate_text_with_prompt_async(
                            article_content=content,
                            processing_tier=tier,
                            model_override=model_to_use,
                            fallback_model=fallback_model  # Pass the tier-specific fallback model
                        ),
                        timeout=180  # 3 minute timeout for individual call
                    )

                    # Increment usage counter on successful call
                    async with self.model_lock:
                        if model_to_use in self.model_usage_counter:
                            self.model_usage_counter[model_to_use] += 1

                except Exception as api_err:
                    # Check if the error is a rate limit error
                    error_str = str(api_err).lower()
                    is_rate_limit_error = "rate limit" in error_str or "quota" in error_str or "resource has been exhausted" in error_str

                    if is_rate_limit_error:
                        # Add model to cooldown
                        async with self.model_lock:
                            self.models_in_cooldown.add(model_to_use)

                        # Calculate wait time, default to a reasonable value if rate limiter not available
                        wait_time = 40  # Default to max wait of 40 seconds
                        if hasattr(gemini_client, 'rate_limiter') and gemini_client.rate_limiter:
                            wait_time = await gemini_client.rate_limiter.get_wait_time_async(model_to_use)
                            # Cap wait time at 40 seconds
                            # Between 10-40 seconds
                            wait_time = min(max(wait_time, 10), 40)

                        self.logger.warning(
                            f"Rate limit hit for model {model_to_use}. Adding to cooldown for {wait_time:.2f}s")

                        # Schedule removal after wait time
                        asyncio.create_task(
                            self._remove_from_cooldown(model_to_use, wait_time))

                    # Re-raise the exception to be caught by the outer try-except
                    raise api_err

                if extraction_result:
                    # Client now returns the already parsed dictionary
                    entity_count = len(extraction_result.get('entities', []))
                    self.logger.debug(
                        f"Successfully extracted {entity_count} entities for article {article_id}")
                    return (article_id, extraction_result)
                else:
                    self.logger.warning(
                        f"No entity extraction result from Gemini for article {article_id}")
                    return (article_id, {
                        "error": "No response from API"
                    })
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Task timeout for article {article_id} using model {model_to_use}")
                return (article_id, {
                    "error": f"Task timeout after 180 seconds"
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
        task_article_ids = []  # Track article_ids in the same order as awaitables
        for i, task_data in enumerate(tasks_definitions):
            if not task_data.get('article_id'):
                self.logger.warning("Skipping task without article_id")
                continue

            article_id = task_data.get('article_id')
            task_article_ids.append(article_id)  # Store article_id
            model = task_data.get('model_to_use', 'default')
            self.logger.debug(
                f"Creating task {i+1}/{len(tasks_definitions)} for article {article_id} with model {model}")

            # Create coroutine for each task
            awaitables.append(self._run_single_task(gemini_client, task_data))

        if not awaitables:
            self.logger.warning("No valid tasks to run")
            return {}

        # Staggered task execution
        self.logger.info(
            f"Running {len(awaitables)} tasks with staggered execution")
        results = []
        for i, task in enumerate(awaitables):
            # Add small delay between task starts to prevent concurrent rate limit hits
            if i > 0:
                await asyncio.sleep(0.5)  # 500ms stagger between task starts

            self.logger.debug(
                f"Starting task {i+1}/{len(awaitables)} for article {task_article_ids[i]}")
            try:
                # Wait for the task with timeout
                result = await asyncio.wait_for(task, timeout=180)
                results.append(result)
            except asyncio.TimeoutError:
                # Handle timeout for individual task
                self.logger.error(
                    f"Task {i+1} for article {task_article_ids[i]} timed out")
                results.append(
                    (task_article_ids[i], {"error": "Task timeout"}))
            except Exception as e:
                # Handle other exceptions
                self.logger.error(
                    f"Task {i+1} for article {task_article_ids[i]} failed: {e}")
                results.append(
                    (task_article_ids[i], {"error": f"Task failed: {str(e)}"}))

        # Process results
        final_results = {}
        error_count = 0

        for result in results:
            # Process normal result
            if isinstance(result, tuple) and len(result) == 2:
                article_id, data = result
                self.logger.debug(f"Processed result for article {article_id}")
                final_results[article_id] = data

                # Check if this is an error
                if isinstance(data, dict) and 'error' in data:
                    error_count += 1
            else:
                self.logger.error(f"Unexpected result format: {result}")
                # We can't identify the article ID for this result
                continue

        success_count = len(final_results) - error_count
        self.logger.info(
            f"Processed {len(final_results)} results: {success_count} successful, {error_count} errors")

        # Log the model usage stats
        self.logger.info(f"Model usage statistics: {self.model_usage_counter}")
        self.logger.info(f"Models in cooldown: {self.models_in_cooldown}")

        # Return any results we have, even if some failed
        return final_results
