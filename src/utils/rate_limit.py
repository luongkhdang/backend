"""
rate_limit.py - Simple Rate Limiter for API Calls

This module provides a basic rate limiter to manage API call frequency,
primarily based on Requests Per Minute (RPM) for different models.

Exported Classes:
- RateLimiter: Tracks API calls per minute for different models and enforces limits.
    - __init__(self, model_configs: Dict[str, int])
    - is_allowed(self, model_name: str) -> bool
    - register_call(self, model_name: str) -> None
    - wait_if_needed(self, model_name: str) -> None: Synchronous wait method
    - async wait_if_needed_async(self, model_name: str) -> None: Asynchronous wait method
    - async register_call_async(self, model_name: str) -> None: Asynchronous register method

Related Files:
- src/gemini/gemini_client.py: Uses this rate limiter.
"""

import time
import logging
import asyncio
from collections import defaultdict, deque
from typing import Dict, Deque
import threading

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimiter:
    """Manages rate limits for different API models based on RPM."""

    def __init__(self, model_rpm_limits: Dict[str, int]):
        """
        Initializes the RateLimiter.

        Args:
            model_rpm_limits (Dict[str, int]): A dictionary mapping model names to their RPM limits.
                                               Example: {'models/gemini-2.0-flash': 15}
        """
        self.model_rpm_limits = model_rpm_limits
        # Store timestamps of recent calls for each model
        self.call_timestamps: Dict[str, Deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()  # Lock for thread safety
        logger.info(
            f"RateLimiter initialized with RPM limits: {model_rpm_limits}")

    def get_current_rpm(self, model_name: str) -> int:
        """Calculates the approximate current RPM based on timestamps."""
        with self.lock:
            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]
            # Remove timestamps older than 60 seconds
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()
            return len(timestamps)

    def get_wait_time(self, model_name: str) -> float:
        """
        Calculates the time to wait until a call is allowed for the specified model.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            float: The time in seconds to wait. Returns 0.0 if no wait is needed
                  or if the model has no configured limit.
        """
        with self.lock:
            # Check if there's a limit for this model
            limit = self.model_rpm_limits.get(model_name)
            if limit is None:
                logger.debug(
                    f"RateLimiter [{model_name}]: No limit configured, no wait time needed.")
                return 0.0  # No wait needed if no limit is defined

            # Clean up old timestamps
            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]

            # Remove timestamps older than 60 seconds
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()

            # Check if the number of calls in the last 60 seconds exceeds the limit
            current_count = len(timestamps)
            if current_count < limit:
                # Call is allowed, no wait needed
                return 0.0

            if not timestamps:  # Should not happen if current_count >= limit, but safe check
                logger.debug(
                    f"RateLimiter [{model_name}]: No timestamps found during wait time calculation.")
                return 0.0

            # Calculate how long to wait until the oldest call expires
            oldest_call_time = timestamps[0]
            wait_time = (oldest_call_time + 60) - \
                current_time + 0.1  # Add small buffer

            logger.debug(
                f"RateLimiter [{model_name}]: Wait time calculation: {max(0, wait_time):.2f}s until next slot.")
            return max(0, wait_time)  # Ensure non-negative

    async def get_wait_time_async(self, model_name: str) -> float:
        """
        Asynchronous version of get_wait_time.
        Calculates the time to wait until a call is allowed for the specified model.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            float: The time in seconds to wait. Returns 0.0 if no wait is needed
                  or if the model has no configured limit.
        """
        # For the async version, we just call the sync version since the calculation is quick
        return self.get_wait_time(model_name)

    def is_allowed(self, model_name: str) -> bool:
        """
        Checks if a call to the specified model is allowed based on RPM limit.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the call is allowed, False otherwise.
        """
        with self.lock:
            limit = self.model_rpm_limits.get(model_name)
            if limit is None:
                logger.warning(
                    f"No RPM limit configured for model '{model_name}'. Allowing call.")
                return True  # Allow if no limit is defined for the model

            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]

            # Remove timestamps older than 60 seconds
            # Use a copy for logging to avoid modifying while iterating/checking
            original_count = len(timestamps)
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()
            removed_count = original_count - len(timestamps)
            if removed_count > 0:
                logger.debug(
                    f"RateLimiter [{model_name}]: Removed {removed_count} old timestamps.")

            # Check if the number of calls in the last 60 seconds exceeds the limit
            current_count = len(timestamps)
            allowed = current_count < limit
            logger.debug(
                f"RateLimiter [{model_name}]: Check: {current_count} calls / {limit} RPM limit -> Allowed: {allowed}")

            if allowed:
                return True
            else:
                # Optional: Log when rate limit is hit
                # logger.debug(f"RPM limit reached for model '{model_name}'. Call blocked.")
                return False

    def register_call(self, model_name: str) -> None:
        """
        Registers a successful call for the specified model.

        Args:
            model_name (str): The name of the model for which a call was made.
        """
        with self.lock:
            limit = self.model_rpm_limits.get(model_name)
            if limit is None:
                # Don't track calls for models without limits
                logger.debug(
                    f"RateLimiter [{model_name}]: No limit configured, not registering call.")
                return

            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]

            # Remove old timestamps (consistency check)
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()

            # Add the current timestamp
            timestamps.append(current_time)
            # Optional: Log call registration
            logger.debug(
                f"RateLimiter [{model_name}]: Registered call. Current count in window: {len(timestamps)} / {limit} RPM")

    async def register_call_async(self, model_name: str) -> None:
        """
        Asynchronously registers a successful call for the specified model.
        This is functionally identical to register_call but matches the async interface pattern.

        Args:
            model_name (str): The name of the model for which a call was made.
        """
        # Since this operation is quick and just involves updating internal state,
        # we can simply call the synchronous version
        self.register_call(model_name)

    def wait_if_needed(self, model_name: str) -> None:
        """
        Synchronously blocks until a call is allowed for the specified model.

        Args:
            model_name (str): The name of the model to wait for.
        """
        while not self.is_allowed(model_name):
            with self.lock:
                timestamps = self.call_timestamps[model_name]
                if not timestamps:  # Should not happen if is_allowed is False, but safe check
                    logger.debug(
                        f"RateLimiter [{model_name}]: No timestamps found while waiting, breaking wait loop.")
                    break
                # Calculate how long to wait until the oldest call expires
                oldest_call_time = timestamps[0]
                current_time_in_wait = time.monotonic()
                wait_time = (oldest_call_time + 60) - \
                    current_time_in_wait + 0.1  # Add small buffer

                # Recalculate inside lock scope if needed, or pass value
                current_rpm = self.get_current_rpm(model_name)
                logger.debug(
                    f"RateLimiter [{model_name}]: Limit reached ({current_rpm} RPM). Waiting for {max(0, wait_time):.2f} seconds (until {oldest_call_time + 60:.2f}).")
                time.sleep(max(0, wait_time))
                # Re-check is_allowed in the next loop iteration

    async def wait_if_needed_async(self, model_name: str) -> None:
        """
        Asynchronous version of wait_if_needed.
        Waits the required time if the model's rate limit has been reached.

        Args:
            model_name (str): The name of the model to check.
        """
        while not self.is_allowed(model_name):
            with self.lock:
                timestamps = self.call_timestamps[model_name]
                if not timestamps:  # Should not happen if is_allowed is False, but safe check
                    logger.debug(
                        f"RateLimiter [{model_name}]: No timestamps found while waiting, breaking wait loop.")
                    break
                # Calculate how long to wait until the oldest call expires
                oldest_call_time = timestamps[0]
                current_time_in_wait = time.monotonic()
                wait_time = (oldest_call_time + 60) - \
                    current_time_in_wait + 0.1  # Add small buffer

                # Recalculate inside lock scope if needed, or pass value
                current_rpm = self.get_current_rpm(model_name)
                logger.debug(
                    f"RateLimiter [{model_name}]: Limit reached ({current_rpm} RPM). Waiting for {max(0, wait_time):.2f} seconds (until {oldest_call_time + 60:.2f}).")
                await asyncio.sleep(max(0, wait_time))
                # Re-check is_allowed in the next loop iteration
