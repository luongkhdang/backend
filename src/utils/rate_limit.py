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
    - async reserve_and_register_call_async(self, model_name: str) -> bool: Atomically check and register a call

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

    def __init__(self, model_rpm_limits: Dict[str, int], client_instance=None):
        """
        Initializes the RateLimiter.

        Args:
            model_rpm_limits (Dict[str, int]): A dictionary mapping model names to their RPM limits.
                                               Example: {'models/gemini-2.0-flash': 15}
            client_instance: Optional reference to the GeminiClient instance for fallback logic.
        """
        self.model_rpm_limits = model_rpm_limits
        self.client_instance = client_instance  # Store client reference
        # Store timestamps of recent calls for each model
        self.call_timestamps: Dict[str, Deque[float]] = defaultdict(deque)
        # Use RLock for reentrancy
        self.lock = threading.RLock()
        logger.debug(
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
                # logger.debug(f"RateLimiter [{model_name}]: No limit configured, no wait time needed.") # Removed debug log
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
                logger.warning(
                    # Changed to WARNING
                    f"RateLimiter [{model_name}]: No timestamps found during wait time calculation.")
                return 0.0

            # Calculate how long to wait until the oldest call expires
            oldest_call_time = timestamps[0]
            wait_time = (oldest_call_time + 60) - \
                current_time + 0.1  # Add small buffer

            # Changed to TRACE (or remove if no TRACE level)
            # logger.debug(
            #     f"RateLimiter [{model_name}]: Wait time calculation: {max(0, wait_time):.2f}s until next slot.")
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
        Checks if a call to the specified model is allowed based on RPM limit,
        considering reserved slots for the fallback model.

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

            # Check if this is the fallback model and if reservation applies
            is_fallback = False
            reserved_slots = 0
            if self.client_instance and hasattr(self.client_instance, 'fallback_model_id'):
                # Ensure comparison uses clean IDs (without models/ prefix if present)
                clean_model_name = model_name.split('/')[-1]
                clean_fallback_id = self.client_instance.fallback_model_id.split(
                    '/')[-1]
                if clean_model_name == clean_fallback_id:
                    is_fallback = True
                    if hasattr(self.client_instance, 'reserved_fallback_slots'):
                        reserved_slots = self.client_instance.reserved_fallback_slots

            # Determine the effective limit considering reservations
            effective_limit = limit
            if is_fallback and reserved_slots > 0:
                # Reduce the effective limit by the reserved slots for normal calls
                effective_limit = max(0, limit - reserved_slots)
                # Changed to TRACE (or remove)
                # logger.debug(
                #     f"Applying fallback reservation for {model_name}. Original limit: {limit}, Reserved: {reserved_slots}, Effective limit: {effective_limit}")

            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]

            # Remove timestamps older than 60 seconds
            original_count = len(timestamps)
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()
            removed_count = original_count - len(timestamps)
            # Reduce debug noise
            # if removed_count > 0:
            #     logger.debug(
            #         f"RateLimiter [{model_name}]: Removed {removed_count} old timestamps.")

            # Check if the number of calls in the last 60 seconds exceeds the EFFECTIVE limit
            current_count = len(timestamps)
            allowed = current_count < effective_limit

            # Special case: Allow fallback model usage up to its *actual* limit during emergency
            # (This logic requires GeminiClient to somehow signal emergency mode, e.g., via a flag)
            # Placeholder: Assume an emergency flag could be set on client_instance
            # in_emergency_mode = getattr(self.client_instance, '_emergency_mode', False)
            # if not allowed and is_fallback and in_emergency_mode and current_count < limit:
            #     logger.warning(f"Allowing fallback model {model_name} in emergency mode despite reservation (Current: {current_count}/{limit})")
            #     allowed = True

            # Changed to TRACE (or remove)
            # logger.debug(
            #     f"RateLimiter [{model_name}]: Check: {current_count} calls / {effective_limit} effective RPM limit (Actual Limit: {limit}) -> Allowed: {allowed}")

            return allowed

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
                # Changed to TRACE (or remove)
                # logger.debug(
                #     f"RateLimiter [{model_name}]: No limit configured, not registering call.")
                return

            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]

            # Remove old timestamps (consistency check)
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()

            # Add the current timestamp
            timestamps.append(current_time)
            # Changed to TRACE (or remove)
            # logger.debug(
            #     f"RateLimiter [{model_name}]: Registered call. Current count in window: {len(timestamps)} / {limit} RPM")

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

    async def reserve_and_register_call_async(self, model_name: str) -> bool:
        """
        Atomically check if a call is allowed, and if so, immediately register it.
        This prevents the race condition where multiple tasks check and then register.

        Returns:
            bool: True if the call was allowed and registered, False otherwise
        """
        with self.lock:
            # Check if there's capacity
            # Directly call is_allowed which already handles lock acquisition
            if self.is_allowed(model_name):
                # Immediately register the call
                # Directly call register_call which also handles lock acquisition
                self.register_call(model_name)
                return True
            return False

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
                    logger.warning(
                        # Changed to WARNING
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

    async def wait_if_needed_async(self, model_name: str, max_wait_seconds: float = 60.0) -> None:
        """
        Asynchronous version of wait_if_needed.
        Waits the required time if the model's rate limit has been reached.
        Will break out of waiting if max_wait_seconds is exceeded.

        Args:
            model_name (str): The name of the model to check.
            max_wait_seconds (float): Maximum time to wait in seconds before breaking out.
        """
        start_time = time.monotonic()
        total_wait_time = 0.0

        while not self.is_allowed(model_name):
            # Check if we've waited too long already
            current_wait_time = time.monotonic() - start_time
            if current_wait_time >= max_wait_seconds:
                logger.warning(
                    f"RateLimiter [{model_name}]: Breaking out after waiting {current_wait_time:.2f}s (max: {max_wait_seconds}s)"
                )
                break

            # logger.debug(f"RateLimiter [{model_name}]: Acquiring lock for wait check...") # Removed debug log
            with self.lock:
                # logger.debug(f"RateLimiter [{model_name}]: Acquired lock for wait check.") # Removed debug log
                timestamps = self.call_timestamps[model_name]
                if not timestamps:  # Should not happen if is_allowed is False, but safe check
                    logger.warning(
                        # Changed to WARNING
                        f"RateLimiter [{model_name}]: No timestamps found while waiting, breaking wait loop.")
                    break
                # Calculate how long to wait until the oldest call expires
                oldest_call_time = timestamps[0]
                current_time_in_wait = time.monotonic()
                wait_time = (oldest_call_time + 60) - \
                    current_time_in_wait + 0.1  # Add small buffer
                wait_time = max(0, wait_time)  # Ensure non-negative

                # Ensure we don't exceed max_wait_seconds
                remaining_wait_allowed = max_wait_seconds - current_wait_time
                if wait_time > remaining_wait_allowed:
                    wait_time = remaining_wait_allowed
                    # Changed to DEBUG
                    logger.debug(
                        f"RateLimiter [{model_name}]: Capping wait time to {wait_time:.2f}s to respect max_wait_seconds={max_wait_seconds}s"
                    )

                # Recalculate inside lock scope if needed, or pass value
                # Note: get_current_rpm also uses the lock
                current_rpm = self.get_current_rpm(model_name)
                # Changed to DEBUG
                logger.debug(
                    f"RateLimiter [{model_name}]: Limit reached ({current_rpm} RPM). Calculated wait: {wait_time:.2f}s. Sleeping...")
                # Removed redundant time log: (until {oldest_call_time + 60:.2f}). Releasing lock and sleeping...")

            # Sleep outside the lock
            if wait_time > 0:
                total_wait_time += wait_time
                await asyncio.sleep(wait_time)
                # Changed to TRACE (or remove)
                # logger.debug(
                #     f"RateLimiter [{model_name}]: Finished sleep of {wait_time:.2f}s. Re-checking limit...")
            else:
                # If wait_time is 0, yield control briefly to avoid busy-waiting in edge cases
                # Changed to TRACE (or remove)
                # logger.debug(
                #     f"RateLimiter [{model_name}]: Calculated wait time is 0. Yielding control before re-checking.")
                await asyncio.sleep(0.01)
                # Re-check is_allowed in the next loop iteration
        # Changed to TRACE (or remove)
        # logger.debug(
        #     f"RateLimiter [{model_name}]: Wait no longer needed. Total waited: {total_wait_time:.2f}s")
