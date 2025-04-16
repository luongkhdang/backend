"""
rate_limit.py - Simple Rate Limiter for API Calls

This module provides a basic rate limiter to manage API call frequency,
primarily based on Requests Per Minute (RPM) for different models.

Exported Classes:
- RateLimiter: Tracks API calls per minute for different models and enforces limits.
    - __init__(self, model_configs: Dict[str, int])
    - is_allowed(self, model_name: str) -> bool
    - register_call(self, model_name: str) -> None

Related Files:
- src/gemini/gemini_client.py: Uses this rate limiter.
"""

import time
import logging
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
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()

            # Check if the number of calls in the last 60 seconds exceeds the limit
            if len(timestamps) < limit:
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
                return

            current_time = time.monotonic()
            timestamps = self.call_timestamps[model_name]

            # Remove old timestamps (consistency check)
            while timestamps and timestamps[0] <= current_time - 60:
                timestamps.popleft()

            # Add the current timestamp
            timestamps.append(current_time)
            # Optional: Log call registration
            # logger.debug(f"Registered call for model '{model_name}'. Current count in window: {len(timestamps)}")

    def wait_if_needed(self, model_name: str) -> None:
        """
        Blocks until a call is allowed for the specified model.

        Args:
            model_name (str): The name of the model to wait for.
        """
        while not self.is_allowed(model_name):
            with self.lock:
                timestamps = self.call_timestamps[model_name]
                if not timestamps:  # Should not happen if is_allowed is False, but safe check
                    break
                # Calculate how long to wait until the oldest call expires
                wait_time = (timestamps[0] + 60) - \
                    time.monotonic() + 0.1  # Add small buffer

            logger.debug(
                f"Rate limit reached for {model_name}. Waiting for {wait_time:.2f} seconds.")
            time.sleep(max(0, wait_time))
            # Re-check is_allowed in the next loop iteration
