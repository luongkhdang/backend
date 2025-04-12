#!/usr/bin/env python3
"""
rate_limit.py - Rate limiting utilities for API calls

This module provides rate limiting functionality for API calls to external services,
particularly for managing rate limits with Google's Gemini API.

Exported functions/classes:
- RateLimiter: Class for rate limiting API calls
  - __init__(max_calls_per_minute): Initializes the rate limiter with a rate limit
  - wait_if_needed(): Waits if necessary to stay within rate limits
  - register_call(): Registers an API call to track rate limiting
  - get_wait_time(): Calculates how long to wait before the next API call

Related files:
- src/steps/step1.py: Uses this rate limiter for embedding generation
- src/gemini/gemini_client.py: Makes API calls that need rate limiting
"""

import time
import os
import logging
from typing import List, Dict, Any, Optional
from threading import Lock
from collections import deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for API calls to prevent exceeding rate limits."""

    def __init__(self, max_calls_per_minute: int = None):
        """
        Initialize the rate limiter.

        Args:
            max_calls_per_minute: Maximum number of API calls allowed per minute.
                                  If None, will try to get from environment variable.
        """
        # Use provided value or get from environment
        self.max_calls_per_minute = max_calls_per_minute or int(
            os.getenv("GEMINI_EMBEDDING_RATE_LIMIT_PER_MINUTE", "1500"))

        # Initialize call history with timestamps (thread-safe)
        self.call_history = deque(maxlen=self.max_calls_per_minute)
        self.lock = Lock()  # For thread safety

        logger.info(
            f"Initialized rate limiter: {self.max_calls_per_minute} calls per minute maximum")

    def wait_if_needed(self) -> float:
        """
        Wait if necessary to stay within rate limits.

        Returns:
            float: The time waited in seconds (0 if no wait was needed)
        """
        wait_time = self.get_wait_time()

        if wait_time > 0:
            # Log only for significant waits
            if wait_time > 0.5:
                logger.info(
                    f"Rate limit approaching - waiting {wait_time:.2f} seconds before next API call")
            time.sleep(wait_time)
            return wait_time
        return 0

    def register_call(self):
        """Register an API call to track for rate limiting."""
        with self.lock:
            now = datetime.now()
            self.call_history.append(now)

    def get_wait_time(self) -> float:
        """
        Calculate how long to wait before the next API call to stay within rate limits.

        Returns:
            float: Time to wait in seconds (0 if no wait needed)
        """
        with self.lock:
            if len(self.call_history) < self.max_calls_per_minute:
                # We haven't reached the maximum number of calls yet
                return 0

            # Check if the oldest call in history was less than a minute ago
            now = datetime.now()
            oldest_call = self.call_history[0]  # Oldest call in our history
            minute_ago = now - timedelta(minutes=1)

            if oldest_call > minute_ago:
                # We've made max_calls_per_minute within the last minute
                # Calculate how long to wait until the oldest call is one minute old
                wait_seconds = (oldest_call - minute_ago).total_seconds()
                return max(0, wait_seconds)

            # The oldest call is already more than a minute old
            return 0
