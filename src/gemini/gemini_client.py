#!/usr/bin/env python3
"""
gemini_client.py - Client for Google's Gemini API

This module provides a client for Google's Gemini API, specifically for generating
text embeddings using the text-embedding-004 model.

Exported functions/classes:
- GeminiClient: Class for interacting with the Gemini API
  - __init__(): Initializes the client with API key from environment
  - generate_embedding(text, task_type, retries, initial_delay): Generates embeddings for text with retry logic

Related files:
- src/steps/step1.py: Uses this client to generate embeddings for articles
- src/database/reader_db_client.py: Stores generated embeddings
"""

import google.generativeai as genai
import os
import time
import random
import logging
import sys
import subprocess
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the Google GenAI package


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    def __init__(self):
        """
        Initialize the Gemini API client.

        Loads the API key from environment variables and configures the client.
        """
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            error_msg = "GEMINI_API_KEY environment variable not set"
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Get model name and task type from environment (or use defaults)
        self.embedding_model = os.getenv(
            "GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.default_task_type = os.getenv(
            "GEMINI_EMBEDDING_TASK_TYPE", "CLUSTERING")

        # Initialize the Gemini client
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.Client(api_key=self.api_key)
            logger.info(
                f"Initialized Gemini client with model: {self.embedding_model}, default task type: {self.default_task_type}")
        except Exception as e:
            error_msg = f"Failed to initialize Gemini client: {e}"
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

    def generate_embedding(self, text: str, task_type: str = None,
                           retries: int = 3, initial_delay: float = 1.0) -> Optional[List[float]]:
        """
        Generate an embedding for the given text using the Gemini API.

        Args:
            text: Text to generate embedding for
            task_type: The task type for embedding optimization (default from environment or "CLUSTERING")
            retries: Maximum number of retry attempts (default: 3)
            initial_delay: Initial delay between retries in seconds (default: 1.0)

        Returns:
            List[float] or None: Embedding vector if successful, None if failed after retries
        """
        # Use task_type from argument or default from environment
        task_type = task_type or self.default_task_type
        delay = initial_delay

        for attempt in range(retries):
            try:
                logger.debug(
                    f"Generating embedding for text of length {len(text)} (attempt {attempt+1}/{retries})")

                # Create embedding config with task type
                # Use dict-based configuration instead of types module
                embed_config = {"task_type": task_type}

                # Generate embedding using the client.models API
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=embed_config
                )

                # Extract embeddings from result
                if result and hasattr(result, "embeddings"):
                    embeddings = result.embeddings
                    logger.debug(
                        f"Successfully generated embedding of dimension {len(embeddings)}")
                    return embeddings
                else:
                    logger.warning(
                        f"Empty or invalid embedding result: {result}")

            except Exception as e:
                logger.warning(
                    f"Embedding generation error (attempt {attempt+1}/{retries}): {e}")

                if attempt < retries - 1:
                    # Add jitter to backoff delay
                    jitter = random.random() * 0.5 + 0.5  # 0.5-1.0 multiplier
                    sleep_time = delay * jitter
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to generate embedding after {retries} attempts: {e}")

        return None
