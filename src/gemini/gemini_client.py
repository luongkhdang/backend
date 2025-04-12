#!/usr/bin/env python3
"""
gemini_client.py - Client for Google's Gemini API

This module provides a client for Google's Gemini API, specifically for generating
text embeddings using the models/text-embedding-004 model.

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

    def __init__(self, rate_limiter=None):
        """
        Initialize the Gemini API client.

        Loads the API key from environment variables and configures the client.

        Args:
            rate_limiter: Optional rate limiter instance to control API call frequency
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
            "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

        # Ensure model name has "models/" prefix if not already present
        if not self.embedding_model.startswith("models/"):
            self.embedding_model = f"models/{self.embedding_model}"

        self.default_task_type = os.getenv(
            "GEMINI_EMBEDDING_TASK_TYPE", "CLUSTERING")

        # Store the rate limiter
        self.rate_limiter = rate_limiter

        # Initialize the Gemini client
        try:
            genai.configure(api_key=self.api_key)
            # The Client class doesn't exist - use the top-level genai module directly
            # self.client = genai.Client(api_key=self.api_key)
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

        # Validate input text
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            logger.error(f"Invalid input text for embedding: {text[:30]}...")
            return None

        # Check if text is within limits (but don't truncate)
        # For Gemini models, 1 token ≈ 4 characters as per Google documentation
        max_tokens = int(
            os.getenv("GEMINI_EMBEDDING_INPUT_TOKEN_LIMIT", "2048"))
        # Character limit based on token ratio of 1:4
        max_chars = max_tokens * 4

        if len(text) > max_chars:
            logger.warning(f"Text length ({len(text)} chars) exceeds recommended limit of {max_chars} chars. " +
                           f"This may cause embedding generation to fail.")

        for attempt in range(retries):
            try:
                logger.debug(
                    f"Generating embedding for text of length {len(text)} (attempt {attempt+1}/{retries})")

                # Apply rate limiting if a rate limiter is provided
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                # Generate embedding using the genai embeddings API
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type=task_type
                )

                # Register this call with the rate limiter if provided
                if self.rate_limiter:
                    self.rate_limiter.register_call()

                # Log the raw result structure for debugging
                logger.debug(f"Raw API response type: {type(result)}")
                logger.debug(f"Raw API response dir: {dir(result)}")

                # Different versions of the API may have different response structures
                # Try different attribute paths that might contain the embedding
                embedding = None

                # First try the expected path in newer versions
                if hasattr(result, "embedding"):
                    embedding = result.embedding
                # Try accessing 'embeddings' attribute (plural) from older versions
                elif hasattr(result, "embeddings"):
                    embedding = result.embeddings
                # Try dict-style access
                elif isinstance(result, dict) and "embedding" in result:
                    embedding = result["embedding"]
                elif isinstance(result, dict) and "embeddings" in result:
                    embedding = result["embeddings"]

                # For some API versions, result might be a list of embedding objects
                elif isinstance(result, list) and len(result) > 0:
                    if hasattr(result[0], "embedding"):
                        embedding = result[0].embedding
                    elif hasattr(result[0], "embeddings"):
                        embedding = result[0].embeddings

                # Validate the embedding we found (if any)
                if embedding is not None and isinstance(embedding, list) and len(embedding) > 0:
                    logger.debug(
                        f"Successfully generated embedding of dimension {len(embedding)}")
                    return embedding

                # Log detailed error based on what we found
                if embedding is None:
                    logger.warning(
                        f"Could not find embedding in result: {result}")
                elif not isinstance(embedding, list):
                    logger.warning(
                        f"Embedding is not a list but {type(embedding)}")
                elif len(embedding) == 0:
                    logger.warning(f"Embedding is an empty list")

                # Log full structure as string for debugging
                # Limit to 500 chars to avoid huge logs
                result_repr = str(result)[:500]
                logger.warning(
                    f"Invalid embedding result format: {result_repr}...")

            except Exception as e:
                logger.warning(
                    f"Embedding generation error (attempt {attempt+1}/{retries}): {e}")

            # Try again unless this was the last attempt
            if attempt < retries - 1:
                # Add jitter to backoff delay
                jitter = random.random() * 0.5 + 0.5  # 0.5-1.0 multiplier
                sleep_time = delay * jitter
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                delay *= 2  # Exponential backoff
            else:
                logger.error(
                    f"Failed to generate embedding after {retries} attempts")

        return None
