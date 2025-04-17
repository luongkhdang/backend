#!/usr/bin/env python3
"""
gemini_client.py - Client for Google's Gemini API

This module provides a client for Google's Gemini API, supporting both text embeddings
and text generation with model selection and rate limiting.

The client now supports extracting both entities and narrative frame phrases,
which helps identify the dominant perspective or framing used in news articles.
These frame phrases are short descriptors (2-4 words) representing how stories
are presented (e.g., "economic impact", "national security", "moral obligation").

Exported functions/classes:
- GeminiClient: Class for interacting with the Gemini API
  - __init__(self): Initializes the client with API key and creates an internal rate limiter.
  - generate_embedding(text, task_type, retries, initial_delay): Generates embeddings for text with retry logic.
  - generate_text_with_prompt(self, article_content: str, processing_tier: int, retries: int = 3, initial_delay: float = 1.0, model_override: Optional[str] = None) -> Optional[Dict[str, Any]]: Generates text based on article content and a prompt, using model selection logic.
  - generate_text_with_prompt_async(self, article_content: str, processing_tier: int, retries: int = 3, initial_delay: float = 1.0, model_override: Optional[str] = None) -> Optional[Dict[str, Any]]: Async version of text generation method.

Related files:
- src/steps/step1.py: Uses this client for embedding generation.
- src/steps/step3.py: Uses this client for text generation (entity and frame phrase extraction).
- src/database/reader_db_client.py: Stores generated embeddings, extracted entities, and frame phrases.
- src/utils/rate_limit.py: Provides the RateLimiter class used by this client.
- src/prompts/entity_extraction_prompt.txt: Contains the prompt template for text generation.
"""

import google.generativeai as genai
import os
import time
import random
import logging
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Try importing the rate limiter
try:
    from src.utils.rate_limit import RateLimiter
except ImportError:
    RateLimiter = None
    logging.getLogger(__name__).warning(
        "RateLimiter class not found. Rate limiting will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Generation Models and their RPM limits (as per user input)
# Store as a class variable for clarity


class GeminiClient:
    """Client for interacting with Google's Gemini API."""

    # Define model preferences and fallback logic
    # Order matters: Best/Preferred first
    GENERATION_MODELS_CONFIG = {
        'models/gemini-2.0-flash-thinking-exp-01-21': 10,  # RPM
        'models/gemini-2.0-flash-exp': 10,               # RPM
        'models/gemini-2.0-flash': 15,                   # RPM from .env/compose was 15
    }
    FALLBACK_MODEL = 'models/gemini-2.0-flash-lite'
    FALLBACK_MODEL_RPM = 30

    # Combined dict for RateLimiter initialization
    ALL_MODEL_RPMS = {
        **GENERATION_MODELS_CONFIG,
        FALLBACK_MODEL: FALLBACK_MODEL_RPM
    }
    # Add embedding model RPM if needed, assuming it's 1500 as per .env
    ALL_MODEL_RPMS[os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
                   ] = int(os.getenv("GEMINI_EMBEDDING_RATE_LIMIT_PER_MINUTE", "1500"))

    DEFAULT_PROMPT_PATH = "src/prompts/entity_extraction_prompt.txt"

    # Default max wait time for rate limit cooldown in seconds
    DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 40

    def __init__(self):
        """
        Initialize the Gemini API client.

        Loads the API key from environment variables, configures the client,
        and initializes the rate limiter with model RPMs.
        """
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            error_msg = "GEMINI_API_KEY environment variable not set"
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Embedding model configuration
        self.embedding_model = os.getenv(
            "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        if not self.embedding_model.startswith("models/"):
            self.embedding_model = f"models/{self.embedding_model}"
        self.default_task_type = os.getenv(
            "GEMINI_EMBEDDING_TASK_TYPE", "CLUSTERING")
        self.embedding_input_token_limit = int(
            os.getenv("GEMINI_EMBEDDING_INPUT_TOKEN_LIMIT", "2048"))

        # Rate limit wait configuration
        self.max_rate_limit_wait_seconds = float(os.getenv(
            "GEMINI_MAX_WAIT_SECONDS", str(self.DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS)))
        logger.info(
            f"Maximum rate limit wait time set to {self.max_rate_limit_wait_seconds} seconds")

        # Initialize Rate Limiter if available
        if RateLimiter:
            self.rate_limiter = RateLimiter(self.ALL_MODEL_RPMS)
            logger.info("GeminiClient initialized with RateLimiter.")
        else:
            self.rate_limiter = None
            logger.warning("GeminiClient initialized WITHOUT RateLimiter.")

        # Initialize the Gemini client library
        try:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully.")
        except Exception as e:
            error_msg = f"Failed to initialize Gemini client: {e}"
            logger.critical(error_msg)
            raise RuntimeError(error_msg)

        # Load the generation prompt template
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self, prompt_path: str = DEFAULT_PROMPT_PATH) -> Optional[str]:
        """Loads the prompt template from the specified file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info(
                f"Successfully loaded prompt template from {prompt_path}")
            return template
        except FileNotFoundError:
            logger.error(f"Prompt template file not found at {prompt_path}")
            return None
        except Exception as e:
            logger.error(
                f"Error loading prompt template from {prompt_path}: {e}")
            return None

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
        model_to_use = self.embedding_model  # Use the configured embedding model

        # Validate input text
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            logger.error(f"Invalid input text for embedding: {text[:30]}...")
            return None

        # Character limit based on token ratio of 1:4
        max_chars = self.embedding_input_token_limit * 4
        if len(text) > max_chars:
            logger.warning(
                f"Text length ({len(text)} chars) exceeds recommended embedding limit of {max_chars} chars for model {model_to_use}")
            # Do not truncate here, let the API handle it or error if needed

        for attempt in range(retries):
            try:
                # Log only on first attempt or if we've had failures
                if attempt > 0:
                    logger.info(
                        f"Retry attempt {attempt+1}/{retries} for embedding generation using {model_to_use}")

                # Apply rate limiting if a rate limiter is provided
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed(model_to_use)

                # Generate embedding using the genai embeddings API
                result = genai.embed_content(
                    model=model_to_use,
                    content=text,
                    task_type=task_type
                )

                # Log the raw API response for debugging
                logger.info(
                    f"Gemini API raw embedding response for model {model_to_use}: {result}")

                # Register this call with the rate limiter if provided
                if self.rate_limiter:
                    self.rate_limiter.register_call(model_to_use)

                embedding = self._extract_embedding_from_result(result)

                if embedding:
                    return embedding
                else:
                    # Log unexpected format only once per function call
                    if attempt == 0:
                        logger.warning(
                            f"Invalid or empty embedding format received from API for model {model_to_use}. Result: {result}")

            except Exception as e:
                logger.warning(
                    f"Embedding generation error (Attempt {attempt+1}/{retries}) using {model_to_use}: {e}")
                # Check if it's a rate limit error (specific exception types might vary)
                if "rate limit" in str(e).lower():
                    logger.warning(
                        f"Rate limit likely hit for {model_to_use}. Retrying after delay...")
                    # Rate limiter should handle waiting, but add extra delay just in case
                    # Jittered delay before next attempt
                    time.sleep(delay * (random.random() * 0.5 + 0.5))
                    # No need to switch models for embedding here, just retry

            # Exponential backoff for retries
            if attempt < retries - 1:
                jitter = random.random() * 0.5 + 0.5  # 0.5-1.0 multiplier
                sleep_time = delay * jitter
                logger.debug(
                    f"Waiting {sleep_time:.2f} seconds before next embedding attempt.")
                time.sleep(sleep_time)
                delay *= 2
            else:
                logger.error(
                    f"Failed to generate embedding using {model_to_use} after {retries} attempts")

        return None

    def _extract_embedding_from_result(self, result: Any) -> Optional[List[float]]:
        """Helper to extract embedding from potentially different API response structures."""
        embedding = None
        if hasattr(result, "embedding"):
            embedding = result.embedding
        elif isinstance(result, dict) and "embedding" in result:
            embedding = result["embedding"]
        # Add other checks if needed based on observed API responses

        if embedding is not None and isinstance(embedding, list) and len(embedding) > 0:
            return embedding
        return None

    def generate_text_with_prompt(self, article_content: str, processing_tier: int,
                                  retries: int = 3, initial_delay: float = 1.0,
                                  model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generates text (e.g., entity extraction JSON) based on article content and a prompt.

        Uses model selection based on rate limits and falls back if necessary.

        Args:
            article_content (str): The content of the article to analyze.
            processing_tier (int): The processing tier (0, 1, or 2) - currently unused but available.
            retries (int): Maximum number of retry attempts for the entire process (across models).
            initial_delay (float): Initial delay for retries in seconds.
            model_override (Optional[str]): Optional specific model to use instead of tier-based selection.

        Returns:
            Optional[Dict[str, Any]]: The parsed JSON response as a dictionary, or None if failed.
        """
        if not self.prompt_template:
            logger.error(
                "Cannot generate text: Prompt template is not loaded.")
            return None
        if not article_content or not isinstance(article_content, str) or len(article_content.strip()) == 0:
            logger.error(
                "Invalid article content provided for text generation.")
            return None

        # Construct the full prompt
        full_prompt = self.prompt_template.replace(
            "{ARTICLE_CONTENT_HERE}", article_content)

        # Define model preference order, handling model_override if provided
        if model_override and model_override.startswith("models/"):
            # Use specified model first, then fall back to defaults if needed
            preferred_models = [model_override]
            all_models_to_try = preferred_models + \
                list(self.GENERATION_MODELS_CONFIG.keys()) + \
                [self.FALLBACK_MODEL]
            # Remove duplicates while preserving order
            all_models_to_try = list(dict.fromkeys(all_models_to_try))
            logger.info(
                f"Using overridden model selection with primary model: {model_override}")
        else:
            # Use default tier-based selection
            preferred_models = list(self.GENERATION_MODELS_CONFIG.keys())
            all_models_to_try = preferred_models + [self.FALLBACK_MODEL]
            logger.info(
                f"Using default model selection with tier {processing_tier}")

        current_delay = initial_delay
        total_attempts = 0

        for model_name in all_models_to_try:
            logger.info(f"Attempting text generation with model: {model_name}")

            # Check rate limit *before* attempting the call
            if self.rate_limiter:
                logger.debug(f"Checking rate limit for {model_name}...")
                is_allowed = self.rate_limiter.is_allowed(model_name)

                if not is_allowed:
                    # Check how long we would need to wait
                    wait_time = self.rate_limiter.get_wait_time(model_name)

                    # If wait time is within acceptable threshold, wait and continue with this model
                    if 0 < wait_time <= self.max_rate_limit_wait_seconds:
                        logger.info(
                            f"Rate limit hit for {model_name}. Waiting {wait_time:.2f}s...")
                        self.rate_limiter.wait_if_needed(model_name)
                        # Proceed with this model after waiting
                    else:
                        # Wait time is too long, skip to next model
                        logger.warning(
                            f"Rate limit wait for {model_name} ({wait_time:.2f}s) > threshold ({self.max_rate_limit_wait_seconds}s). Skipping.")
                        continue  # Skip this model and try the next one
                else:
                    logger.debug(f"Rate limit check passed for {model_name}.")
            else:
                logger.debug(
                    "Rate limiter not available, proceeding without check.")

            # Reset attempt counter for this specific model
            model_attempt = 0
            max_model_retries = 2

            # Allow a couple of retries per model before switching
            while model_attempt < max_model_retries and total_attempts < retries:
                total_attempts += 1
                model_attempt += 1

                try:
                    # Wait if needed (should be rare if is_allowed check passed, but safety)
                    if self.rate_limiter:
                        logger.debug(
                            f"Calling wait_if_needed for {model_name} (safety check)...")
                        self.rate_limiter.wait_if_needed(model_name)

                    # Configure the generative model instance
                    # Note: genai.GenerativeModel configuration might need adjustment based on API version
                    model = genai.GenerativeModel(model_name)

                    # Make the API call
                    # Adjust parameters (temperature, top_p, etc.) if needed
                    # Ensure the response MIME type is set for JSON output if applicable/possible
                    response = model.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            # candidate_count=1, # Default is 1
                            # stop_sequences=['...'], # If needed
                            # max_output_tokens=8192, # Use model default unless specific need
                            temperature=0.1,  # Low temperature for deterministic JSON
                            # top_p=1.0, # Defaults
                            # top_k=1, # Defaults
                        )
                        # safety_settings=... # Add safety settings if required
                    )

                    # Log the raw API response for debugging
                    logger.info(
                        f"Gemini API raw generation response for model {model_name}: {response}")

                    # Register the successful call *after* it completes
                    if self.rate_limiter:
                        self.rate_limiter.register_call(model_name)
                        # Add a debug log after registration if desired
                        # logger.debug(f"Call registered for {model_name} in GeminiClient.")

                    # --- Process Response ---
                    # Accessing response text might vary slightly depending on API version/response structure
                    generated_text = None
                    if hasattr(response, 'text'):
                        generated_text = response.text
                    elif hasattr(response, 'parts') and response.parts:
                        # Check if parts contain text
                        text_parts = [
                            part.text for part in response.parts if hasattr(part, 'text')]
                        if text_parts:
                            # Concatenate if multiple parts
                            generated_text = "".join(text_parts)

                    if generated_text:
                        logger.info(
                            f"Successfully generated text using {model_name}")
                        # Basic check for JSON structure - find first '{' and last '}'
                        start_index = generated_text.find('{')
                        end_index = generated_text.rfind('}')
                        if start_index != -1 and end_index != -1 and start_index < end_index:
                            json_part = generated_text[start_index:end_index+1]
                            # Try to parse the JSON
                            try:
                                parsed_json = json.loads(json_part.strip())
                                logger.info(
                                    f"Successfully parsed JSON response from {model_name}")
                                return parsed_json  # Return the parsed JSON dictionary
                            except json.JSONDecodeError as json_err:
                                logger.warning(
                                    f"JSON parsing error from {model_name}: {json_err}. Raw JSON: {json_part[:100]}...")
                                # Continue with retries or model switches
                        else:
                            logger.warning(
                                f"Generated text from {model_name} does not appear to be valid JSON: {generated_text[:100]}...")
                            # Treat as failure for this attempt, maybe retry or switch model
                    else:
                        logger.warning(
                            f"Received empty or invalid response structure from {model_name}. Response: {response}")

                except Exception as e:
                    logger.warning(
                        f"Text generation error (Attempt {model_attempt}/{max_model_retries}) using {model_name}: {e}")
                    # Check if it's a rate limit error - this might trigger skipping the model
                    error_str = str(e).lower()
                    # More robust check for rate limit / quota errors
                    is_rate_limit_error = "rate limit" in error_str or "quota" in error_str or "resource has been exhausted" in error_str

                    if is_rate_limit_error:
                        # Check if we should retry this model after a sleep
                        if model_attempt < max_model_retries and total_attempts < retries:
                            logger.warning(
                                f"Rate limit/Quota error on {model_name} (Attempt {model_attempt}/{max_model_retries}). Sleeping 10s and retrying...")
                            time.sleep(10)  # Sleep for 10 seconds
                            continue  # Retry the same model
                        else:
                            # Exhausted retries for this model or global retries, force switch
                            logger.error(
                                f"Rate limit/Quota error encountered with {model_name}: {e}. Retries exhausted for this model. Switching model if possible.")
                            break  # Exit inner loop to try next model
                    else:
                        # Handle non-rate-limit errors
                        if model_attempt < max_model_retries and total_attempts < retries:
                            jitter = random.random() * 0.5 + 0.5
                            sleep_time = current_delay * jitter
                            logger.debug(
                                f"Non-rate-limit error. Waiting {sleep_time:.2f} seconds before retrying {model_name}.")
                            time.sleep(sleep_time)
                            current_delay *= 1.5  # Increase delay slightly for model-specific retries
                            continue  # Retry the same model
                        else:
                            # Exhausted retries for this model or global retries
                            logger.error(
                                f"Failed to generate text with {model_name} after {model_attempt} attempts due to non-rate-limit error: {e}")
                            break  # Exit inner loop

                # If successful (returned above), this part is skipped
                # If failed but retries remain for this model, loop continues

            # If the inner loop finished (either success or exhausted model retries),
            # check if we got a result. If so, exit outer loop.
            # (Success case handled by return inside the loop)
            # If rate limit error broke the inner loop, the outer loop continues to next model.

        # If loop finishes without returning, all models/retries failed
        logger.error(
            f"Failed to generate text after trying all models and {retries} total attempts.")
        return None

    async def generate_text_with_prompt_async(self, article_content: str, processing_tier: int,
                                              retries: int = 3, initial_delay: float = 1.0,
                                              model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Async version of text generation method.

        Args:
            article_content (str): The content of the article to analyze.
            processing_tier (int): The processing tier (0, 1, or 2) - currently unused but available.
            retries (int): Maximum number of retry attempts for the entire process (across models).
            initial_delay (float): Initial delay for retries in seconds.
            model_override (Optional[str]): Optional specific model to use instead of tier-based selection.

        Returns:
            Optional[Dict[str, Any]]: The parsed JSON response as a dictionary, or None if failed.
        """
        if not self.prompt_template:
            logger.error(
                "Cannot generate text: Prompt template is not loaded.")
            return None
        if not article_content or not isinstance(article_content, str) or len(article_content.strip()) == 0:
            logger.error(
                "Invalid article content provided for text generation.")
            return None

        # Construct the full prompt
        full_prompt = self.prompt_template.replace(
            "{ARTICLE_CONTENT_HERE}", article_content)

        # Define model preference order, handling model_override if provided
        if model_override and model_override.startswith("models/"):
            # Use specified model first, then fall back to defaults if needed
            preferred_models = [model_override]
            all_models_to_try = preferred_models + \
                list(self.GENERATION_MODELS_CONFIG.keys()) + \
                [self.FALLBACK_MODEL]
            # Remove duplicates while preserving order
            all_models_to_try = list(dict.fromkeys(all_models_to_try))
            logger.info(
                f"Using overridden model selection with primary model: {model_override}")
        else:
            # Use default tier-based selection
            preferred_models = list(self.GENERATION_MODELS_CONFIG.keys())
            all_models_to_try = preferred_models + [self.FALLBACK_MODEL]
            logger.info(
                f"Using default model selection with tier {processing_tier}")

        current_delay = initial_delay
        total_attempts = 0

        for model_name in all_models_to_try:
            logger.info(f"Attempting text generation with model: {model_name}")

            # Check rate limit *before* attempting the call
            if self.rate_limiter:
                logger.debug(f"Checking rate limit for {model_name}...")
                is_allowed = self.rate_limiter.is_allowed(model_name)

                if not is_allowed:
                    # Check how long we would need to wait
                    wait_time = await self.rate_limiter.get_wait_time_async(model_name)

                    # If wait time is within acceptable threshold, wait and continue with this model
                    if 0 < wait_time <= self.max_rate_limit_wait_seconds:
                        logger.info(
                            f"Rate limit hit for {model_name}. Waiting {wait_time:.2f}s...")
                        await self.rate_limiter.wait_if_needed_async(model_name)
                        # Proceed with this model after waiting
                    else:
                        # Wait time is too long, skip to next model
                        logger.warning(
                            f"Rate limit wait for {model_name} ({wait_time:.2f}s) > threshold ({self.max_rate_limit_wait_seconds}s). Skipping.")
                        continue  # Skip this model and try the next one
                else:
                    logger.debug(f"Rate limit check passed for {model_name}.")
            else:
                logger.debug(
                    "Rate limiter not available, proceeding without check.")

            # Reset attempt counter for this specific model
            model_attempt = 0
            max_model_retries = 2

            # Allow a couple of retries per model before switching
            while model_attempt < max_model_retries and total_attempts < retries:
                total_attempts += 1
                model_attempt += 1

                try:
                    # Wait if needed (should be rare if is_allowed check passed, but safety)
                    if self.rate_limiter:
                        logger.debug(
                            f"Calling wait_if_needed_async for {model_name} (safety check)...")
                        await self.rate_limiter.wait_if_needed_async(model_name)

                    # Configure the generative model instance
                    # Note: genai.GenerativeModel configuration might need adjustment based on API version
                    model = genai.GenerativeModel(model_name)

                    # Make the API call
                    # Adjust parameters (temperature, top_p, etc.) if needed
                    # Ensure the response MIME type is set for JSON output if applicable/possible
                    response = await model.generate_content_async(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            # candidate_count=1, # Default is 1
                            # stop_sequences=['...'], # If needed
                            # max_output_tokens=8192, # Use model default unless specific need
                            temperature=0.1,  # Low temperature for deterministic JSON
                            # top_p=1.0, # Defaults
                            # top_k=1, # Defaults
                        )
                        # safety_settings=... # Add safety settings if required
                    )

                    # Log the raw API response for debugging
                    logger.info(
                        f"Gemini API raw async generation response for model {model_name}: {response}")

                    # Register the successful call *after* it completes
                    if self.rate_limiter:
                        await self.rate_limiter.register_call_async(model_name)
                        # Add a debug log after registration if desired
                        # logger.debug(f"Call registered for {model_name} in GeminiClient.")

                    # --- Process Response ---
                    # Accessing response text might vary slightly depending on API version/response structure
                    generated_text = None
                    if hasattr(response, 'text'):
                        generated_text = response.text
                    elif hasattr(response, 'parts') and response.parts:
                        # Check if parts contain text
                        text_parts = [
                            part.text for part in response.parts if hasattr(part, 'text')]
                        if text_parts:
                            # Concatenate if multiple parts
                            generated_text = "".join(text_parts)

                    if generated_text:
                        logger.info(
                            f"Successfully generated text using {model_name}")
                        # Basic check for JSON structure - find first '{' and last '}'
                        start_index = generated_text.find('{')
                        end_index = generated_text.rfind('}')
                        if start_index != -1 and end_index != -1 and start_index < end_index:
                            json_part = generated_text[start_index:end_index+1]
                            # Try to parse the JSON
                            try:
                                parsed_json = json.loads(json_part.strip())
                                logger.info(
                                    f"Successfully parsed JSON response from {model_name}")
                                return parsed_json  # Return the parsed JSON dictionary
                            except json.JSONDecodeError as json_err:
                                logger.warning(
                                    f"JSON parsing error from {model_name}: {json_err}. Raw JSON: {json_part[:100]}...")
                                # Continue with retries or model switches
                        else:
                            logger.warning(
                                f"Generated text from {model_name} does not appear to be valid JSON: {generated_text[:100]}...")
                            # Treat as failure for this attempt, maybe retry or switch model
                    else:
                        logger.warning(
                            f"Received empty or invalid response structure from {model_name}. Response: {response}")

                except Exception as e:
                    logger.warning(
                        f"Text generation error (Attempt {model_attempt}/{max_model_retries}) using {model_name}: {e}")
                    # Check if it's a rate limit error - this might trigger skipping the model
                    error_str = str(e).lower()
                    # More robust check for rate limit / quota errors
                    is_rate_limit_error = "rate limit" in error_str or "quota" in error_str or "resource has been exhausted" in error_str

                    if is_rate_limit_error:
                        # Check if we should retry this model after a sleep
                        if model_attempt < max_model_retries and total_attempts < retries:
                            logger.warning(
                                f"Rate limit/Quota error on {model_name} (Attempt {model_attempt}/{max_model_retries}). Sleeping 10s and retrying...")
                            # Async sleep for 10 seconds
                            await asyncio.sleep(10)
                            continue  # Retry the same model
                        else:
                            # Exhausted retries for this model or global retries, force switch
                            logger.error(
                                f"Rate limit/Quota error encountered with {model_name}: {e}. Retries exhausted for this model. Switching model if possible.")
                            break  # Exit inner loop to try next model
                    else:
                        # Handle non-rate-limit errors
                        if model_attempt < max_model_retries and total_attempts < retries:
                            jitter = random.random() * 0.5 + 0.5
                            sleep_time = current_delay * jitter
                            logger.debug(
                                f"Non-rate-limit error. Waiting {sleep_time:.2f} seconds before retrying {model_name}.")
                            await asyncio.sleep(sleep_time)
                            current_delay *= 1.5  # Increase delay slightly for model-specific retries
                            continue  # Retry the same model
                        else:
                            # Exhausted retries for this model or global retries
                            logger.error(
                                f"Failed to generate text with {model_name} after {model_attempt} attempts due to non-rate-limit error: {e}")
                            break  # Exit inner loop

                # If successful (returned above), this part is skipped
                # If failed but retries remain for this model, loop continues

            # If the inner loop finished (either success or exhausted model retries),
            # check if we got a result. If so, exit outer loop.
            # (Success case handled by return inside the loop)
            # If rate limit error broke the inner loop, the outer loop continues to next model.

        # If loop finishes without returning, all models/retries failed
        logger.error(
            f"Failed to generate text after trying all models and {retries} total attempts.")
        return None
