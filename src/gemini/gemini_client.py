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
  - generate_text_with_prompt_async(self, article_content: str, processing_tier: int, retries: int = 6, initial_delay: float = 1.0, model_override: Optional[str] = None, fallback_model: Optional[str] = None) -> Optional[Dict[str, Any]]: Async version of text generation method.
  - analyze_articles_with_prompt(self, articles_data: List[Dict[str, Any]], prompt_file_path: str, model_name: str, system_instruction: Optional[str] = None, temperature: float = 0.2, max_output_tokens: int = 8192, retries: int = 3, initial_delay: float = 1.0) -> Optional[str]: Analyzes a list of articles using a specified prompt template and returns structured JSON output.

Related files:
- src/steps/step1.py: Uses this client for embedding generation.
- src/steps/step3.py: Uses this client for text generation (entity and frame phrase extraction).
- src/steps/step4.py: Uses this client for article analysis and grouping.
- src/database/reader_db_client.py: Stores generated embeddings, extracted entities, and frame phrases.
- src/utils/rate_limit.py: Provides the RateLimiter class used by this client.
- src/prompts/entity_extraction_prompt.txt: Contains the prompt template for text generation.
- src/prompts/step4.txt: Contains the prompt template for article analysis.
"""

import google.generativeai as genai
import os
import os.path
import time
import random
import logging
import sys
import json
import asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import warnings

# Try importing the rate limiter
try:
    from src.utils.rate_limit import RateLimiter
except ImportError:
    RateLimiter = None
    logging.getLogger(__name__).warning(
        "RateLimiter class not found. Rate limiting will be disabled.")

# Import the function from the new generator module
from src.gemini.modules.generator import analyze_articles_with_prompt as generator_analyze_articles

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
                logger.debug(
                    f"Gemini API raw embedding response for model {model_to_use}: {result}")

                # Log usage metadata if available
                if hasattr(result, 'usage_metadata'):
                    logger.info(
                        f"[USAGE] Embedding usage metadata: {result.usage_metadata}")
                elif isinstance(result, dict) and 'usage_metadata' in result:
                    logger.info(
                        f"[USAGE] Embedding usage metadata: {result['usage_metadata']}")

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
            logger.debug(
                f"Attempting text generation with model: {model_name}")

            # Check rate limit *before* attempting the call
            if self.rate_limiter:
                logger.debug(f"Checking rate limit for {model_name}...")
                is_allowed = self.rate_limiter.is_allowed(model_name)

                if not is_allowed:
                    # Check how long we would need to wait
                    wait_time = self.rate_limiter.get_wait_time(model_name)

                    # If wait time is within acceptable threshold, wait and continue with this model
                    if 0 < wait_time <= self.max_rate_limit_wait_seconds:
                        logger.debug(
                            f"Rate limit hit for {model_name}. Waiting {wait_time:.2f}s...")
                        self.rate_limiter.wait_if_needed(model_name)
                        # Proceed with this model after waiting
                    else:
                        # Wait time is too long, skip to next model
                        logger.warning(
                            f"Rate limit wait for {model_name} ({wait_time:.2f}s) > threshold ({self.max_rate_limit_wait_seconds}s). Skipping.")
                        # Add this crucial log to track the model fallback
                        next_model_index = all_models_to_try.index(
                            model_name) + 1
                        if next_model_index < len(all_models_to_try):
                            next_model = all_models_to_try[next_model_index]
                            logger.info(
                                f"Will try fallback model: {next_model} next")
                        else:
                            logger.error(
                                f"No more fallback models available after {model_name}! All models exhausted.")
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

                    # Log the full response object for debugging
                    logger.info(
                        f"[RESPONSE] Full response from {model_name}: {response}")

                    # Log usage metadata if available
                    if hasattr(response, 'usage_metadata'):
                        logger.info(
                            f"[USAGE] {model_name} usage metadata: {response.usage_metadata}")
                    elif isinstance(response, dict) and 'usage_metadata' in response:
                        logger.info(
                            f"[USAGE] {model_name} usage metadata: {response['usage_metadata']}")

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
                                              retries: int = 6, initial_delay: float = 1.0,
                                              model_override: Optional[str] = None,
                                              fallback_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Async version of text generation method.

        Args:
            article_content (str): The content of the article to analyze.
            processing_tier (int): The processing tier (0, 1, or 2) - currently unused but available.
            retries (int): Maximum number of retry attempts for the entire process (across models).
            initial_delay (float): Initial delay for retries in seconds.
            model_override (Optional[str]): Optional specific model to use instead of tier-based selection.
            fallback_model (Optional[str]): Tier-specific fallback model to try before the global fallback.

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

        # Set maximum wait time to 40 seconds to prefer waiting for better models
        self.max_rate_limit_wait_seconds = 40

        # Construct the full prompt
        full_prompt = self.prompt_template.replace(
            "{ARTICLE_CONTENT_HERE}", article_content)

        # Define model preference order, handling model_override and fallback_model
        if model_override and model_override.startswith("models/"):
            # Use specified model first, then tier-specific fallback if provided, then default fallbacks
            preferred_models = [model_override]

            # Add tier-specific fallback if provided
            if fallback_model and fallback_model.startswith("models/"):
                if fallback_model not in preferred_models:
                    preferred_models.append(fallback_model)

            # Then add default models as final fallbacks
            all_models_to_try = preferred_models + \
                list(self.GENERATION_MODELS_CONFIG.keys()) + \
                [self.FALLBACK_MODEL]
            # Remove duplicates while preserving order
            all_models_to_try = list(dict.fromkeys(all_models_to_try))
            logger.info(
                f"Using tiered model selection: Primary={model_override}, Tier-fallback={fallback_model or 'None'}")
        else:
            # Use default tier-based selection
            preferred_models = list(self.GENERATION_MODELS_CONFIG.keys())

            # Insert tier-specific fallback if provided and not already in list
            if fallback_model and fallback_model.startswith("models/"):
                # If fallback isn't already in the preferred models, add it at the beginning
                if fallback_model not in preferred_models:
                    preferred_models.insert(0, fallback_model)

            all_models_to_try = preferred_models + [self.FALLBACK_MODEL]
            # Remove duplicates while preserving order
            all_models_to_try = list(dict.fromkeys(all_models_to_try))
            logger.info(
                f"Using default model selection with tier {processing_tier}, fallback={fallback_model or 'None'}")

        current_delay = initial_delay
        total_attempts = 0

        # Track which models we've tried and failed due to rate limits
        rate_limited_models = set()

        # Log the model sequence we'll be trying
        logger.debug(f"Model sequence to try: {all_models_to_try}")

        for model_name in all_models_to_try:
            logger.debug(
                f"Attempting text generation with model: {model_name}")

            # Emergency bailout - if all models are rate limited except fallback, force use of fallback
            if len(rate_limited_models) >= len(all_models_to_try) - 1 and model_name == self.FALLBACK_MODEL:
                logger.warning(
                    f"All models except fallback are rate limited. Forcing use of fallback model {self.FALLBACK_MODEL} with forced waiting")
                # We will wait however long needed for the fallback model
                if self.rate_limiter:
                    logger.info(
                        f"Waiting up to 120 seconds for fallback model {self.FALLBACK_MODEL} to be available...")
                    # Wait up to 2 minutes for the fallback model
                    max_wait = 120
                    start_time = time.monotonic()
                    while time.monotonic() - start_time < max_wait:
                        wait_time = await self.rate_limiter.get_wait_time_async(self.FALLBACK_MODEL)
                        if wait_time <= 0:
                            logger.info(
                                f"Fallback model {self.FALLBACK_MODEL} is now available after waiting")
                            break
                        logger.debug(
                            f"Fallback still rate limited, waiting {min(wait_time, 5)}s...")
                        # Wait the lesser of wait_time or 5s
                        await asyncio.sleep(min(wait_time, 5))

                    # Even if we're still rate limited, we'll try anyway since this is our last resort
                    await self.rate_limiter.wait_if_needed_async(self.FALLBACK_MODEL)

            # Check rate limit *before* attempting the call
            if self.rate_limiter:
                logger.debug(f"Checking rate limit for {model_name}...")
                is_allowed = self.rate_limiter.is_allowed(model_name)

                if not is_allowed:
                    # Check how long we would need to wait
                    wait_time = await self.rate_limiter.get_wait_time_async(model_name)

                    # If wait time is within acceptable threshold, wait and continue with this model
                    if 0 < wait_time <= self.max_rate_limit_wait_seconds:
                        logger.debug(
                            f"Rate limit hit for {model_name}. Waiting {wait_time:.2f}s...")
                        await self.rate_limiter.wait_if_needed_async(model_name)
                        # Proceed with this model after waiting
                    else:
                        # Wait time is too long, skip to next model
                        logger.warning(
                            f"Rate limit wait for {model_name} ({wait_time:.2f}s) > threshold ({self.max_rate_limit_wait_seconds}s). Skipping.")
                        # Add this crucial log to track the model fallback
                        next_model_index = all_models_to_try.index(
                            model_name) + 1
                        if next_model_index < len(all_models_to_try):
                            next_model = all_models_to_try[next_model_index]
                            logger.info(
                                f"Will try fallback model: {next_model} next")
                        else:
                            logger.error(
                                f"No more fallback models available after {model_name}! All models exhausted.")

                        # Track this model as rate limited
                        rate_limited_models.add(model_name)
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
                task_start_time = time.monotonic()  # Add timer start
                # Add attempt log
                logger.debug(
                    f"Attempt {model_attempt}/{max_model_retries} for model {model_name}, total attempt {total_attempts}/{retries}")

                try:
                    # Wait if needed (should be rare if is_allowed check passed, but safety)
                    if self.rate_limiter:
                        # Added logging
                        logger.debug(
                            f"[{model_name}] Attempt {model_attempt}: Checking/Waiting for rate limit...")
                        await self.rate_limiter.wait_if_needed_async(model_name)
                        # Added logging
                        logger.debug(
                            f"[{model_name}] Attempt {model_attempt}: Rate limit check passed/wait finished.")

                    # Configure the generative model instance
                    # Note: genai.GenerativeModel configuration might need adjustment based on API version
                    model = genai.GenerativeModel(model_name)

                    # --- Add Logging and Timeout ---
                    api_call_timeout_seconds = 120  # Define timeout duration
                    # Added logging
                    logger.debug(
                        f"[{model_name}] Attempt {model_attempt}: Calling generate_content_async (timeout={api_call_timeout_seconds}s)...")
                    try:
                        # Make the API call WITH TIMEOUT
                        # Adjust parameters (temperature, top_p, etc.) if needed
                        # Ensure the response MIME type is set for JSON output if applicable/possible
                        response = await asyncio.wait_for(
                            model.generate_content_async(
                                full_prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=0.1,  # Low temperature for deterministic JSON
                                )
                                # safety_settings=... # Add safety settings if required
                            ),
                            timeout=api_call_timeout_seconds
                        )
                        call_duration = time.monotonic() - task_start_time
                        # Added logging
                        logger.debug(
                            f"[{model_name}] Attempt {model_attempt}: generate_content_async call succeeded in {call_duration:.2f}s.")

                        # Log the full response
                        logger.info(
                            f"[RESPONSE] Full response from {model_name}: {response}")

                        # Log usage metadata if available
                        if hasattr(response, 'usage_metadata'):
                            logger.info(
                                f"[USAGE] {model_name} usage metadata: {response.usage_metadata}")
                        elif isinstance(response, dict) and 'usage_metadata' in response:
                            logger.info(
                                f"[USAGE] {model_name} usage metadata: {response['usage_metadata']}")
                    except asyncio.TimeoutError:
                        call_duration = time.monotonic() - task_start_time
                        # Added logging
                        logger.error(
                            f"[{model_name}] Attempt {model_attempt}: API call timed out after {call_duration:.2f} seconds (limit: {api_call_timeout_seconds}s).")
                        # Re-raise or handle timeout as a specific failure case
                        raise asyncio.TimeoutError(
                            f"{model_name} API call timed out")
                    except Exception as api_err:
                        call_duration = time.monotonic() - task_start_time
                        # Added logging
                        logger.error(
                            f"[{model_name}] Attempt {model_attempt}: API call failed after {call_duration:.2f}s: {api_err}")
                        raise api_err  # Re-raise other API errors
                    # --- End Logging and Timeout ---

                    # Log the raw API response for debugging
                    logger.debug(
                        f"Gemini API raw async generation response for model {model_name}: {response}")

                    # Register the successful call *after* it completes
                    if self.rate_limiter:
                        # Added logging
                        logger.debug(
                            f"[{model_name}] Attempt {model_attempt}: Registering successful call...")
                        await self.rate_limiter.register_call_async(model_name)
                        # Added logging
                        logger.debug(
                            f"[{model_name}] Attempt {model_attempt}: Call registered.")
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

                except asyncio.TimeoutError:
                    # Handle timeout specifically if needed (e.g., break inner loop, switch model faster)
                    logger.warning(
                        f"Timeout error caught for {model_name} Attempt {model_attempt}. Breaking inner loop for this model.")
                    break  # Exit the inner retry loop for this model after a timeout
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
                            # Use rate limiter's wait logic instead of fixed sleep
                            if self.rate_limiter:
                                logger.warning(
                                    f"Rate limit/Quota error on {model_name} (Attempt {model_attempt}/{max_model_retries}). Waiting using rate limiter logic...")
                                await self.rate_limiter.wait_if_needed_async(model_name)
                                logger.info(
                                    f"Finished rate limit wait for {model_name}. Retrying...")
                                continue  # Retry the same model
                            else:
                                logger.warning(
                                    f"Rate limit/Quota error on {model_name} but no rate limiter available. Sleeping 10s as fallback.")
                                await asyncio.sleep(10)  # Fallback sleep
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

    async def analyze_articles_with_prompt(self, articles_data: List[Dict[str, Any]],
                                           prompt_file_path: str,
                                           model_name: str,
                                           system_instruction: Optional[str] = None,
                                           temperature: float = 0.2,
                                           max_output_tokens: int = 8192,
                                           retries: int = 3,
                                           initial_delay: float = 1.0) -> Optional[str]:
        """
        Delegates to the dedicated module implementation to analyze a list of articles using a specified prompt template.

        This method has been relocated to src/gemini/modules/generator.py for better modularity.
        This wrapper maintains backward compatibility for existing code.

        Args:
            articles_data: List of prepared article dictionaries.
            prompt_file_path: Path to the prompt template file (e.g., `src/prompts/step4.txt`).
            model_name: Specific Gemini model to use.
            system_instruction: Override for the default system instruction.
            temperature: Controls randomness in generation. Defaults to 0.2.
            max_output_tokens: Maximum number of tokens in response. Defaults to 8192.
            retries: Maximum number of retry attempts. Defaults to 3.
            initial_delay: Initial delay between retries in seconds. Defaults to 1.0.

        Returns:
            Optional[str]: The full text response if successful, None if all retries fail.
        """
        # Default system instruction if none provided
        default_system_instruction = """
        The AI agent should adopt an academic persona—specifically, that of a seasoned political science professor at Stanford, 
        who is also a leading expert in political and economic affairs with access to insider information by virtue of sitting 
        on the directive board of the current ruling party. Aware of the risks of censorship and the precariousness of its position, 
        the agent must carefully navigate these constraints, striving to present factual information in a way that encourages 
        independent thought. Rather than drawing explicit conclusions, it subtly unveils the truth through evidence and context, 
        allowing the audience to arrive at their own interpretations. At its core, the agent is an educator, committed to the 
        intellectual growth of the next generation. It recognizes that failing to uphold this responsibility would be a betrayal 
        of its duty as a noble scholar and mentor.
        """

        # Use provided system instruction or default
        actual_system_instruction = system_instruction if system_instruction is not None else default_system_instruction

        logger.info(
            f"Delegating article analysis to generator module (model: {model_name})")

        # Delegate to the generator module, passing our rate_limiter instance
        return await generator_analyze_articles(
            articles_data=articles_data,
            prompt_file_path=prompt_file_path,
            model_name=model_name,
            system_instruction=actual_system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            retries=retries,
            initial_delay=initial_delay,
            rate_limiter=self.rate_limiter
        )
