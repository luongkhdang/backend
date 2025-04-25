#!/usr/bin/env python3
"""
gemini_client.py - Client for Google's Gemini API using google-genai

This module provides a client for Google's Gemini API (using the google-genai library),
supporting both text embeddings and text generation with model selection and rate limiting.

Supports text embeddings, structured text generation (e.g., entity extraction),
article analysis, and free-form text generation from prompts (e.g., essay generation).
Includes model selection, rate limiting (via RateLimiter), and retry logic.

Exported functions/classes:
- GeminiClient: Class for interacting with the Gemini API
  - __init__(self): Initializes the google-genai client with API key and creates an internal rate limiter.
  - generate_embedding(text, task_type, retries, initial_delay): Generates embeddings for text with retry logic using self.client.models.embed_content.
  - generate_text_with_prompt(self, ...): Synchronously generates text based on article content and a prompt, using model selection logic and self.client.generate_content.
  - generate_text_with_prompt_async(self, ...): Async version of text generation method using self.client.generate_content_async. Handles model selection, fallback, retries, and rate limiting internally.
  - analyze_articles_with_prompt(self, ...): Delegates analysis to the generator module, passing the initialized client.
  - generate_essay_from_prompt(self, ...): Delegates essay generation to the generator module, passing the initialized client.

Related files:
- src/steps/step1.py: Uses this client for embedding generation.
- src/steps/step3.py: Uses this client for text generation (entity and frame phrase extraction).
- src/steps/step4.py: Uses this client for article analysis and grouping.
- src/steps/step5.py: Uses this client for essay generation.
- src/database/reader_db_client.py: Stores generated embeddings, extracted entities, and frame phrases.
- src/utils/rate_limit.py: Provides the RateLimiter class used by this client.
- src/gemini/modules/generator.py: Contains helper functions called by this client.
- src/prompts/entity_extraction_prompt.txt: Contains the prompt template for text generation.
- src/prompts/step4.txt: Contains the prompt template for article analysis.
- src/prompts/haystack_prompt.txt: Contains the prompt template for essay generation.
"""

# Use alias to avoid conflicts if needed later
from google import genai as google_genai
import google.api_core.exceptions  # Import common exceptions for error handling
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
from src.gemini.modules.generator import generate_text_from_prompt as generator_generate_text

# Add import for robust JSON parser
from src.utils.json_parser import parse_json, extract_json_from_text, safe_loads

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Generation Models and their RPM limits (as per user input)
# Store as a class variable for clarity


class GeminiClient:
    """
    Client for interacting with Google's Gemini API using the google-genai SDK.
    Supports text embeddings, structured text generation (e.g., entity extraction),
    article analysis, and free-form text generation from prompts (e.g., essay generation).
    Includes model selection, rate limiting, and retry logic.
    """

    # Define model preferences and fallback logic
    # Order matters: Best/Preferred first
    GENERATION_MODELS_CONFIG = {
        # Use short IDs as keys for rate limiter consistency
        # Updated to 2.0 versions based on logs
        # RPM - Assuming this is a valid ID, check API/library docs if needed
        'gemini-2.0-flash-thinking-exp-01-21': 10,
        'gemini-2.0-flash-exp': 10,               # RPM - Assuming this is a valid ID
        'gemini-2.0-flash': 15,            # RPM - Assuming this is the base 2.0 flash
    }
    # Use short ID
    # Updated to 2.0 version based on logs
    FALLBACK_MODEL = 'gemini-2.0-flash'
    FALLBACK_MODEL_RPM = 30

    # Combined dict for RateLimiter initialization using SHORT IDs
    ALL_MODEL_RPMS = {
        **GENERATION_MODELS_CONFIG,
        FALLBACK_MODEL: FALLBACK_MODEL_RPM
    }
    # Add embedding model SHORT ID and RPM
    # Extract short ID from the potentially prefixed env var
    embedding_model_full = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    embedding_model_short_id = embedding_model_full.split('/')[-1]
    # Default to 150 RPM for embedding model if ENV VAR not set, matching observed API limit
    embedding_rpm_limit = int(
        os.getenv("GEMINI_EMBEDDING_RATE_LIMIT_PER_MINUTE", "150"))
    ALL_MODEL_RPMS[embedding_model_short_id] = embedding_rpm_limit
    # Log the effective limit being used
    # Temp variable for logging in __init__
    _log_initial_rate_limit = embedding_rpm_limit

    DEFAULT_PROMPT_PATH = "src/prompts/entity_extraction_prompt.txt"

    # Default max wait time for rate limit cooldown in seconds
    DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 40

    def __init__(self):
        """
        Initialize the google-genai API client.
        Loads the API key from environment variables, creates the client,
        and initializes the rate limiter with model RPMs (using short model IDs).
        """
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = "GEMINI_API_KEY environment variable not set"
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # Initialize the google-genai client
        try:
            self.client = google_genai.Client(api_key=api_key)
            logger.info("google-genai client initialized successfully.")
        except Exception as e:
            error_msg = f"Failed to initialize google-genai client: {e}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # Embedding model configuration
        self.embedding_model = os.getenv(
            "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        # Keep the full name with prefix for API calls, but use short ID for rate limiter
        self.embedding_model_short_id = self.embedding_model.split('/')[-1]
        # Store the full name separately if needed for API calls
        # self.embedding_model_full = self.embedding_model

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
            logger.info(
                f"GeminiClient initialized with RateLimiter. Config: {self.ALL_MODEL_RPMS}")
            # Log the specific embedding limit being used (accessing the stored value)
            logger.info(
                f"Effective rate limit for embedding model '{self.embedding_model_short_id}': {self.ALL_MODEL_RPMS.get(self.embedding_model_short_id, 'N/A')} RPM")
        else:
            self.rate_limiter = None
            logger.warning("GeminiClient initialized WITHOUT RateLimiter.")

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
        Generate an embedding for the given text using the google-genai SDK.
        Uses self.client.models.embed_content(). Handles retries and rate limiting (using short model ID).
        """
        # Use task_type from argument or default from environment
        task_type = task_type or self.default_task_type
        delay = initial_delay
        model_to_use_for_api = self.embedding_model  # Full name for API call
        # Short name for rate limiter
        model_to_use_for_limiter = self.embedding_model_short_id

        # Validate input text
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            logger.error(f"Invalid input text for embedding: {text[:30]}...")
            return None

        # Character limit based on token ratio of 1:4
        max_chars = self.embedding_input_token_limit * 4
        if len(text) > max_chars:
            logger.warning(
                f"Text length ({len(text)} chars) exceeds recommended embedding limit of {max_chars} chars for model {model_to_use_for_api}")
            # Do not truncate here, let the API handle it or error if needed

        for attempt in range(retries):
            try:
                # Log only on first attempt or if we've had failures
                if attempt > 0:
                    logger.info(
                        f"Retry attempt {attempt+1}/{retries} for embedding generation using {model_to_use_for_api}")

                # Apply rate limiting if a rate limiter is provided (use short ID)
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed(model_to_use_for_limiter)

                # Generate embedding using the new client and method (use full name)
                result = self.client.models.embed_content(
                    model=model_to_use_for_api,
                    contents=text
                )

                # Log the raw API response for debugging (result is likely a Pydantic object or dict)
                logger.debug(
                    f"google-genai raw embedding response for model {model_to_use_for_api}: {result}")

                # Log usage metadata if available
                usage_metadata = getattr(result, 'usage_metadata', None) or result.get(
                    'usage_metadata', None) if isinstance(result, dict) else None
                if usage_metadata:
                    logger.info(
                        f"[USAGE] Embedding usage metadata: {usage_metadata}")

                # Register this call with the rate limiter if provided (use short ID)
                if self.rate_limiter:
                    self.rate_limiter.register_call(model_to_use_for_limiter)

                # Extract embedding from the new result structure
                # Access result.embeddings[0].values based on observed output
                embedding = None
                if result and hasattr(result, 'embeddings') and isinstance(result.embeddings, list) and len(result.embeddings) > 0:
                    first_embedding = result.embeddings[0]
                    if hasattr(first_embedding, 'values') and isinstance(first_embedding.values, list):
                        embedding = first_embedding.values
                    else:
                        logger.warning(
                            f"First embedding object lacks 'values' list: {first_embedding}")
                else:
                    logger.warning(
                        f"Result object missing 'embeddings' list or list is empty: {result}")

                if embedding and len(embedding) > 0:
                    return embedding
                else:
                    # Log unexpected format only once per function call
                    if attempt == 0:
                        # Log the original result object for better debugging
                        logger.warning(
                            f"Invalid or empty embedding format received from google-genai API for model {model_to_use_for_api}. Original result: {result}")

            except google.api_core.exceptions.ResourceExhausted as e:  # Catch specific rate limit exception
                logger.warning(
                    f"Embedding generation ResourceExhausted (Rate Limit/Quota) (Attempt {attempt+1}/{retries}) using {model_to_use_for_api}: {e}")
                # Use sync wait (use short ID)
                if self.rate_limiter:
                    logger.warning(
                        f"Waiting based on rate limiter for {model_to_use_for_limiter}...")
                    # Wait and retry loop will continue
                    self.rate_limiter.wait_if_needed(model_to_use_for_limiter)
                else:
                    # Fallback wait if no rate limiter
                    time.sleep(delay * (random.random() * 0.5 + 0.5))
            except google.api_core.exceptions.GoogleAPIError as e:  # Catch other general Google API errors
                logger.warning(
                    # Less verbose logging
                    f"Embedding generation Google API error (Attempt {attempt+1}/{retries}) using {model_to_use_for_api}: {e}", exc_info=False)
            except Exception as e:
                logger.warning(
                    f"Embedding generation generic error (Attempt {attempt+1}/{retries}) using {model_to_use_for_api}: {e}", exc_info=True)
                # Consider other specific google.api_core exceptions if needed

            # Exponential backoff for retries
            if attempt < retries - 1:
                jitter = random.random() * 0.5 + 0.5
                sleep_time = delay * jitter
                logger.debug(
                    f"Waiting {sleep_time:.2f} seconds before next embedding attempt.")
                time.sleep(sleep_time)  # Use time.sleep
                delay *= 2
            else:
                logger.error(
                    f"Failed to generate embedding using {model_to_use_for_api} after {retries} attempts")

        return None

    def generate_text_with_prompt(self, article_content: str, processing_tier: int,
                                  retries: int = 3, initial_delay: float = 1.0,
                                  model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Synchronously generates text (e.g., entity extraction JSON) using google-genai.
        Uses self.client.generate_content() with model selection and fallback logic.
        Handles retries and rate limiting.
        """
        if not self.prompt_template:
            logger.error(
                "Cannot generate text: Prompt template is not loaded.")
            return None
        if not article_content or not isinstance(article_content, str) or len(article_content.strip()) == 0:
            logger.error(
                "Invalid article content provided for text generation.")
            return None

        max_wait_threshold = self.max_rate_limit_wait_seconds
        full_prompt = self.prompt_template.replace(
            "{ARTICLE_CONTENT_HERE}", article_content)

        # --- Model Selection Logic (mirrors async version but uses IDs) ---
        if model_override and model_override.startswith("models/"):
            model_override_id = model_override.split('/')[-1]
            preferred_models = [model_override_id]
            all_models_to_try_ids = preferred_models + \
                [m.split('/')[-1] for m in self.GENERATION_MODELS_CONFIG.keys()] + \
                [self.FALLBACK_MODEL.split('/')[-1]]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))
        else:
            preferred_models = [m.split('/')[-1]
                                for m in self.GENERATION_MODELS_CONFIG.keys()]
            all_models_to_try_ids = preferred_models + \
                [self.FALLBACK_MODEL.split('/')[-1]]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))
        logger.debug(f"SYNC: Model sequence to try: {all_models_to_try_ids}")
        # --- End Model Selection ---

        current_delay = initial_delay
        total_attempts = 0
        rate_limited_models = set()

        for model_id in all_models_to_try_ids:
            logger.debug(
                f"SYNC: Attempting text generation with model: {model_id}")

            # --- Emergency Bailout (sync version) ---
            fallback_model_id = self.FALLBACK_MODEL.split('/')[-1]
            if len(rate_limited_models) >= len(all_models_to_try_ids) - 1 and model_id == fallback_model_id:
                logger.warning(
                    f"SYNC: Forcing use of fallback model {fallback_model_id}...")
                if self.rate_limiter:
                    # Wait potentially long time for fallback (sync)
                    max_wait = 120
                    start_time = time.monotonic()
                    while time.monotonic() - start_time < max_wait:
                        wait_time = self.rate_limiter.get_wait_time(
                            fallback_model_id)
                        if wait_time <= 0:
                            break
                        time.sleep(min(wait_time, 5))
                    self.rate_limiter.wait_if_needed(fallback_model_id)
            # --- End Bailout ---

            # --- Rate Limit Check (sync) ---
            if self.rate_limiter:
                is_allowed = self.rate_limiter.is_allowed(model_id)
                if not is_allowed:
                    wait_time = self.rate_limiter.get_wait_time(model_id)
                    if 0 < wait_time <= max_wait_threshold:
                        logger.debug(
                            f"SYNC: Rate limit hit for {model_id}. Waiting {wait_time:.2f}s...")
                        self.rate_limiter.wait_if_needed(model_id)
                    else:
                        logger.warning(
                            f"SYNC: Rate limit wait for {model_id} ({wait_time:.2f}s) > threshold. Skipping.")
                        rate_limited_models.add(model_id)
                        continue
            # --- End Rate Limit Check ---

            model_attempt = 0
            max_model_retries = 2

            while model_attempt < max_model_retries and total_attempts < retries:
                total_attempts += 1
                model_attempt += 1
                logger.debug(
                    f"SYNC: Attempt {model_attempt}/{max_model_retries} for model {model_id}, total attempt {total_attempts}/{retries}")

                try:
                    if self.rate_limiter:
                        self.rate_limiter.wait_if_needed(model_id)  # Sync wait

                    # --- NEW SYNC API CALL ---
                    gen_config = google_genai.types.GenerateContentConfig(
                        temperature=0.1,
                        automatic_function_calling={'disable': True}
                    )

                    # Use the synchronous client method
                    response = self.client.generate_content(
                        model=f'models/{model_id}',
                        contents=[full_prompt],
                        config=gen_config
                    )
                    # --- END NEW SYNC API CALL ---

                    logger.debug(
                        f"SYNC: [RESPONSE] Full response from {model_id}: {response}")
                    usage_metadata = getattr(response, 'usage_metadata', None)
                    if usage_metadata:
                        logger.debug(
                            f"SYNC: [USAGE] {model_id} usage metadata: {usage_metadata}")

                    if self.rate_limiter:
                        self.rate_limiter.register_call(
                            model_id)  # Sync register

                    # --- Process Response (same logic as async) ---
                    generated_text = None
                    if hasattr(response, 'text'):
                        generated_text = response.text
                    elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
                        generated_text = "".join(
                            part.text for part in response.parts if hasattr(part, 'text'))

                    if generated_text:
                        parsed_json, error = parse_json(generated_text)
                        if not error:
                            logger.info(
                                f"SYNC: Successfully parsed JSON response from {model_id}")
                            return parsed_json  # Success!
                        else:
                            # ... (JSON extraction/parsing retry logic) ...
                            extracted_json, extract_error = extract_json_from_text(
                                generated_text)
                            if not extract_error:
                                parsed_json, error = parse_json(extracted_json)
                                if not error and parsed_json:
                                    return parsed_json  # Success!
                            # Log failure if needed
                            logger.warning(
                                f"SYNC: Failed to parse/extract JSON from {model_id}")
                    else:
                        finish_reason = getattr(response, 'prompt_feedback', None) or getattr(
                            response, 'finish_reason', None)
                        logger.warning(
                            f"SYNC: Received no generated text from {model_id}. Finish reason/Feedback: {finish_reason}")
                    # --- End Process Response ---

                except google.api_core.exceptions.ResourceExhausted as e:
                    logger.warning(
                        f"SYNC: [{model_id}] ResourceExhausted (Attempt {model_attempt}/{max_model_retries}): {e}")
                    if model_attempt < max_model_retries and total_attempts < retries:
                        if self.rate_limiter:
                            self.rate_limiter.wait_if_needed(model_id)
                            continue
                        else:
                            time.sleep(current_delay *
                                       (random.random()*0.5+0.5))
                            continue
                    else:
                        logger.error(
                            f"SYNC: Rate limit/Quota error with {model_id}, retries exhausted.")
                        rate_limited_models.add(model_id)
                        break
                except google.api_core.exceptions.GoogleAPIError as e:  # Catch other general Google API errors
                    logger.warning(
                        f"SYNC: [{model_id}] Google API error (Attempt {model_attempt}/{max_model_retries}): {e}", exc_info=False)
                    # Retry logic for general API errors
                    if model_attempt < max_model_retries and total_attempts < retries:
                        time.sleep(current_delay * (random.random()*0.5+0.5))
                        continue
                    else:
                        break  # Exhausted retries
                except Exception as e:
                    logger.warning(
                        f"SYNC: Generic error (Attempt {model_attempt}/{max_model_retries}) using {model_id}: {e}", exc_info=True)
                    if model_attempt < max_model_retries and total_attempts < retries:
                        jitter = random.random() * 0.5 + 0.5
                        sleep_time = current_delay * jitter
                        time.sleep(sleep_time)
                        current_delay *= 1.5
                        continue
                    else:
                        logger.error(
                            f"Failed on {model_id} after {model_attempt} attempts due to non-rate-limit error: {e}")
                        break

        logger.error(
            f"SYNC: Failed to generate text after trying all models and {retries} total attempts.")
        return None

    async def generate_text_with_prompt_async(self, article_content: str, processing_tier: int,
                                              retries: int = 6, initial_delay: float = 1.0,
                                              model_override: Optional[str] = None,
                                              fallback_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Asynchronously generates text (e.g., entity extraction JSON) using google-genai.
        Uses self.client.generate_content_async() with model selection and fallback logic.
        Handles retries and rate limiting.
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
        max_wait_threshold = self.max_rate_limit_wait_seconds

        # Construct the full prompt
        full_prompt = self.prompt_template.replace(
            "{ARTICLE_CONTENT_HERE}", article_content)

        # Define model preference order, handling model_override and fallback_model
        if model_override and model_override.startswith("models/"):
            # Ensure model names are just the ID, not starting with "models/" if the new client adds it
            # Example: client.generate_content(model='gemini-1.5-flash-latest', ...)
            # Adjust model_override and fallback_model if needed
            model_override_id = model_override.split('/')[-1]
            fallback_model_id = fallback_model.split(
                '/')[-1] if fallback_model else None
            preferred_models = [model_override_id]
            if fallback_model_id and fallback_model_id not in preferred_models:
                preferred_models.append(fallback_model_id)
            all_models_to_try_ids = preferred_models + \
                [m.split('/')[-1] for m in self.GENERATION_MODELS_CONFIG.keys()] + \
                [self.FALLBACK_MODEL.split('/')[-1]]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))
        else:
            # Default selection
            preferred_models = [m.split('/')[-1]
                                for m in self.GENERATION_MODELS_CONFIG.keys()]
            fallback_model_id = fallback_model.split(
                '/')[-1] if fallback_model else None
            if fallback_model_id and fallback_model_id not in preferred_models:
                preferred_models.insert(0, fallback_model_id)
            all_models_to_try_ids = preferred_models + \
                [self.FALLBACK_MODEL.split('/')[-1]]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))
        logger.debug(f"Model sequence to try: {all_models_to_try_ids}")

        current_delay = initial_delay
        total_attempts = 0
        rate_limited_models = set()

        for model_id in all_models_to_try_ids:  # Iterate through model IDs
            logger.debug(f"Attempting text generation with model: {model_id}")

            # Emergency bailout - if all models are rate limited except fallback, force use of fallback
            fallback_model_id = self.FALLBACK_MODEL.split('/')[-1]
            if len(rate_limited_models) >= len(all_models_to_try_ids) - 1 and model_id == fallback_model_id:
                logger.warning(
                    f"Forcing use of fallback model {fallback_model_id}...")
                if self.rate_limiter:
                    # We will wait however long needed for the fallback model
                    logger.info(
                        f"Waiting up to 120 seconds for fallback model {fallback_model_id} to be available...")
                    # Wait up to 2 minutes for the fallback model
                    max_wait = 120
                    start_time = time.monotonic()
                    while time.monotonic() - start_time < max_wait:
                        wait_time = await self.rate_limiter.get_wait_time_async(fallback_model_id)
                        if wait_time <= 0:
                            logger.info(
                                f"Fallback model {fallback_model_id} is now available after waiting")
                            break
                        logger.debug(
                            f"Fallback still rate limited, waiting {min(wait_time, 5)}s...")
                        # Wait the lesser of wait_time or 5s
                        await asyncio.sleep(min(wait_time, 5))

                    # Even if we're still rate limited, we'll try anyway since this is our last resort
                    await self.rate_limiter.wait_if_needed_async(fallback_model_id)

            # Check rate limit *before* attempting the call (using model_id)
            if self.rate_limiter:
                is_allowed = self.rate_limiter.is_allowed(model_id)
                if not is_allowed:
                    wait_time = await self.rate_limiter.get_wait_time_async(model_id)
                    if 0 < wait_time <= max_wait_threshold:
                        logger.debug(
                            f"Rate limit hit for {model_id}. Waiting {wait_time:.2f}s...")
                        await self.rate_limiter.wait_if_needed_async(model_id)
                    else:
                        logger.warning(
                            f"Rate limit wait for {model_id} ({wait_time:.2f}s) > threshold ({max_wait_threshold}s). Skipping.")
                        rate_limited_models.add(model_id)
                        continue
            else:
                logger.debug(
                    "Rate limiter not available, proceeding without check.")

            model_attempt = 0
            max_model_retries = 2

            while model_attempt < max_model_retries and total_attempts < retries:
                total_attempts += 1
                model_attempt += 1
                task_start_time = time.monotonic()
                logger.debug(
                    f"Attempt {model_attempt}/{max_model_retries} for model {model_id}, total attempt {total_attempts}/{retries}")

                try:
                    if self.rate_limiter:
                        logger.debug(
                            f"[{model_id}] Attempt {model_attempt}: Checking/Waiting for rate limit...")
                        await self.rate_limiter.wait_if_needed_async(model_id)
                        logger.debug(
                            f"[{model_id}] Attempt {model_attempt}: Rate limit check passed/wait finished.")

                    # --- NEW API CALL ---
                    api_call_timeout_seconds = 120
                    logger.debug(
                        f"[{model_id}] Attempt {model_attempt}: Calling generate_content_async (timeout={api_call_timeout_seconds}s)...")

                    # Define generation config using new types if necessary
                    # Check google-genai docs for exact config class & parameters
                    gen_config = google_genai.types.GenerateContentConfig(
                        temperature=0.1,
                        automatic_function_calling={'disable': True},
                        # max_output_tokens=... # Add if needed, defaults might be sufficient
                        # response_mime_type="application/json" # If supported and desired
                    )

                    # Use the client directly, assuming generate_content_async exists
                    # The `contents` parameter expects an iterable (like a list)
                    response = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            # Prepend models/ prefix if client expects it
                            model=f'models/{model_id}',
                            contents=[full_prompt],  # Pass prompt in a list
                            config=gen_config
                            # safety_settings=... # Add if needed
                        ),
                        timeout=api_call_timeout_seconds
                    )
                    call_duration = time.monotonic() - task_start_time
                    logger.debug(
                        f"[{model_id}] Attempt {model_attempt}: generate_content_async call succeeded in {call_duration:.2f}s.")
                    # --- END NEW API CALL ---

                    logger.debug(
                        f"[RESPONSE] Full response from {model_id}: {response}")
                    # Check new response structure
                    usage_metadata = getattr(response, 'usage_metadata', None)
                    if usage_metadata:
                        logger.debug(
                            f"[USAGE] {model_id} usage metadata: {usage_metadata}")
                    # ... (Rate limiter registration call remains the same) ...
                    if self.rate_limiter:
                        await self.rate_limiter.register_call_async(model_id)

                    # --- Process Response (New Structure) ---
                    generated_text = None
                    # Check common ways to access text in new SDK response objects
                    if hasattr(response, 'text'):
                        generated_text = response.text
                    elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
                        generated_text = "".join(
                            part.text for part in response.parts if hasattr(part, 'text'))
                    # Add other checks based on actual response object structure if needed

                    if generated_text:
                        # ... (JSON parsing logic remains the same) ...
                        parsed_json, error = parse_json(generated_text)
                        if not error:
                            logger.info(
                                f"Successfully parsed JSON response from {model_id}")
                            return parsed_json
                        else:
                            logger.warning(
                                f"JSON parsing failed for {model_id}: {error}")
                            # Try extraction
                            extracted_json, extract_error = extract_json_from_text(
                                generated_text)
                            if not extract_error:
                                parsed_json, error = parse_json(extracted_json)
                                if not error and parsed_json:
                                    logger.info(
                                        f"Successfully parsed extracted JSON response from {model_id}")
                                    return parsed_json
                                else:
                                    logger.warning(
                                        f"Extracted JSON parsing failed for {model_id}: {error or 'Empty JSON'}")
                            else:
                                logger.warning(
                                    f"JSON extraction failed for {model_id}: {extract_error}")
                        # If parsing/extraction failed, continue to next attempt/model
                    else:
                        # Check for prompt feedback / finish reason if no text
                        finish_reason = getattr(response, 'prompt_feedback', None) or getattr(
                            response, 'finish_reason', None)
                        logger.warning(
                            f"Received no generated text from {model_id}. Finish reason/Feedback: {finish_reason}")

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout error caught for {model_id} Attempt {model_attempt}. Breaking inner loop.")
                    break  # Exit inner loop for this model
                except google.api_core.exceptions.ResourceExhausted as e:  # Catch specific rate limit exception
                    logger.warning(
                        f"[{model_id}] ResourceExhausted (Rate Limit/Quota) (Attempt {model_attempt}/{max_model_retries}): {e}")
                    if model_attempt < max_model_retries and total_attempts < retries:
                        if self.rate_limiter:
                            logger.warning(
                                f"Waiting based on rate limiter for {model_id}...")
                            await self.rate_limiter.wait_if_needed_async(model_id)
                            logger.info(
                                f"Finished rate limit wait for {model_id}. Retrying...")
                            continue
                        else:  # Fallback sleep
                            await asyncio.sleep(current_delay * (random.random()*0.5+0.5))
                            continue
                    else:
                        logger.error(
                            f"Rate limit/Quota error with {model_id}, retries exhausted. Switching model.")
                        # Mark as rate limited before breaking
                        rate_limited_models.add(model_id)
                        break  # Exit inner loop
                except google.api_core.exceptions.GoogleAPIError as e:  # Catch other general Google API errors
                    logger.warning(
                        f"[{model_id}] Google API error (Attempt {model_attempt}/{max_model_retries}): {e}", exc_info=False)
                    # Retry logic for general API errors
                    if model_attempt < max_model_retries and total_attempts < retries:
                        await asyncio.sleep(current_delay * (random.random()*0.5+0.5))
                        continue
                    else:
                        break  # Exhausted retries
                except Exception as e:
                    logger.warning(
                        f"Generic error (Attempt {model_attempt}/{max_model_retries}) using {model_id}: {e}", exc_info=True)
                    # Exponential backoff for non-rate-limit errors
                    if model_attempt < max_model_retries and total_attempts < retries:
                        jitter = random.random() * 0.5 + 0.5
                        sleep_time = current_delay * jitter
                        logger.debug(
                            f"Non-rate-limit error. Waiting {sleep_time:.2f}s before retrying {model_id}.")
                        await asyncio.sleep(sleep_time)
                        current_delay *= 1.5
                        continue
                    else:
                        logger.error(
                            f"Failed on {model_id} after {model_attempt} attempts due to non-rate-limit error: {e}")
                        break  # Exit inner loop

            # Break outer loop if result was returned successfully in inner loop
            # (This check is technically redundant due to `return` statement but safe)
            # if parsed_json: break

        logger.error(
            f"Failed to generate text after trying all models and {retries} total attempts.")
        return None

    async def analyze_articles_with_prompt(self, articles_data: List[Dict[str, Any]],
                                           prompt_file_path: str,
                                           model_name: str,  # Ensure this is a valid model ID like 'gemini-1.5-flash-latest'
                                           system_instruction: Optional[str] = None,
                                           temperature: float = 0.2,
                                           max_output_tokens: int = 8192,
                                           retries: int = 3,
                                           initial_delay: float = 1.0) -> Optional[str]:
        """
        Delegates article analysis to the generator module.
        Ensures the model name/ID passed is compatible with the generator module's expectations.
        """
        model_id = model_name.split('/')[-1]
        # Define default system instruction here
        default_system_instruction = """The AI agent should adopt an academic persona—specifically."""
        actual_system_instruction = system_instruction if system_instruction is not None else default_system_instruction
        logger.info(
            f"Delegating article analysis to generator module (model: {model_id})")
        return await generator_analyze_articles(
            client=self.client,  # Pass the initialized client
            articles_data=articles_data,
            prompt_file_path=prompt_file_path,
            model_name=model_id,  # Pass the ID
            system_instruction=actual_system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # Retries now handled within GeminiClient methods calling this
            # retries=retries,
            # initial_delay=initial_delay,
            rate_limiter=self.rate_limiter
        )

    async def generate_essay_from_prompt(self,
                                         full_prompt_text: str,
                                         model_name: Optional[str] = None,
                                         system_instruction: Optional[str] = None,
                                         temperature: float = 0.7,
                                         max_output_tokens: int = 8192,
                                         save_debug_info: bool = True,
                                         debug_info_prefix: str = "essay_prompt") -> Optional[str]:
        """
        Delegates essay generation to the generator module.
        Ensures the model name/ID passed is compatible with the generator module's expectations.
        """
        default_gen_model_id = os.getenv("GEMINI_FLASH_THINKING_MODEL",
                                         "gemini-1.5-flash-latest").split('/')[-1]
        effective_model_id = model_name.split(
            '/')[-1] if model_name else default_gen_model_id
        # Define default system instruction here
        default_essay_system_instruction = """You are an expert analytical writer tasked with synthesizing information. Follow the instructions precisely and generate a coherent, well-structured text based *only* on the provided context."""
        actual_system_instruction = system_instruction or default_essay_system_instruction
        logger.info(
            f"Delegating essay generation to generator module (model: {effective_model_id})")
        return await generator_generate_text(
            client=self.client,  # Pass the initialized client
            full_prompt_text=full_prompt_text,
            model_name=effective_model_id,  # Pass the ID
            system_instruction=actual_system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            rate_limiter=self.rate_limiter,
            save_debug_info=save_debug_info,
            debug_info_prefix=debug_info_prefix
        )
