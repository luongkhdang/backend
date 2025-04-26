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

    # --- REMOVED Hardcoded Model Config --- #
    # GENERATION_MODELS_CONFIG = { ... }
    # FALLBACK_MODEL = '...'
    # FALLBACK_MODEL_RPM = ...
    # ALL_MODEL_RPMS = { ... }
    # _log_initial_rate_limit = ...
    # --- END REMOVED Hardcoded Model Config --- #

    DEFAULT_PROMPT_PATH = "src/prompts/entity_extraction_prompt.txt"

    # Default max wait time for rate limit cooldown in seconds
    DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS = 40

    def __init__(self):
        """
        Initialize the google-genai API client.
        Loads API key, model IDs, RPM limits, and token limits from environment variables.
        Initializes the client (forcing v1alpha API version) and the rate limiter.
        """
        # Load environment variables
        load_dotenv()

        # --- API Key --- #
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = "GEMINI_API_KEY environment variable not set"
            logger.critical(error_msg)
            raise ValueError(error_msg)

        # --- Initialize the google-genai client (Force v1alpha for potential Thinking use) --- #
        try:
            # Force v1alpha since Thinking might be attempted
            http_opts = {'api_version': 'v1alpha'}
            self.client = google_genai.Client(
                api_key=api_key, http_options=http_opts)
            logger.info(
                "google-genai client initialized successfully (Using forced API version: v1alpha).")
        except Exception as e:
            error_msg = f"Failed to initialize google-genai client (forcing v1alpha): {e}"
            logger.critical(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # --- Embedding Model Configuration --- #
        self.embedding_model_id = os.getenv(
            "GEMINI_EMBEDDING_MODEL_ID", "models/text-embedding-004")
        self.embedding_rpm_limit = int(
            os.getenv("GEMINI_EMBEDDING_RPM", "1500"))
        self.default_task_type = os.getenv(
            "GEMINI_EMBEDDING_TASK_TYPE", "CLUSTERING")
        self.embedding_input_token_limit = int(
            os.getenv("GEMINI_EMBEDDING_INPUT_TOKEN_LIMIT", "2048"))

        # --- Generation Model Configuration (Read RPMs only) --- #
        self.gen_input_token_limit = int(
            os.getenv("GEMINI_GEN_INPUT_TOKEN_LIMIT", "1048576"))
        self.gen_output_token_limit = int(
            os.getenv("GEMINI_GEN_OUTPUT_TOKEN_LIMIT", "8192"))

        # Dictionary to store model RPM limits for RateLimiter
        model_rpm_limits = {}

        # Embedding model RPM limit
        if self.embedding_model_id:
            # Use full ID for embedding key consistency
            model_rpm_limits[self.embedding_model_id] = self.embedding_rpm_limit
            logger.debug(
                f"Loaded Embedding Model: {self.embedding_model_id} (RPM: {self.embedding_rpm_limit})")

        # Generation models (load ID and RPM)
        gen_model_configs = [
            ("GEMINI_FLASH_EXP", "gemini-2.0-flash-exp", 10),
            ("GEMINI_FLASH", "gemini-2.0-flash", 15),
            ("GEMINI_FLASH_LITE", "gemini-2.0-flash-lite", 30),
            ("GEMINI_PREVIEW", "gemini-2.5-flash-preview-04-17", 10)
        ]

        self.available_gen_models = []  # Store available model IDs (short IDs)

        for prefix, default_id, default_rpm in gen_model_configs:
            model_id = os.getenv(f"{prefix}_MODEL_ID", default_id)
            rpm_str = os.getenv(f"{prefix}_RPM")
            # Removed reading of THINKING/GROUNDING flags here

            if model_id and rpm_str:
                try:
                    rpm = int(rpm_str)
                    # Store RPM for rate limiter (use short ID)
                    model_rpm_limits[model_id] = rpm
                    self.available_gen_models.append(model_id)
                    logger.debug(
                        f"Loaded Generation Model: {model_id} (RPM: {rpm})")
                except ValueError:
                    logger.warning(
                        f"Invalid RPM value '{rpm_str}' for {prefix}_RPM env var. Skipping model {model_id}.")
            elif model_id:
                logger.warning(
                    f"{prefix}_RPM environment variable not set for model {model_id}. Using hardcoded default RPM: {default_rpm}")
                model_rpm_limits[model_id] = default_rpm
                self.available_gen_models.append(model_id)
                logger.debug(
                    f"Loaded Generation Model: {model_id} (RPM: {default_rpm})")

        # --- Preferred Model Order and Fallback --- #
        pref_1 = os.getenv("GEMINI_MODEL_PREF_1", "gemini-2.0-flash-exp")
        pref_2 = os.getenv("GEMINI_MODEL_PREF_2", "gemini-2.0-flash")
        pref_3 = os.getenv("GEMINI_MODEL_PREF_3", "gemini-2.0-flash-lite")
        self.fallback_model_id = os.getenv(
            "GEMINI_FALLBACK_MODEL_ID", "gemini-2.0-flash")

        self.preferred_model_ids = []
        for pref_model in [pref_1, pref_2, pref_3]:
            if pref_model in self.available_gen_models:
                self.preferred_model_ids.append(pref_model)
            else:
                logger.warning(
                    f"Preferred model {pref_model} not found in loaded configs. Skipping.")

        if self.fallback_model_id not in self.available_gen_models:
            logger.warning(
                f"Fallback model {self.fallback_model_id} specified is not available or has no RPM limit. Attempting to use first preferred model as fallback.")
            if self.preferred_model_ids:
                self.fallback_model_id = self.preferred_model_ids[0]
                logger.debug(f"Using {self.fallback_model_id} as fallback.")
            else:
                logger.error(
                    "No valid generation models configured. Rate limiting/fallback ineffective.")
                self.fallback_model_id = "gemini-2.0-flash"  # Dummy
                if self.fallback_model_id not in model_rpm_limits:
                    # Add dummy RPM
                    model_rpm_limits[self.fallback_model_id] = 15
                    self.available_gen_models.append(self.fallback_model_id)

        logger.debug(f"Preferred model order: {self.preferred_model_ids}")
        logger.debug(f"Fallback model: {self.fallback_model_id}")

        # --- Rate Limit Configuration --- #
        self.max_rate_limit_wait_seconds = float(os.getenv(
            "GEMINI_MAX_WAIT_SECONDS", str(self.DEFAULT_MAX_RATE_LIMIT_WAIT_SECONDS)))
        self.emergency_fallback_wait_seconds = float(
            os.getenv("EMERGENCY_FALLBACK_WAIT_SECONDS", "120.0"))
        self.reserved_fallback_slots = int(
            os.getenv("GEMINI_RESERVED_FALLBACK_SLOTS", "2"))
        logger.debug(
            f"Rate Limit Config: Max Wait={self.max_rate_limit_wait_seconds}s, Emergency Wait={self.emergency_fallback_wait_seconds}s, Reserved Slots={self.reserved_fallback_slots}")

        # --- Initialize Rate Limiter --- #
        if RateLimiter:
            # Pass the consolidated RPM limits (embedding + generation)
            self.rate_limiter = RateLimiter(
                model_rpm_limits, client_instance=self)
            logger.debug(
                f"GeminiClient initialized with RateLimiter. Config: {model_rpm_limits}")
        else:
            self.rate_limiter = None
            logger.warning("GeminiClient initialized WITHOUT RateLimiter.")

        # --- Load Prompt Template --- #
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self, prompt_path: str = DEFAULT_PROMPT_PATH) -> Optional[str]:
        """Loads the prompt template from the specified file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            logger.debug(
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
        # Use stored ID directly (contains models/)
        model_to_use_for_api = self.embedding_model_id
        # Short name for rate limiter
        # Use the ID with prefix for rate limiter key consistency
        model_to_use_for_limiter = self.embedding_model_id

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
                    logger.debug(
                        f"Retry attempt {attempt+1}/{retries} for embedding generation using {model_to_use_for_api}")

                # Apply rate limiting if a rate limiter is provided (use short ID)
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed(model_to_use_for_limiter)

                # Generate embedding using the new client and method (use full name/ID directly)
                result = self.client.models.embed_content(
                    model=model_to_use_for_api,
                    contents=text
                )

                # Log usage metadata if available
                usage_metadata = getattr(result, 'usage_metadata', None) or result.get(
                    'usage_metadata', None) if isinstance(result, dict) else None
                if usage_metadata:
                    logger.debug(
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

        # --- Model Selection Logic (Uses instance vars) --- #
        if model_override and model_override in self.available_gen_models:
            all_models_to_try_ids = [
                model_override] + self.preferred_model_ids + [self.fallback_model_id]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))
        elif model_override:
            logger.warning(
                f"Model override '{model_override}' not found in available models. Using default preference order.")
            all_models_to_try_ids = self.preferred_model_ids + \
                [self.fallback_model_id]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))
        else:
            all_models_to_try_ids = self.preferred_model_ids + \
                [self.fallback_model_id]
            all_models_to_try_ids = list(dict.fromkeys(all_models_to_try_ids))

        logger.debug(f"SYNC: Model sequence to try: {all_models_to_try_ids}")
        # --- End Model Selection --- #

        current_delay = initial_delay
        total_attempts = 0
        rate_limited_models = set()

        for model_id in all_models_to_try_ids:
            # logger.debug(f"SYNC: Attempting text generation with model: {model_id}") # Removed debug log

            # --- Emergency Bailout (sync version) --- #
            if len(rate_limited_models) >= len(all_models_to_try_ids) - 1 and model_id == self.fallback_model_id:
                logger.warning(
                    f"SYNC: Forcing use of fallback model {self.fallback_model_id}...")
                if self.rate_limiter:
                    # Wait potentially long time for fallback (sync)
                    max_wait = 120
                    start_time = time.monotonic()
                    while time.monotonic() - start_time < max_wait:
                        wait_time = self.rate_limiter.get_wait_time(
                            self.fallback_model_id)
                        if wait_time <= 0:
                            break
                        time.sleep(min(wait_time, 5))
                    # self.rate_limiter.wait_if_needed(self.fallback_model_id) # This wait happens below anyway
            # --- End Bailout ---

            # --- Rate Limit Check (sync) ---
            if self.rate_limiter:
                is_allowed = self.rate_limiter.is_allowed(model_id)
                if not is_allowed:
                    wait_time = self.rate_limiter.get_wait_time(model_id)
                    if 0 < wait_time <= max_wait_threshold:
                        # logger.debug(f"SYNC: Rate limit hit for {model_id}. Waiting {wait_time:.2f}s...") # Removed debug log
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
                # logger.debug(f"SYNC: Attempt {model_attempt}/{max_model_retries} for model {model_id}, total attempt {total_attempts}/{retries}") # Removed debug log

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

                    # logger.debug(f"SYNC: [RESPONSE] Full response from {model_id}: {response}") # Removed debug log
                    usage_metadata = getattr(response, 'usage_metadata', None)
                    if usage_metadata:
                        # logger.debug(f"SYNC: [USAGE] {model_id} usage metadata: {usage_metadata}") # Removed debug log
                        pass  # Keep structure, log removed

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

    def set_max_rate_limit_wait_seconds(self, seconds: float) -> None:
        """Set the maximum time to wait for rate limiting before giving up."""
        self.max_rate_limit_wait_seconds = seconds
        logger.debug(f"Set max_rate_limit_wait_seconds to {seconds}s")

    def set_emergency_wait_seconds(self, seconds: float) -> None:
        """Set the emergency wait time for extreme rate limiting situations."""
        self.emergency_fallback_wait_seconds = seconds
        logger.debug(f"Set emergency_fallback_wait_seconds to {seconds}s")

    async def generate_text_with_prompt_async(self, article_content: str, processing_tier: int,
                                              retries: int = 6, initial_delay: float = 1.0,
                                              model_override: Optional[str] = None,
                                              fallback_model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Async version of generate_text_with_prompt. Generates structured entity data from an article.
        Uses model selection, fallback, rate limiting with max wait times, and retries.
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

        # Define model preference order
        # Handle model IDs (ensure they are short IDs, not prefixed with "models/")
        def clean_model_id(mid: Optional[str]) -> Optional[str]:
            return mid.split('/')[-1] if mid else None

        model_override_id = clean_model_id(model_override)
        fallback_model_id = clean_model_id(fallback_model)

        # Start with override if provided and valid
        all_models_to_try_ids = []
        if model_override_id and model_override_id in self.available_gen_models:
            all_models_to_try_ids.append(model_override_id)
        elif model_override_id:
            logger.warning(
                f"Model override '{model_override_id}' not found in available models. Ignoring.")

        # Add the passed fallback model next, if it's valid and different from override
        if fallback_model_id and fallback_model_id in self.available_gen_models and fallback_model_id != model_override_id:
            all_models_to_try_ids.append(fallback_model_id)
        elif fallback_model_id and fallback_model_id != model_override_id:
            logger.warning(
                f"Provided fallback model '{fallback_model_id}' not found in available models. Ignoring.")

        # Add remaining available models (excluding those already added)
        for model in self.available_gen_models:
            clean_id = clean_model_id(model)
            if clean_id not in all_models_to_try_ids:
                all_models_to_try_ids.append(clean_id)

        # Ensure the instance fallback model is last (if not already included)
        instance_fallback_id = clean_model_id(self.fallback_model_id)
        if instance_fallback_id and instance_fallback_id not in all_models_to_try_ids:
            all_models_to_try_ids.append(instance_fallback_id)
        elif not instance_fallback_id:
            logger.error(
                "Instance fallback model ID is not configured correctly!")

        logger.debug(f"Model sequence to try: {all_models_to_try_ids}")
        if not all_models_to_try_ids:
            logger.error("No valid models available to attempt generation.")
            return None

        current_delay = initial_delay
        total_attempts_across_models = 0
        max_total_attempts = retries  # Overall limit across all models
        parsed_json_result = None

        for model_id in all_models_to_try_ids:
            logger.info(f"Attempting text generation with model: {model_id}")

            # Max retries per specific model for non-rate-limit errors
            max_model_retries = 2
            model_attempt = 0

            while model_attempt < max_model_retries and total_attempts_across_models < max_total_attempts:
                total_attempts_across_models += 1
                model_attempt += 1
                task_start_time = time.monotonic()
                logger.debug(
                    f"[{model_id}] Model Attempt {model_attempt}/{max_model_retries}, Total Attempt {total_attempts_across_models}/{max_total_attempts}")

                try:
                    # --- Start: Rate Limit Check using Atomic Reservation ---
                    slot_reserved = False
                    if self.rate_limiter:
                        logger.debug(
                            f"[{model_id}] Attempting to reserve rate limit slot...")
                        # Try to atomically reserve a slot for this model
                        slot_reserved = await self.rate_limiter.reserve_and_register_call_async(model_id)

                        if not slot_reserved:
                            # No slot available, try waiting if within threshold
                            wait_time = await self.rate_limiter.get_wait_time_async(model_id)
                            if 0 < wait_time <= self.max_rate_limit_wait_seconds:
                                logger.debug(
                                    f"[{model_id}] Rate limit hit. Waiting {wait_time:.2f}s...")
                                await asyncio.sleep(wait_time)
                                # Try reserving again after waiting
                                slot_reserved = await self.rate_limiter.reserve_and_register_call_async(model_id)
                                if not slot_reserved:
                                    logger.warning(
                                        f"[{model_id}] Still rate limited after waiting. Skipping attempt for this model.")
                                    # Break inner loop to try the next model in the preference list
                                    break
                            else:
                                logger.warning(
                                    f"[{model_id}] Wait time {wait_time:.2f}s exceeds threshold or is zero. Skipping attempt for this model.")
                                # Break inner loop to try the next model
                                break
                        else:
                            logger.debug(
                                f"[{model_id}] Rate limit slot successfully reserved.")
                    else:
                        # No rate limiter configured, proceed without reservation
                        slot_reserved = True

                    # If no slot could be reserved/acquired, break the inner loop to try the next model
                    if not slot_reserved:
                        continue  # Skip the API call for this model attempt
                    # --- End: Rate Limit Check using Atomic Reservation ---

                    # --- API Call (Slot is reserved) ---
                    api_call_timeout_seconds = 120  # Timeout for the API call itself
                    gen_config = google_genai.types.GenerateContentConfig(
                        temperature=0.1,
                        automatic_function_calling={'disable': True}
                    )

                    response = await asyncio.wait_for(
                        self.client.aio.models.generate_content(
                            # Client expects prefix
                            model=f'models/{model_id}',
                            contents=[full_prompt],
                            config=gen_config
                        ),
                        timeout=api_call_timeout_seconds
                    )
                    call_duration = time.monotonic() - task_start_time
                    logger.debug(
                        f"[{model_id}] API call succeeded in {call_duration:.2f}s.")

                    # No need to register call here, it was done atomically during reservation
                    # if self.rate_limiter:
                    #    await self.rate_limiter.register_call_async(model_id)
                    # --- End API Call ---

                    # --- Process Response ---
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
                                f"Successfully parsed JSON response from {model_id}")
                            parsed_json_result = parsed_json
                            break  # Success! Exit inner loop
                        else:
                            extracted_json, extract_error = extract_json_from_text(
                                generated_text)
                            if not extract_error:
                                parsed_json, error = parse_json(extracted_json)
                                if not error and parsed_json:
                                    logger.info(
                                        f"Successfully parsed EXTRACTED JSON response from {model_id}")
                                    parsed_json_result = parsed_json
                                    break  # Success! Exit inner loop
                                else:
                                    logger.warning(
                                        f"[{model_id}] Extracted JSON parsing failed: {error or 'Empty JSON'}")
                            else:
                                logger.warning(
                                    f"[{model_id}] JSON extraction failed: {extract_error}")
                            # If parsing/extraction failed, continue to next attempt for this model
                    else:
                        finish_reason = getattr(response, 'prompt_feedback', None) or getattr(
                            response, 'finish_reason', None)
                        logger.warning(
                            f"[{model_id}] Received no generated text. Finish reason/Feedback: {finish_reason}")
                    # --- End Process Response ---

                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{model_id}] API call timeout after {api_call_timeout_seconds}s (Attempt {model_attempt}/{max_model_retries}). Retrying if possible.")
                    # Continue to next attempt for this model if retries remain
                except google.api_core.exceptions.ResourceExhausted as e:
                    logger.warning(
                        f"[{model_id}] ResourceExhausted (Rate Limit/Quota) during API call (Attempt {model_attempt}/{max_model_retries}). Skipping model.")
                    # Break inner loop immediately, move to next model
                    break
                except google.api_core.exceptions.GoogleAPIError as e:
                    logger.warning(
                        f"[{model_id}] Google API error during call (Attempt {model_attempt}/{max_model_retries}): {e}", exc_info=False)
                    # Retry for this model if attempts remain
                    if model_attempt < max_model_retries:
                        await asyncio.sleep(current_delay * (random.random()*0.5+0.5))
                        current_delay *= 1.5
                        continue  # Retry with same model
                    else:
                        logger.error(
                            f"[{model_id}] Exhausted retries due to GoogleAPIError. Skipping model.")
                        break  # Skip to next model
                except Exception as e:
                    logger.warning(
                        f"[{model_id}] Generic error during call (Attempt {model_attempt}/{max_model_retries}): {e}", exc_info=True)
                    # Retry for this model if attempts remain
                    if model_attempt < max_model_retries:
                        await asyncio.sleep(current_delay * (random.random()*0.5+0.5))
                        current_delay *= 1.5
                        continue  # Retry with same model
                    else:
                        logger.error(
                            f"[{model_id}] Exhausted retries due to generic error. Skipping model.")
                        break  # Skip to next model

            # If we successfully got a result, break the outer loop
            if parsed_json_result is not None:
                logger.info(f"Successfully obtained result using {model_id}.")
                break

        # --- Emergency Fallback Attempt ---
        if parsed_json_result is None and total_attempts_across_models >= max_total_attempts:
            logger.warning(
                f"All attempts failed after {max_total_attempts} total tries across models. Triggering emergency fallback.")
            emergency_model_id = clean_model_id(
                self.fallback_model_id)  # Use the instance fallback
            if emergency_model_id and self.rate_limiter:
                logger.info(
                    f"Attempting emergency fallback with {emergency_model_id} using emergency wait time: {self.emergency_fallback_wait_seconds}s")
                try:
                    await self.rate_limiter.wait_if_needed_async(emergency_model_id, max_wait_seconds=self.emergency_fallback_wait_seconds)
                    if self.rate_limiter.is_allowed(emergency_model_id):
                        logger.info(
                            f"[{emergency_model_id}] Emergency Rate limit check passed. Making final API call attempt.")
                        # --- Final API Call Attempt ---
                        api_call_timeout_seconds = 120
                        gen_config = google_genai.types.GenerateContentConfig(
                            temperature=0.1, automatic_function_calling={'disable': True})
                        response = await asyncio.wait_for(
                            self.client.aio.models.generate_content(
                                model=f'models/{emergency_model_id}', contents=[full_prompt], config=gen_config),
                            timeout=api_call_timeout_seconds
                        )
                        await self.rate_limiter.register_call_async(emergency_model_id)
                        # --- Process Emergency Response ---
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
                                    f"Successfully parsed JSON response from EMERGENCY FALLBACK {emergency_model_id}")
                                parsed_json_result = parsed_json
                            else:  # Try extraction on fallback
                                extracted_json, extract_error = extract_json_from_text(
                                    generated_text)
                                if not extract_error:
                                    parsed_json, error = parse_json(
                                        extracted_json)
                                    if not error and parsed_json:
                                        logger.info(
                                            f"Successfully parsed EXTRACTED JSON response from EMERGENCY FALLBACK {emergency_model_id}")
                                        parsed_json_result = parsed_json
                        if not parsed_json_result:
                            logger.error(
                                f"[{emergency_model_id}] Emergency fallback attempt failed to produce parsable JSON.")
                        # --- End Process Emergency Response ---
                    else:
                        logger.error(
                            f"[{emergency_model_id}] Still rate limited even after emergency wait. Cannot proceed.")
                except Exception as e:
                    logger.error(
                        f"[{emergency_model_id}] Error during emergency fallback attempt: {e}", exc_info=True)
            elif not emergency_model_id:
                logger.error(
                    "Cannot perform emergency fallback: Instance fallback model ID is not configured.")
            elif not self.rate_limiter:
                logger.warning(
                    "Cannot perform emergency fallback rate limit check: RateLimiter not available.")
        # --- End Emergency Fallback Attempt ---

        if parsed_json_result is None:
            logger.error(
                f"Failed to generate text after trying all models and {total_attempts_across_models} total attempts (max: {max_total_attempts}).")

        return parsed_json_result

    async def analyze_articles_with_prompt(self, articles_data: List[Dict[str, Any]],
                                           prompt_file_path: str,
                                           model_name: str,  # Expecting model ID like 'gemini-1.5-flash-latest'
                                           system_instruction: Optional[str] = None,
                                           temperature: float = 0.2,
                                           max_output_tokens: Optional[int] = None) -> Optional[str]:
        """
        Delegates article analysis to the generator module.
        The generator module now handles retry logic including fallback without thinking/grounding.
        Uses the configured output token limit as default if not specified.
        """
        # Determine effective output token limit
        effective_max_output_tokens = max_output_tokens if max_output_tokens is not None else self.gen_output_token_limit

        model_id = model_name.split('/')[-1]  # Use short ID
        # Removed lookup for thinking/grounding flags

        # Define default system instruction here
        default_system_instruction = """The AI agent should adopt an academic persona—specifically."""
        actual_system_instruction = system_instruction if system_instruction is not None else default_system_instruction
        logger.info(
            # Removed flags from log
            f"Delegating article analysis to generator module (model: {model_id})")
        return await generator_analyze_articles(
            client=self.client,  # Pass the initialized client
            articles_data=articles_data,
            prompt_file_path=prompt_file_path,
            model_name=model_id,  # Pass the short ID
            system_instruction=actual_system_instruction,
            temperature=temperature,
            max_output_tokens=effective_max_output_tokens,
            # Removed thinking_enabled/grounding_enabled flags
            rate_limiter=self.rate_limiter
        )

    async def generate_essay_from_prompt(self,
                                         full_prompt_text: str,
                                         model_name: Optional[str] = None,
                                         system_instruction: Optional[str] = None,
                                         temperature: float = 0.7,
                                         max_output_tokens: Optional[int] = None,
                                         save_debug_info: bool = True,
                                         debug_info_prefix: str = "essay_prompt") -> Optional[str]:
        """
        Delegates essay generation to the generator module.
        The generator module now handles retry logic including fallback without thinking/grounding.
        Uses the configured preferred model and output token limit as defaults.
        """
        # Determine effective model and output token limit
        default_gen_model_id = self.preferred_model_ids[
            0] if self.preferred_model_ids else self.fallback_model_id
        effective_model_id = model_name.split(
            '/')[-1] if model_name else default_gen_model_id
        effective_max_output_tokens = max_output_tokens if max_output_tokens is not None else self.gen_output_token_limit

        # Removed lookup for thinking/grounding flags

        # Define default system instruction here
        default_essay_system_instruction = """You are an expert analytical writer tasked with synthesizing information. Follow the instructions precisely and generate a coherent, well-structured text based *only* on the provided context."""
        actual_system_instruction = system_instruction or default_essay_system_instruction
        logger.info(
            # Removed flags from log
            f"Delegating essay generation to generator module (model: {effective_model_id})")
        return await generator_generate_text(
            client=self.client,  # Pass the initialized client
            full_prompt_text=full_prompt_text,
            model_name=effective_model_id,  # Pass the short ID
            system_instruction=actual_system_instruction,
            temperature=temperature,
            max_output_tokens=effective_max_output_tokens,
            # Removed thinking_enabled/grounding_enabled flags
            rate_limiter=self.rate_limiter,
            save_debug_info=save_debug_info,
            debug_info_prefix=debug_info_prefix
        )
