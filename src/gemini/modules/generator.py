"""
generator.py - Gemini API Text Generation Module

This module provides specialized text generation functionality using Google's Gemini API,
particularly for analyzing multiple articles or generating text from a single large prompt.

Exported functions:
- analyze_articles_with_prompt(): Analyzes a list of articles using a specified prompt template.
  - Returns Optional[str]: Structured JSON response or None if analysis fails.
- generate_text_from_prompt(): Generates text from a provided prompt string.
  - Returns Optional[str]: Generated text response or None if generation fails.

Related files:
- src/gemini/gemini_client.py: Main client that uses this module.
- src/steps/step4.py: Uses analyze_articles_with_prompt indirectly via GeminiClient.
- src/steps/step5.py: Uses generate_text_from_prompt indirectly via GeminiClient.
- src/prompts/step4.txt: Contains the prompt template for article analysis.
- src/prompts/haystack_prompt.txt: Contains the prompt template for essay generation.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def analyze_articles_with_prompt(
    articles_data: List[Dict[str, Any]],
    prompt_file_path: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 65536,
    retries: int = 1,
    initial_delay: float = 1.0,
    rate_limiter=None
) -> Optional[str]:
    """
    Analyzes a list of articles using a specified prompt template and returns the text response.
    Will not retry on failure (to avoid wasting quota).

    This method:
    1. Loads a prompt template from the specified file path
    2. Injects the article data as JSON into the prompt
    3. Calls Gemini API with appropriate system instruction
    4. Returns the full text response without any JSON parsing

    Args:
        articles_data (List[Dict[str, Any]]): List of prepared article dictionaries.
        prompt_file_path (str): Path to the prompt template file (e.g., `src/prompts/step4.txt`).
        model_name (str): Gemini model to use for analysis.
        system_instruction (Optional[str]): Optional system instruction to guide the model.
        temperature (float): Temperature parameter for generation (0.0-1.0).
        max_output_tokens (int): Maximum tokens to generate in the response.
        retries (int): Number of retry attempts if analysis fails (default: 1 = no retries).
        initial_delay (float): Initial delay in seconds before first retry (not used).
        rate_limiter: Optional rate limiter object to manage API call frequency.

    Returns:
        Optional[str]: Full text response if successful, None if failed.
    """
    logger.info(
        f"Analyzing {len(articles_data)} snippets with prompt from {prompt_file_path}")

    # Specify API key with environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    genai.configure(api_key=api_key)

    # Load the prompt template
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        if not prompt_template:
            logger.error(f"Empty prompt template in {prompt_file_path}")
            return None
    except (IOError, FileNotFoundError) as e:
        logger.error(
            f"Failed to load prompt template from {prompt_file_path}: {e}")
        return None

    # Serialize the article data to JSON for the prompt
    try:
        articles_json = json.dumps(
            articles_data, separators=(',', ':'), ensure_ascii=True)
    except (TypeError, OverflowError) as e:
        logger.error(f"Failed to serialize article data to JSON: {e}")
        return None

    # Inject article data into the prompt
    full_prompt_text = prompt_template.replace(
        "{INPUT_DATA_JSON}", articles_json)

    logger.debug(f"Full prompt size: {len(full_prompt_text)} characters")

    # Save full prompt for debugging
    output_dir = "src/output/"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_debug_filename = f"full_prompt_{timestamp}.txt"
    prompt_debug_path = os.path.join(output_dir, prompt_debug_filename)

    try:
        with open(prompt_debug_path, 'w', encoding='utf-8') as f:
            f.write(full_prompt_text)
        logger.info(f"Saved full prompt for debugging to {prompt_debug_path}")
    except IOError as e:
        logger.warning(f"Failed to save full prompt for debugging: {e}")

    # Default system instruction if none provided
    actual_system_instruction = system_instruction or """The AI agent should adopt an academic persona—specifically."""

    # Single attempt, no retries
    logger.debug(f"Starting API call to model {model_name}")
    try:
        # Check rate limit if a rate limiter is provided
        if rate_limiter:
            logger.debug(f"Checking rate limit for {model_name}...")
            await rate_limiter.wait_if_needed_async(model_name)
            logger.debug(
                f"Rate limit check passed/wait finished for {model_name}.")
        else:
            logger.debug("No rate limiter provided.")

        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            # "response_mime_type": "application/json"  # Removed: Not supported by all models
        }
        logger.debug(f"Generation config prepared: {generation_config}")

        # Create model instance, passing system instruction here
        logger.debug(f"Initializing GenerativeModel for {model_name}...")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=actual_system_instruction
        )
        logger.debug(f"GenerativeModel initialized.")

        # Make the API call
        logger.debug(f"Calling generate_content_async...")
        api_call_start_time = asyncio.get_event_loop().time()
        response = await model.generate_content_async(
            contents=[full_prompt_text],
            generation_config=generation_config
        )
        api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
        logger.debug(
            f"generate_content_async call completed in {api_call_duration:.2f} seconds.")

        # Register the successful call
        if rate_limiter:
            logger.debug(f"Registering call with rate limiter...")
            await rate_limiter.register_call_async(model_name)
            logger.debug(f"Call registered.")

        # Extract response text
        logger.debug(f"Extracting text from response...")
        if hasattr(response, 'text'):
            generated_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            text_parts = [
                part.text for part in response.parts if hasattr(part, 'text')]
            if text_parts:
                generated_text = "".join(text_parts)
            else:
                logger.error(f"Response does not contain expected text parts")
                return None
        else:
            logger.error(f"Unexpected response format: {response}")
            return None

        # Log response details for diagnostic purposes
        text_length = len(generated_text)
        logger.debug(f"Response length: {text_length} characters")
        logger.debug(f"Response first 500 chars: {generated_text[:500]}")
        if text_length > 1000:
            logger.debug(f"Response last 500 chars: {generated_text[-500:]}")

        # Save usage metadata if available
        if hasattr(response, 'usage_metadata'):
            usage_metadata = response.usage_metadata
            usage_metadata_filename = f"usage_metadata_{timestamp}.json"
            usage_metadata_path = os.path.join(
                output_dir, usage_metadata_filename)
            try:
                with open(usage_metadata_path, 'w', encoding='utf-8') as f:
                    if hasattr(usage_metadata, '_asdict'):
                        # Convert to dictionary if it's a namedtuple-like object
                        metadata_dict = usage_metadata._asdict()
                        json.dump(metadata_dict, f, indent=2)
                    else:
                        # Try direct serialization
                        json.dump(usage_metadata, f, indent=2)
                logger.info(f"Saved usage metadata to {usage_metadata_path}")
            except (IOError, TypeError) as e:
                logger.warning(f"Failed to save usage metadata: {e}")
                # If direct serialization fails, try a manual approach
                try:
                    metadata_str = str(usage_metadata)
                    with open(usage_metadata_path, 'w', encoding='utf-8') as f:
                        f.write(metadata_str)
                    logger.info(
                        f"Saved usage metadata as string to {usage_metadata_path}")
                except IOError as e2:
                    logger.warning(
                        f"Failed to save usage metadata as string: {e2}")

        # Check if response is empty
        if not generated_text or not generated_text.strip():
            logger.error(f"Empty response from model")
            return None

        # Return the text response
        logger.info(
            f"Successfully received text response of {text_length} characters")
        return generated_text

    except asyncio.TimeoutError:
        api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
        logger.error(
            f"Request to {model_name} timed out after {api_call_duration:.2f} seconds.")
        return None
    except Exception as e:
        logger.error(f"Error calling {model_name}: {e}", exc_info=True)
        return None

    # No retry logic needed anymore - directly return None if we get here
    return None


async def generate_text_from_prompt(
    full_prompt_text: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,  # Default higher for creative generation
    max_output_tokens: int = 8192,  # Default from plan/previous usage
    rate_limiter=None,
    save_debug_info: bool = True,
    debug_info_prefix: str = "essay_prompt"
) -> Optional[str]:
    """
    Generates text from a provided prompt string using the Gemini API.

    This method:
    1. Takes a pre-formatted prompt string.
    2. Calls Gemini API with appropriate system instruction and generation config.
    3. Returns the full text response without any JSON parsing.

    Args:
        full_prompt_text (str): The complete prompt text to send to the model.
        model_name (str): Gemini model to use for generation.
        system_instruction (Optional[str]): Optional system instruction to guide the model.
        temperature (float): Temperature parameter for generation (0.0-1.0).
        max_output_tokens (int): Maximum tokens to generate in the response.
        rate_limiter: Optional rate limiter object to manage API call frequency.
        save_debug_info (bool): Whether to save the prompt and usage metadata for debugging.
        debug_info_prefix (str): Prefix for the debug file names (e.g., "essay_prompt").

    Returns:
        Optional[str]: Full text response if successful, None if failed.
    """
    logger.info(
        f"Generating text from prompt (length: {len(full_prompt_text)} chars) using model {model_name}")

    # Specify API key with environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        return None

    genai.configure(api_key=api_key)

    if not full_prompt_text or len(full_prompt_text.strip()) == 0:
        logger.error("Empty or invalid prompt text provided.")
        return None

    # Save full prompt for debugging if requested
    output_dir = "src/output/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_debug_info:
        os.makedirs(output_dir, exist_ok=True)
        prompt_debug_filename = f"{debug_info_prefix}_{timestamp}.txt"
        prompt_debug_path = os.path.join(output_dir, prompt_debug_filename)
        try:
            with open(prompt_debug_path, 'w', encoding='utf-8') as f:
                f.write(full_prompt_text)
            logger.info(
                f"Saved full prompt for debugging to {prompt_debug_path}")
        except IOError as e:
            logger.warning(f"Failed to save full prompt for debugging: {e}")

    # Use provided system instruction or a default if necessary
    actual_system_instruction = system_instruction or "You are a helpful AI assistant."  # Basic default

    # Single attempt, no retries within this function (retries handled by GeminiClient)
    logger.debug(f"Starting API call to model {model_name}")
    try:
        # Check rate limit if a rate limiter is provided
        if rate_limiter:
            logger.debug(f"Checking rate limit for {model_name}...")
            await rate_limiter.wait_if_needed_async(model_name)
            logger.debug(
                f"Rate limit check passed/wait finished for {model_name}.")
        else:
            logger.debug("No rate limiter provided.")

        # Configure generation parameters
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        logger.debug(f"Generation config prepared: {generation_config}")

        # Create model instance, passing system instruction here
        logger.debug(f"Initializing GenerativeModel for {model_name}...")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=actual_system_instruction
        )
        logger.debug(f"GenerativeModel initialized.")

        # Make the API call
        logger.debug(f"Calling generate_content_async...")
        api_call_start_time = asyncio.get_event_loop().time()
        response = await model.generate_content_async(
            contents=[full_prompt_text],  # Send the prompt directly
            generation_config=generation_config
        )
        api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
        logger.debug(
            f"generate_content_async call completed in {api_call_duration:.2f} seconds.")

        # Register the successful call
        if rate_limiter:
            logger.debug(f"Registering call with rate limiter...")
            await rate_limiter.register_call_async(model_name)
            logger.debug(f"Call registered.")

        # Extract response text
        logger.debug(f"Extracting text from response...")
        if hasattr(response, 'text'):
            generated_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            text_parts = [
                part.text for part in response.parts if hasattr(part, 'text')]
            if text_parts:
                generated_text = "".join(text_parts)
            else:
                logger.error(f"Response does not contain expected text parts")
                return None
        else:
            logger.error(f"Unexpected response format: {response}")
            return None

        # Log response details for diagnostic purposes
        text_length = len(generated_text)
        logger.debug(f"Response length: {text_length} characters")
        logger.debug(f"Response first 500 chars: {generated_text[:500]}")
        if text_length > 1000:
            logger.debug(f"Response last 500 chars: {generated_text[-500:]}")

        # Save usage metadata if available and requested
        if save_debug_info and hasattr(response, 'usage_metadata'):
            usage_metadata = response.usage_metadata
            usage_metadata_filename = f"usage_metadata_{debug_info_prefix}_{timestamp}.json"
            usage_metadata_path = os.path.join(
                output_dir, usage_metadata_filename)
            try:
                with open(usage_metadata_path, 'w', encoding='utf-8') as f:
                    if hasattr(usage_metadata, '_asdict'):
                        metadata_dict = usage_metadata._asdict()
                        json.dump(metadata_dict, f, indent=2)
                    else:
                        json.dump(usage_metadata, f, indent=2)
                logger.info(f"Saved usage metadata to {usage_metadata_path}")
            except (IOError, TypeError) as e:
                logger.warning(f"Failed to save usage metadata: {e}")
                try:  # Fallback to string
                    metadata_str = str(usage_metadata)
                    with open(usage_metadata_path, 'w', encoding='utf-8') as f:
                        f.write(metadata_str)
                    logger.info(
                        f"Saved usage metadata as string to {usage_metadata_path}")
                except IOError as e2:
                    logger.warning(
                        f"Failed to save usage metadata as string: {e2}")

        # Check if response is empty
        if not generated_text or not generated_text.strip():
            logger.error(f"Empty response from model {model_name}")
            return None

        # Return the text response
        logger.info(
            f"Successfully received text response of {text_length} characters from {model_name}")
        return generated_text

    except asyncio.TimeoutError:
        api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
        logger.error(
            f"Request to {model_name} timed out after {api_call_duration:.2f} seconds.")
        return None
    except Exception as e:
        logger.error(f"Error calling {model_name}: {e}", exc_info=True)
        return None

    return None  # Should not be reached unless error occurs
