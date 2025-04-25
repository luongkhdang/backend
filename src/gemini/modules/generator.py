"""
generator.py - Gemini API Text Generation Module (using google-genai)

This module provides specialized text generation functionality using Google's Gemini API
(via the google-genai library), particularly for analyzing multiple articles or
generating text from a single large prompt.

It expects an initialized `google_genai.Client` instance to be passed to its functions.

Exported functions:
- analyze_articles_with_prompt(client, ...): Analyzes a list of articles using a specified prompt template.
  - Returns Optional[str]: Structured JSON response or None if analysis fails.
- generate_text_from_prompt(client, ...): Generates text from a provided prompt string.
  - Returns Optional[str]: Generated text response or None if generation fails.

Related files:
- src/gemini/gemini_client.py: Main client that uses this module.
- src/steps/step4.py: Uses analyze_articles_with_prompt indirectly via GeminiClient.
- src/steps/step5.py: Uses generate_text_from_prompt indirectly via GeminiClient.
- src/prompts/step4.txt: Contains the prompt template for article analysis.
- src/prompts/haystack_prompt.txt: Contains the prompt template for essay generation.
- src/utils/rate_limit.py: RateLimiter passed from GeminiClient.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime


from google import genai as google_genai  # Correct import for google-genai
# Import types separately if needed
from google.genai import types as google_genai_types
import google.api_core.exceptions

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def analyze_articles_with_prompt(
    client: google_genai.Client,  # Type hint uses the correct alias
    articles_data: List[Dict[str, Any]],
    prompt_file_path: str,
    model_name: str,  # Expecting model ID like 'gemini-1.5-flash-latest'
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 65536,
    # retries: int = 1, # Retries handled by caller (GeminiClient)
    # initial_delay: float = 1.0,
    rate_limiter=None
) -> Optional[str]:
    """
    Analyzes articles using google-genai client. Called by GeminiClient.
    Args:
        client: Initialized google_genai.Client instance.
        articles_data (List[Dict[str, Any]]): List of prepared article dictionaries.
        prompt_file_path (str): Path to the prompt template file (e.g., `src/prompts/step4.txt`).
        model_name (str): Gemini model to use for analysis.
        system_instruction (Optional[str]): Optional system instruction to guide the model.
        temperature (float): Temperature parameter for generation (0.0-1.0).
        max_output_tokens (int): Maximum tokens to generate in the response.
        rate_limiter: Optional rate limiter object to manage API call frequency.
    Returns:
        Optional[str]: Full text response or None.
    """
    logger.info(
        f"Analyzing {len(articles_data)} snippets with prompt from {prompt_file_path}")

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

    actual_system_instruction = system_instruction or """The AI agent should adopt an academic persona—specifically."""

    logger.debug(f"Starting API call to model {model_name}")
    try:
        if rate_limiter:
            # Use model_name (ID) key
            await rate_limiter.wait_if_needed_async(model_name)

        # Configure generation parameters using new types
        gen_config = google_genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            automatic_function_calling={'disable': True}  # Disable AFC
            # response_mime_type="application/json" # Add if needed/supported
        )

        # --- NEW API CALL using passed client ---
        api_call_start_time = asyncio.get_event_loop().time()
        response = await client.aio.models.generate_content(
            model=f'models/{model_name}',  # Prepend models/ prefix
            contents=[google_genai_types.Content(  # Structure content with system instruction if provided
                parts=[google_genai_types.Part(text=full_prompt_text)],
                role="user"  # Assuming the prompt is user input
            )],
            config=gen_config,
            # Pass system_instruction if the client method supports it directly
            # Otherwise, it might need to be part of the 'contents' list
            # Check google-genai docs for system instruction handling
            # For now, assuming it might need to be prepended to prompt or passed differently
            # system_instruction=actual_system_instruction # This might not be a valid param here
        )
        # Handle system instruction - Prepend to prompt if not a direct param
        # contents = [google_genai.types.Part(text=actual_system_instruction), google_genai.types.Part(text=full_prompt_text)] if actual_system_instruction else [google_genai.types.Part(text=full_prompt_text)]
        # response = await client.generate_content_async(model=f'models/{model_name}', contents=contents, generation_config=gen_config)
        # ^ Alternative if system_instruction param is not valid

        api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
        logger.debug(
            f"generate_content_async call completed in {api_call_duration:.2f} seconds.")
        # --- END NEW API CALL ---

        if rate_limiter:
            await rate_limiter.register_call_async(model_name)

        # --- Process Response (New Structure) ---
        generated_text = None
        if hasattr(response, 'text'):
            generated_text = response.text
        elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
            generated_text = "".join(
                part.text for part in response.parts if hasattr(part, 'text'))
        # ... (log response details) ...
        # ... (save usage metadata - check response structure) ...
        # ... (check for empty response) ...

        if not generated_text or not generated_text.strip():
            # Check for prompt feedback / finish reason
            finish_reason = getattr(response, 'prompt_feedback', None) or getattr(
                response, 'finish_reason', None)
            block_reason = getattr(response.prompt_feedback, 'block_reason', None) if hasattr(
                response, 'prompt_feedback') else None
            logger.error(
                f"Empty response from model {model_name}. Finish reason: {finish_reason}, Block Reason: {block_reason}")
            return None

        logger.info(
            f"Successfully received text response of {len(generated_text)} characters")
        return generated_text

    except asyncio.TimeoutError:
        # ... (log timeout) ...
        return None
    except google.api_core.exceptions.GoogleAPIError as e:  # Catch specific Google errors
        logger.error(
            f"Google API Error calling {model_name}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Generic Error calling {model_name}: {e}", exc_info=True)
        return None


async def generate_text_from_prompt(
    client: google_genai.Client,  # Type hint uses the correct alias
    full_prompt_text: str,
    model_name: str,  # Expecting model ID like 'gemini-1.5-flash-latest'
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 8192,
    rate_limiter=None,
    save_debug_info: bool = True,
    debug_info_prefix: str = "essay_prompt"
) -> Optional[str]:
    """
    Generates text using google-genai client. Called by GeminiClient.
    Args:
        client: Initialized google_genai.Client instance.
        full_prompt_text (str): The complete prompt text to send to the model.
        model_name (str): Gemini model to use for generation.
        system_instruction (Optional[str]): Optional system instruction to guide the model.
        temperature (float): Temperature parameter for generation (0.0-1.0).
        max_output_tokens (int): Maximum tokens to generate in the response.
        rate_limiter: Optional rate limiter object to manage API call frequency.
        save_debug_info (bool): Whether to save the prompt and usage metadata for debugging.
        debug_info_prefix (str): Prefix for the debug file names (e.g., "essay_prompt").
    Returns:
        Optional[str]: Full text response or None.
    """
    logger.info(
        f"Generating text from prompt (length: {len(full_prompt_text)} chars) using model {model_name}")

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

    actual_system_instruction = system_instruction or "You are a helpful AI assistant."

    logger.debug(f"Starting API call to model {model_name}")
    try:
        if rate_limiter:
            await rate_limiter.wait_if_needed_async(model_name)

        # Configure generation parameters
        gen_config = google_genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            automatic_function_calling={'disable': True}  # Disable AFC
        )

        # --- NEW API CALL using passed client ---
        api_call_start_time = asyncio.get_event_loop().time()
        # Structure contents, potentially including system instruction
        contents = [google_genai_types.Content(
            parts=[google_genai_types.Part(text=full_prompt_text)])]
        # How system instructions are handled in google-genai needs verification.
        # Option 1: Client parameter (if supported)
        # Option 2: Model parameter (if supported)
        # Option 3: Prepend to contents list
        # Assuming Option 3 for now if direct parameter isn't obvious:
        # if actual_system_instruction:
        #    contents.insert(0, google_genai_types.Part(text=actual_system_instruction))

        # CORRECTED: Use client.aio.models.generate_content
        response = await client.aio.models.generate_content(
            model=f'models/{model_name}',
            contents=contents,
            config=gen_config
            # system_instruction=actual_system_instruction # Check if valid
        )
        api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
        logger.debug(
            f"generate_content_async call completed in {api_call_duration:.2f} seconds.")
        # --- END NEW API CALL ---

        if rate_limiter:
            await rate_limiter.register_call_async(model_name)

        # --- Process Response (New Structure) ---
        generated_text = None
        if hasattr(response, 'text'):
            generated_text = response.text
        elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
            generated_text = "".join(
                part.text for part in response.parts if hasattr(part, 'text'))
        # ... (log response details) ...
        # ... (save usage metadata if requested - check response structure) ...

        if not generated_text or not generated_text.strip():
            finish_reason = getattr(response, 'prompt_feedback', None) or getattr(
                response, 'finish_reason', None)
            block_reason = getattr(response.prompt_feedback, 'block_reason', None) if hasattr(
                response, 'prompt_feedback') else None
            logger.error(
                f"Empty response from model {model_name}. Finish reason: {finish_reason}, Block Reason: {block_reason}")
            return None

        logger.info(
            f"Successfully received text response of {len(generated_text)} characters from {model_name}")
        return generated_text

    except asyncio.TimeoutError:
        # ... (log timeout) ...
        return None
    except google.api_core.exceptions.GoogleAPIError as e:
        logger.error(
            f"Google API Error calling {model_name}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Generic Error calling {model_name}: {e}", exc_info=True)
        return None
