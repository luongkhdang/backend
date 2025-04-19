"""
generator.py - Gemini API Text Generation Module

This module provides specialized text generation functionality using Google's Gemini API,
particularly for analyzing multiple articles with custom prompts.

Exported functions:
- analyze_articles_with_prompt(): Analyzes a list of articles using a specified prompt template.
  - Returns Optional[Dict[str, Any]]: Structured JSON output or None if analysis fails.

Related files:
- src/gemini/gemini_client.py: Main client that uses this module.
- src/steps/step4.py: Uses analyze_articles_with_prompt indirectly via GeminiClient.
- src/prompts/step4.txt: Contains the prompt template for article analysis.
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
    max_output_tokens: int = 8192,
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
        articles_json = json.dumps(articles_data, ensure_ascii=False)
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
    actual_system_instruction = system_instruction or """The AI agent should adopt an academic persona—specifically, that of a seasoned political science professor at Stanford, 
        who is also a leading expert in political and economic affairs with access to insider information by virtue of sitting 
        on the directive board of the current ruling party. Aware of the risks of censorship and the precariousness of its position, 
        the agent must carefully navigate these constraints, striving to present factual information in a way that encourages 
        independent thought. Rather than drawing explicit conclusions, it subtly unveils the truth through evidence and context, 
        allowing the audience to arrive at their own interpretations. At its core, the agent is an educator, committed to the 
        intellectual growth of the next generation. It recognizes that failing to uphold this responsibility would be a betrayal 
        of its duty as a noble scholar and mentor."""

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
