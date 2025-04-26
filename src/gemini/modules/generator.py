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
    client: google_genai.Client,
    articles_data: List[Dict[str, Any]],
    prompt_file_path: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 65536,
    rate_limiter=None
) -> Optional[str]:
    """
    Analyzes articles using google-genai client. Called by GeminiClient.
    Implements retry logic: Tries Thinking+Grounding -> Thinking -> Base.
    Args:
        client: Initialized google_genai.Client instance.
        articles_data (List[Dict[str, Any]]): List of prepared article dictionaries.
        prompt_file_path (str): Path to the prompt template file.
        model_name (str): Gemini model ID to use for analysis.
        system_instruction (Optional[str]): Optional system instruction.
        temperature (float): Temperature for generation.
        max_output_tokens (int): Maximum output tokens.
        rate_limiter: Optional rate limiter object.
    Returns:
        Optional[str]: Full text response or None.
    """
    logger.info(
        f"Analyzing {len(articles_data)} snippets with prompt from {prompt_file_path} using {model_name}")

    # Load prompt template (remains the same)
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

    # Prepare prompt content (remains the same)
    try:
        articles_json = json.dumps(
            articles_data, separators=(',', ':'), ensure_ascii=True)
        full_prompt_text = prompt_template.replace(
            "{INPUT_DATA_JSON}", articles_json)
    except (TypeError, OverflowError) as e:
        logger.error(f"Failed to serialize article data to JSON: {e}")
        return None

    logger.debug(f"Full prompt size: {len(full_prompt_text)} characters")
    # (Optional debug saving remains the same)
    # ... save prompt logic ...

    actual_system_instruction = system_instruction or """The AI agent should adopt an academic persona—specifically."""

    # --- Retry Logic --- #
    attempts_config = [
        # Attempt 1: Thinking + Grounding
        {"thinking": True, "grounding": True, "name": "Thinking+Grounding"},
        # Attempt 2: Thinking only
        {"thinking": True, "grounding": False, "name": "Thinking"},
        # Attempt 3: Base (No Thinking, No Grounding)
        {"thinking": False, "grounding": False, "name": "Base"}
    ]

    last_error = None

    for i, attempt_config in enumerate(attempts_config):
        attempt_num = i + 1
        use_thinking = attempt_config["thinking"]
        use_grounding = attempt_config["grounding"]
        attempt_name = attempt_config["name"]

        logger.info(
            f"[{model_name}] Attempt {attempt_num}/3 ({attempt_name}) for prompt analysis.")

        try:
            # Rate Limit Check (before each attempt)
            if rate_limiter:
                await rate_limiter.wait_if_needed_async(model_name)

            # --- Dynamic Configuration for this attempt --- #
            gen_config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "automatic_function_calling": {'disable': True}
            }
            tools_list = None

            if use_thinking:
                try:
                    gen_config_kwargs["thinking_config"] = google_genai_types.ThinkingConfig(
                        include_thoughts=True)
                    logger.debug(
                        f"[{model_name}] Attempt {attempt_num}: Enabling Thinking.")
                except AttributeError:
                    logger.warning(
                        f"[{model_name}] Attempt {attempt_num}: Cannot enable Thinking (AttributeError). Skipping for this attempt.")
                    # If ThinkingConfig fails, we can't proceed with this attempt configuration
                    if i < len(attempts_config) - 1:  # If not the last attempt
                        logger.warning(
                            f"[{model_name}] Skipping attempt {attempt_num} due to Thinking config error.")
                        last_error = AttributeError("ThinkingConfig not found")
                        continue  # Move to next attempt configuration
                    else:
                        raise  # Reraise if it's the last attempt

            if use_grounding:
                try:
                    tools_list = [google_genai_types.Tool(
                        google_search_retrieval=google_genai_types.GoogleSearchRetrieval()
                    )]
                    logger.debug(
                        f"[{model_name}] Attempt {attempt_num}: Enabling Grounding.")
                except AttributeError:
                    logger.warning(
                        f"[{model_name}] Attempt {attempt_num}: Cannot enable Grounding (AttributeError). Skipping Grounding for this attempt.")
                    tools_list = None  # Ensure tools are None if grounding fails

            gen_config = google_genai_types.GenerateContentConfig(
                **gen_config_kwargs)
            # --- End Dynamic Configuration --- #

            # --- API Call --- #
            api_call_start_time = asyncio.get_event_loop().time()
            response = await client.aio.models.generate_content(
                model=f'models/{model_name}',
                contents=[google_genai_types.Content(
                    parts=[google_genai_types.Part(text=full_prompt_text)], role="user")],
                config=gen_config,
                tools=tools_list
            )
            api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
            logger.debug(
                f"[{model_name}] Attempt {attempt_num}: API call succeeded in {api_call_duration:.2f}s.")

            if rate_limiter:  # Register successful call
                await rate_limiter.register_call_async(model_name)
            # --- End API Call --- #

            # --- Process Response --- #
            generated_text = ""
            thoughts_log = []
            try:
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought is True:
                            if part.text:
                                thoughts_log.append(part.text.strip())
                        elif hasattr(part, 'text'):
                            generated_text += part.text
                if thoughts_log:
                    logger.debug(
                        f"[{model_name}] Attempt {attempt_num} Thoughts: {' | '.join(thoughts_log)}")
            except Exception as proc_e:
                logger.error(
                    f"[{model_name}] Attempt {attempt_num}: Error processing response parts: {proc_e}", exc_info=True)
                generated_text = getattr(response, 'text', "")  # Fallback

            if generated_text and generated_text.strip():
                logger.info(
                    f"[{model_name}] Attempt {attempt_num} ({attempt_name}) successful. Returning response.")
                return generated_text  # SUCCESS
            else:
                finish_reason = getattr(response, 'prompt_feedback', getattr(
                    response, 'finish_reason', None))
                logger.warning(
                    f"[{model_name}] Attempt {attempt_num} ({attempt_name}) resulted in empty text. Finish Reason: {finish_reason}. Retrying if possible.")
                last_error = ValueError(
                    f"Empty response text (Finish Reason: {finish_reason})")
                # Continue to next attempt in the loop

        except google.api_core.exceptions.GoogleAPIError as e:
            logger.warning(
                f"[{model_name}] Attempt {attempt_num} ({attempt_name}) failed with GoogleAPIError: {e}. Retrying if possible.", exc_info=False)
            last_error = e
            # Continue to next attempt
        except asyncio.TimeoutError as e:
            logger.warning(
                f"[{model_name}] Attempt {attempt_num} ({attempt_name}) timed out. Retrying if possible.")
            last_error = e
            # Continue to next attempt
        except Exception as e:
            logger.error(
                f"[{model_name}] Attempt {attempt_num} ({attempt_name}) failed with unexpected error: {e}. Retrying if possible.", exc_info=True)
            last_error = e
            # Continue to next attempt

    # If loop finishes without returning, all attempts failed
    logger.error(
        f"[{model_name}] All {len(attempts_config)} attempts failed for prompt analysis. Last error: {last_error}")
    return None


async def generate_text_from_prompt(
    client: google_genai.Client,
    full_prompt_text: str,
    model_name: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 8192,
    rate_limiter=None,
    save_debug_info: bool = True,
    debug_info_prefix: str = "essay_prompt"
) -> Optional[str]:
    """
    Generates text using google-genai client. Called by GeminiClient.
    Implements retry logic: Tries Thinking+Grounding -> Thinking -> Base.
    Args:
        client: Initialized google_genai.Client instance.
        full_prompt_text (str): The complete prompt text.
        model_name (str): Gemini model ID to use.
        system_instruction (Optional[str]): Optional system instruction.
        temperature (float): Temperature for generation.
        max_output_tokens (int): Maximum output tokens.
        rate_limiter: Optional rate limiter object.
        save_debug_info (bool): Whether to save debug info.
        debug_info_prefix (str): Prefix for debug file names.
    Returns:
        Optional[str]: Full text response or None.
    """
    logger.info(
        f"Generating text (prompt len: {len(full_prompt_text)}) using {model_name}")

    if not full_prompt_text or len(full_prompt_text.strip()) == 0:
        logger.error("Empty or invalid prompt text provided.")
        return None

    # (Optional debug saving remains the same)
    # ... save prompt logic ...

    actual_system_instruction = system_instruction or "You are a helpful AI assistant."

    # --- Retry Logic --- #
    attempts_config = [
        {"thinking": True, "grounding": True, "name": "Thinking+Grounding"},
        {"thinking": True, "grounding": False, "name": "Thinking"},
        {"thinking": False, "grounding": False, "name": "Base"}
    ]

    last_error = None

    for i, attempt_config in enumerate(attempts_config):
        attempt_num = i + 1
        use_thinking = attempt_config["thinking"]
        use_grounding = attempt_config["grounding"]
        attempt_name = attempt_config["name"]

        logger.info(
            f"[{model_name}] Attempt {attempt_num}/3 ({attempt_name}) for text generation.")

        try:
            # Rate Limit Check
            if rate_limiter:
                await rate_limiter.wait_if_needed_async(model_name)

            # --- Dynamic Configuration for this attempt --- #
            gen_config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "automatic_function_calling": {'disable': True}
            }
            tools_list = None

            if use_thinking:
                try:
                    gen_config_kwargs["thinking_config"] = google_genai_types.ThinkingConfig(
                        include_thoughts=True)
                    logger.debug(
                        f"[{model_name}] Attempt {attempt_num}: Enabling Thinking.")
                except AttributeError:
                    logger.warning(
                        f"[{model_name}] Attempt {attempt_num}: Cannot enable Thinking (AttributeError). Skipping for this attempt.")
                    if i < len(attempts_config) - 1:
                        logger.warning(
                            f"[{model_name}] Skipping attempt {attempt_num} due to Thinking config error.")
                        last_error = AttributeError("ThinkingConfig not found")
                        continue
                    else:
                        raise

            if use_grounding:
                try:
                    tools_list = [google_genai_types.Tool(
                        google_search_retrieval=google_genai_types.GoogleSearchRetrieval()
                    )]
                    logger.debug(
                        f"[{model_name}] Attempt {attempt_num}: Enabling Grounding.")
                except AttributeError:
                    logger.warning(
                        f"[{model_name}] Attempt {attempt_num}: Cannot enable Grounding (AttributeError). Skipping Grounding for this attempt.")
                    tools_list = None

            gen_config = google_genai_types.GenerateContentConfig(
                **gen_config_kwargs)
            # --- End Dynamic Configuration --- #

            # --- API Call --- #
            api_call_start_time = asyncio.get_event_loop().time()
            contents = [google_genai_types.Content(
                parts=[google_genai_types.Part(text=full_prompt_text)])]
            # Note: System instructions are handled differently. Add if needed based on SDK/API specifics.
            response = await client.aio.models.generate_content(
                model=f'models/{model_name}',
                contents=contents,
                config=gen_config,
                tools=tools_list
            )
            api_call_duration = asyncio.get_event_loop().time() - api_call_start_time
            logger.debug(
                f"[{model_name}] Attempt {attempt_num}: API call succeeded in {api_call_duration:.2f}s.")

            if rate_limiter:  # Register successful call
                await rate_limiter.register_call_async(model_name)
            # --- End API Call --- #

            # --- Process Response --- #
            generated_text = ""
            thoughts_log = []
            try:
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought is True:
                            if part.text:
                                thoughts_log.append(part.text.strip())
                        elif hasattr(part, 'text'):
                            generated_text += part.text
                if thoughts_log:
                    logger.debug(
                        f"[{model_name}] Attempt {attempt_num} Thoughts: {' | '.join(thoughts_log)}")
            except Exception as proc_e:
                logger.error(
                    f"[{model_name}] Attempt {attempt_num}: Error processing response parts: {proc_e}", exc_info=True)
                generated_text = getattr(response, 'text', "")  # Fallback

            if generated_text and generated_text.strip():
                logger.info(
                    f"[{model_name}] Attempt {attempt_num} ({attempt_name}) successful. Returning response.")
                return generated_text  # SUCCESS
            else:
                finish_reason = getattr(response, 'prompt_feedback', getattr(
                    response, 'finish_reason', None))
                logger.warning(
                    f"[{model_name}] Attempt {attempt_num} ({attempt_name}) resulted in empty text. Finish Reason: {finish_reason}. Retrying if possible.")
                last_error = ValueError(
                    f"Empty response text (Finish Reason: {finish_reason})")
                # Continue to next attempt

        except google.api_core.exceptions.GoogleAPIError as e:
            logger.warning(
                f"[{model_name}] Attempt {attempt_num} ({attempt_name}) failed with GoogleAPIError: {e}. Retrying if possible.", exc_info=False)
            last_error = e
            # Continue to next attempt
        except asyncio.TimeoutError as e:
            logger.warning(
                f"[{model_name}] Attempt {attempt_num} ({attempt_name}) timed out. Retrying if possible.")
            last_error = e
            # Continue to next attempt
        except Exception as e:
            logger.error(
                f"[{model_name}] Attempt {attempt_num} ({attempt_name}) failed with unexpected error: {e}. Retrying if possible.", exc_info=True)
            last_error = e
            # Continue to next attempt

    # If loop finishes without returning, all attempts failed
    logger.error(
        f"[{model_name}] All {len(attempts_config)} attempts failed for text generation. Last error: {last_error}")
    return None
