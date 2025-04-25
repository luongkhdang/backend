"""
utils/json_parser.py - Robust JSON Parsing Utilities

This module provides robust JSON parsing utilities with enhanced error handling,
using json5 for more flexible parsing and fallback mechanisms to standard JSON.

Exported functions:
- parse_json(json_str, default=None, fallback_to_standard=True): Parses JSON with error handling
- safe_loads(json_str, default=None): Always returns a value, even if parsing fails
- get_nested_value(obj, path, default=None): Safely gets nested values from parsed objects
- extract_json_from_text(text): Attempts to extract JSON from text containing other content

Related files:
- src/steps/step3/helpers.py: Uses these utilities for entity extraction
- src/gemini/gemini_client.py: May use these utilities for API response parsing
"""

import json
import json5
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


def parse_json(
    json_str: str,
    default: Any = None,
    fallback_to_standard: bool = True
) -> Tuple[Any, Optional[str]]:
    """
    Parse a JSON string with enhanced error handling.

    This function tries to parse using json5 first (which is more lenient),
    then falls back to standard json if configured to do so.

    Args:
        json_str: The JSON string to parse
        default: The default value to return if parsing fails
        fallback_to_standard: Whether to try standard json if json5 fails

    Returns:
        Tuple[Any, Optional[str]]: (Parsed data or default, error message if parsing failed)
    """
    if not json_str:
        return default, "Empty JSON string"

    # Try json5 first (more lenient, handles comments, trailing commas, etc.)
    try:
        return json5.loads(json_str), None
    except Exception as e:
        json5_error = f"json5 parsing error: {str(e)}"

        # Fall back to standard json if configured
        if fallback_to_standard:
            try:
                return json.loads(json_str), None
            except json.JSONDecodeError as je:
                standard_error = f"standard json parsing error: {str(je)}"
                logger.debug(standard_error)
                return default, f"{json5_error}; {standard_error}"

        return default, json5_error


def safe_loads(json_str: str, default: Any = None) -> Any:
    """
    Parse JSON safely, always returning a value even if parsing fails.

    Args:
        json_str: The JSON string to parse
        default: The default value to return if parsing fails

    Returns:
        Any: The parsed JSON object or the default value
    """
    result, error = parse_json(json_str, default)
    if error:
        logger.warning(f"JSON parsing failed: {error}")
    return result


def get_nested_value(obj: Any, path: str, default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary using a dot-notation path.

    Args:
        obj: The object to extract values from
        path: A dot-notation path like "data.results.0.name"
        default: The default value to return if the path doesn't exist

    Returns:
        Any: The value at the path or the default value
    """
    if not obj:
        return default

    if not path:
        return obj

    keys = path.split(".")
    current = obj

    for key in keys:
        # Handle array indexes in the path (e.g., "items.0.name")
        if key.isdigit() and isinstance(current, list):
            index = int(key)
            if 0 <= index < len(current):
                current = current[index]
            else:
                return default
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def extract_json_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempt to extract a JSON object or array from text that may contain other content.

    This is useful for handling LLM responses that might have JSON embedded in
    explanatory text or markdown.

    Args:
        text: The text that may contain JSON

    Returns:
        Tuple[Optional[str], Optional[str]]: (Extracted JSON string if found, error message if not found)
    """
    if not text:
        return None, "Empty text"

    # 1. Explicitly check for and strip ```json ... ``` markdown blocks
    stripped_text = text.strip()
    if stripped_text.startswith("```json") and stripped_text.endswith("```"):
        json_candidate = stripped_text[len("```json"): -len("```")].strip()
        # Basic validation before returning
        if (json_candidate.startswith("{") and json_candidate.endswith("}")) or \
           (json_candidate.startswith("[") and json_candidate.endswith("]")):
            logger.debug("Extracted JSON by stripping ```json block.")
            return json_candidate, None
    # 2. Explicitly check for and strip ``` ... ``` markdown blocks
    elif stripped_text.startswith("```") and stripped_text.endswith("```"):
        json_candidate = stripped_text[len("```"): -len("```")].strip()
        # Basic validation before returning
        if (json_candidate.startswith("{") and json_candidate.endswith("}")) or \
           (json_candidate.startswith("[") and json_candidate.endswith("]")):
            logger.debug("Extracted JSON by stripping ``` block.")
            return json_candidate, None

    # 3. Fallback: Try the previous regex approach for embedded JSON
    logger.debug(
        "Markdown block stripping failed or not applicable, trying regex fallback.")

    # Try to find JSON object (starting with { and ending with })
    # Use non-greedy matching for nested structures
    # Modified regex to be less likely to grab surrounding text
    object_match = re.search(
        r'{\s*(?:\"[^"]*\"|[^:{\[\],])*?\s*:(?:.|\n)*?}', text)
    # Try to find JSON array (starting with [ and ending with ])
    array_match = re.search(r'\[(?:.|\n)*?\]', text)

    # Prioritize the longer match if both are found, assuming it's more likely the full JSON
    json_candidate = None
    if object_match and array_match:
        if object_match.start() <= array_match.start() and object_match.end() >= array_match.end():
            json_candidate = object_match.group(0)
        elif array_match.start() <= object_match.start() and array_match.end() >= object_match.end():
            json_candidate = array_match.group(0)
        # If they don't overlap cleanly, prefer the object match as it's more common for LLM JSON output
        else:
            json_candidate = object_match.group(0)
    elif object_match:
        json_candidate = object_match.group(0)
    elif array_match:
        json_candidate = array_match.group(0)

    if json_candidate:
        # Attempt to parse the candidate to verify it's valid JSON
        result, error = parse_json(json_candidate)
        if not error:
            logger.debug(
                f"Extracted JSON using regex fallback: {json_candidate[:50]}...")
            return json_candidate, None
        else:
            logger.debug(
                f"Regex candidate failed parsing: {error}. Candidate: {json_candidate[:50]}...")

    # Final Fallback: Search specifically within ``` blocks again if primary stripping failed
    # This handles cases where the stripping logic was too simple (e.g., extra whitespace)
    code_blocks = re.findall(r'```(?:json)?\\s*([\\s\\S]*?)```', text)
    for block in code_blocks:
        block = block.strip()  # Strip whitespace from the extracted block
        result, error = parse_json(block)
        if not error:
            logger.debug(
                f"Extracted JSON using final code block regex fallback: {block[:50]}...")
            return block, None

    return None, "No valid JSON found in text"
