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
        logger.debug(json5_error)

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

    # Try to find JSON object (starting with { and ending with })
    object_match = re.search(r'({[\s\S]*?})', text)
    array_match = re.search(r'(\[[\s\S]*?\])', text)

    # If we found a potential JSON object
    if object_match:
        json_candidate = object_match.group(1)
        result, error = parse_json(json_candidate)
        if not error:
            return json_candidate, None

    # If we found a potential JSON array
    if array_match:
        json_candidate = array_match.group(1)
        result, error = parse_json(json_candidate)
        if not error:
            return json_candidate, None

    # Try with code block extraction (for markdown-formatted text)
    code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
    for block in code_blocks:
        result, error = parse_json(block)
        if not error:
            return block, None

    # Look for the largest matching braces that might contain valid JSON
    # This is a more aggressive approach for malformed text
    stack = []
    start_indices = []
    potential_jsons = []

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_indices.append(i)
            stack.append('{')
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
            if not stack:
                start_idx = start_indices.pop()
                potential_jsons.append(text[start_idx:i+1])

    # Try each potential JSON string
    for json_str in potential_jsons:
        result, error = parse_json(json_str)
        if not error:
            return json_str, None

    return None, "No valid JSON found in text"
