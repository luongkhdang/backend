"""
steps/step3/helpers.py - Entity Extraction Response Helpers

This module provides helper functions for parsing and processing entity extraction
responses from the Gemini API in the Step 3 entity extraction process.

Exported functions:
- parse_entity_response(response_text): Parses API response with error handling
- extract_entities_from_parsed_response(parsed_data): Extracts structured entity data
- validate_entity_data(entity_data): Validates entity data contains required fields
- format_entity_for_storage(entity): Standardizes entity format for database storage

Related files:
- src/steps/step3/__init__.py: Main entity extraction module that uses these helpers
- src/utils/json_parser.py: Provides core JSON parsing functionality
- src/gemini/gemini_client.py: Generates API responses that need parsing
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import re

from src.utils.json_parser import (
    parse_json,
    extract_json_from_text,
    get_nested_value
)

# Configure logging
logger = logging.getLogger(__name__)

# Define entity data structure requirements
REQUIRED_ENTITY_FIELDS = ['name', 'type', 'relevance_score']
VALID_ENTITY_TYPES = [
    'PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT',
    'WORK_OF_ART', 'CONSUMER_GOOD', 'OTHER'
]


def parse_entity_response(response_text: str) -> Tuple[Any, Optional[str]]:
    """
    Parse the entity extraction API response with comprehensive error handling.

    This function handles various response formats including:
    - Clean JSON responses
    - JSON embedded in explanatory text
    - Malformed JSON that needs extraction/repair

    Args:
        response_text: The raw text response from the entity extraction API

    Returns:
        Tuple[Any, Optional[str]]: (Parsed data object or None, error message if parsing failed)
    """
    if not response_text:
        return None, "Empty response from API"

    # First, try to parse directly as JSON
    parsed_data, error = parse_json(response_text)

    # If direct parsing fails, try to extract JSON from the text
    if error:
        logger.debug(f"Direct JSON parsing failed: {error}")

        # Try to extract JSON from the response text
        extracted_json, extract_error = extract_json_from_text(response_text)
        if extract_error:
            logger.warning(
                f"Failed to extract JSON from response: {extract_error}")
            return None, f"JSON extraction failed: {extract_error}"

        # Parse the extracted JSON
        parsed_data, error = parse_json(extracted_json)
        if error:
            logger.warning(f"Failed to parse extracted JSON: {error}")
            return None, f"Extracted JSON parsing failed: {error}"

    # Check if we actually have data after successful parsing
    if parsed_data is None:
        return None, "Parsed response is None"

    return parsed_data, None


def extract_entities_from_parsed_response(parsed_data: Any) -> Tuple[List[Dict], Optional[str]]:
    """
    Extract structured entity data from the parsed API response.

    This function handles different response structures that the API might return:
    - Direct entity list at the root
    - Entities nested under common paths like 'entities', 'data.entities', etc.

    Args:
        parsed_data: The parsed response data object

    Returns:
        Tuple[List[Dict], Optional[str]]: (List of entity objects, error message if extraction failed)
    """
    if parsed_data is None:
        return [], "No data to extract entities from"

    # Common paths where entities might be found in the response
    common_entity_paths = [
        "",  # Root of the response
        "entities",
        "data.entities",
        "result.entities",
        "results.entities",
        "analysis.entities",
        "document.entities",
        "response.entities"
    ]

    # Try each path to find entities
    for path in common_entity_paths:
        potential_entities = get_nested_value(parsed_data, path, [])

        # Skip empty results
        if not potential_entities:
            continue

        # If we found a list of entities
        if isinstance(potential_entities, list):
            # Ensure all items look like entity objects (have the minimum required fields)
            valid_entities = []
            for entity in potential_entities:
                if isinstance(entity, dict) and 'name' in entity:
                    # Standardize the entity format
                    standardized_entity = {
                        'name': entity.get('name', ''),
                        'type': entity.get('type', entity.get('entity_type', 'OTHER')),
                        'relevance_score': float(entity.get('relevance_score',
                                                            entity.get('score',
                                                                       entity.get('confidence', 0.5)))),
                        # Include other fields that might be present
                        'mentions': entity.get('mentions', []),
                        'metadata': entity.get('metadata', entity.get('properties', {}))
                    }
                    valid_entities.append(standardized_entity)

            if valid_entities:
                return valid_entities, None

    # Handle the case where entities are found but not in a standard format
    if isinstance(parsed_data, dict) and len(parsed_data) > 0:
        # Try to interpret the response as a dictionary of entity_name -> entity_data
        artificial_entities = []
        for key, value in parsed_data.items():
            if isinstance(value, dict):
                entity = {
                    'name': key,
                    'type': value.get('type', 'OTHER'),
                    'relevance_score': float(value.get('score', 0.5)),
                    'metadata': value
                }
                artificial_entities.append(entity)

        if artificial_entities:
            return artificial_entities, None

    return [], "Could not find valid entities in the response"


def validate_entity_data(entity_data: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Validate that the entity data contains all required fields and valid values.

    Args:
        entity_data: List of entity objects to validate

    Returns:
        Tuple[List[Dict], List[str]]: (List of valid entities, List of validation error messages)
    """
    if not entity_data:
        return [], ["No entities to validate"]

    valid_entities = []
    validation_errors = []

    for i, entity in enumerate(entity_data):
        entity_errors = []

        # Check for required fields
        for field in REQUIRED_ENTITY_FIELDS:
            if field not in entity:
                entity_errors.append(f"Missing required field '{field}'")

        # Validate entity type if present
        if 'type' in entity and entity['type'] not in VALID_ENTITY_TYPES:
            # Try to normalize the entity type to a standard value
            normalized_type = _normalize_entity_type(entity['type'])
            if normalized_type in VALID_ENTITY_TYPES:
                entity['type'] = normalized_type
            else:
                entity_errors.append(
                    f"Invalid entity type: '{entity['type']}', expected one of {VALID_ENTITY_TYPES}"
                )

        # Validate relevance score if present
        if 'relevance_score' in entity:
            try:
                score = float(entity['relevance_score'])
                if not (0 <= score <= 1):
                    entity_errors.append(
                        f"Invalid relevance score: {score}, expected value between 0 and 1"
                    )
                else:
                    # Ensure the score is stored as a float
                    entity['relevance_score'] = score
            except (ValueError, TypeError):
                entity_errors.append(
                    f"Invalid relevance score format: '{entity['relevance_score']}', expected float"
                )

        # Add validation messages to the error list
        if entity_errors:
            entity_id = entity.get('name', f'Entity #{i}')
            for error in entity_errors:
                validation_errors.append(f"{entity_id}: {error}")
        else:
            valid_entities.append(entity)

    return valid_entities, validation_errors


def format_entity_for_storage(entity: Dict) -> Dict:
    """
    Standardize entity format for database storage.

    Args:
        entity: Single entity object

    Returns:
        Dict: Standardized entity object ready for storage
    """
    # Create a standardized entity object with required and optional fields
    standardized_entity = {
        # Required fields
        'name': entity.get('name', ''),
        'type': entity.get('type', 'OTHER'),
        'relevance_score': float(entity.get('relevance_score', 0.5)),

        # Optional fields with defaults
        'mentions': entity.get('mentions', []),
        'metadata': entity.get('metadata', {}),

        # Additional fields that may be useful
        'aliases': entity.get('aliases', []),
        'description': entity.get('description', ''),
        'source': entity.get('source', 'gemini_api')
    }

    return standardized_entity


def _normalize_entity_type(entity_type: str) -> str:
    """
    Normalize entity type strings to match the expected valid types.

    This is an internal helper function to standardize entity types
    that may be returned in different formats by the API.

    Args:
        entity_type: Original entity type string

    Returns:
        str: Normalized entity type that matches one of the valid types
    """
    # Convert to uppercase and remove spaces
    normalized = entity_type.upper().replace(' ', '_')

    # Handle common variations
    type_mapping = {
        'PERSON': ['PERSON', 'PEOPLE', 'HUMAN', 'INDIVIDUAL'],
        'ORGANIZATION': ['ORGANIZATION', 'ORG', 'COMPANY', 'CORPORATION', 'INSTITUTION'],
        'LOCATION': ['LOCATION', 'PLACE', 'ADDRESS', 'GEOGRAPHICAL_ENTITY', 'CITY', 'COUNTRY'],
        'EVENT': ['EVENT', 'HAPPENING', 'OCCURRENCE'],
        'WORK_OF_ART': ['WORK_OF_ART', 'ARTWORK', 'CREATIVE_WORK', 'BOOK', 'MOVIE', 'SONG'],
        'CONSUMER_GOOD': ['CONSUMER_GOOD', 'PRODUCT', 'GOODS', 'MERCHANDISE'],
        'OTHER': ['OTHER', 'MISCELLANEOUS', 'UNKNOWN', 'UNDEFINED']
    }

    # Check if normalized type matches any of the valid type variations
    for valid_type, variations in type_mapping.items():
        if normalized in variations:
            return valid_type

    # If no match found, return the normalized string
    return normalized
