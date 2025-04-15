"""
utils.py - Utility functions for article clustering

This module provides utility functions used by the step2 clustering process.

Exported functions/variables:
- initialize_nlp() -> Any: Initializes and returns a spaCy NLP model
- SPACY_AVAILABLE: Boolean indicating whether spaCy is available

Related files:
- src/steps/step2/core.py: Uses these utilities for clustering
- src/steps/step2/interpretation.py: Uses the NLP model for text processing
"""

import logging
import importlib.util
from typing import Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Check if spaCy is available for cluster interpretation
SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None


def initialize_nlp(model_name: str = "en_core_web_sm") -> Optional[Any]:
    """
    Initialize and return a spaCy NLP model.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        Loaded spaCy model or None if initialization fails
    """
    if not SPACY_AVAILABLE:
        logger.warning(
            "spaCy is not available. NLP functionalities will be limited.")
        return None

    try:
        import spacy
        nlp = spacy.load(model_name)
        logger.info(f"Successfully loaded spaCy model: {model_name}")
        return nlp
    except Exception as e:
        logger.error(f"Failed to load spaCy model '{model_name}': {e}")
        return None
