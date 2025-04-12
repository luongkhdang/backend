"""
content_processor.py - Article content processing module for Data Refinery Pipeline

This module provides functions to process raw article content:
- process_article_content: Cleans article content by removing noise and standardizing text
- validate_and_prepare_for_storage: Validates processed content and prepares article data for storage

Related files:
- src/main.py: Uses these functions in the data pipeline
- src/database/news_api_client.py: Provides the raw article data
- src/database/reader_db_client.py: Handles storage of processed articles
"""
import re
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load spaCy model with error handling
try:
    import spacy
    nlp = spacy.load("en_core_web_lg")
    SPACY_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.critical(
        f"spaCy or model not available: {e}. Sentence boundary detection will be limited.")
    SPACY_AVAILABLE = False


def process_article_content(content: str) -> str:
    """
    Process raw article content by removing noise and standardizing text.

    Processing steps:
    1. HTML Artifacts Removal: Strip tags and convert entities to plain text
    2. Whitespace Normalization: Collapse multiple spaces, tabs, or newlines
    3. Case Normalization: Convert to lowercase
    4. Punctuation/Special Character Cleaning: Preserve sentence structure, remove clutter
    5. Boilerplate Removal: Filter out common advertisement and subscription texts
    6. Sentence Boundary Detection: Ensure proper sentence splits

    Args:
        content (str): Raw article content

    Returns:
        str: Processed and cleaned content
    """
    if not content:
        logger.warning("Received empty content for processing")
        return ""

    try:
        # 1. HTML Artifacts Removal
        try:
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.warning(f"Error in HTML parsing: {e}, using raw content")
            text = content

        # 2. Initial Whitespace Normalization
        text = re.sub(r'\s+', ' ', text).strip()

        # 3. Case Normalization (to lowercase)
        text = text.lower()

        # 4. Punctuation/Special Character Cleaning
        # Keep essential punctuation (.,!?-) but remove other non-alphanumeric characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Collapse repeating punctuation
        text = re.sub(r'([.,!?-])\1+', r'\1', text)

        # 5. Boilerplate Removal (Heuristic)
        # Common phrases that indicate boilerplate content
        boilerplate_patterns = [
            r'\bclick here\b', r'\bsubscribe now\b', r'\badvertisement\b',
            r'\bprivacy policy\b', r'\bterms of service\b', r'\bcookie policy\b',
            r'\bsubscribe to our newsletter\b', r'\bsign up for our newsletter\b',
            r'\ball rights reserved\b', r'\bcopyrights?\b'
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text)

        # 6. Sentence Boundary Detection using spaCy if available
        if SPACY_AVAILABLE and text:
            try:
                # Process with spaCy for better sentence boundaries
                doc = nlp(text)
                text = " ".join([sent.text for sent in doc.sents])
            except Exception as e:
                logger.warning(f"Error in spaCy processing: {e}")

        # Final Whitespace Normalization
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    except Exception as e:
        logger.error(f"Error processing content: {e}")
        return ""  # Return empty string on error


def validate_and_prepare_for_storage(article: Dict[str, Any], processed_content: str) -> Dict[str, Any]:
    """
    Validate processed content and prepare article data for storage.

    Validation criteria:
    - Content must have at least 80 words after processing

    Args:
        article (Dict[str, Any]): Original article data with at least id, title, pub_date, domain
        processed_content (str): Processed article content

    Returns:
        Dict[str, Any]: Dictionary with all fields needed for database insertion
            including scraper_id, title, pub_date, domain, content, processed_at
    """
    # Validate the processed content (must have at least 80 words)
    is_valid = len(processed_content.split()) >= 80

    # Prepare data for storage
    if is_valid:
        content_to_store = processed_content
        processed_at = datetime.now(timezone.utc)
        logger.info(
            f"Article {article.get('id')} processed successfully: {len(processed_content.split())} words")
    else:
        content_to_store = 'ERROR'
        processed_at = None
        logger.warning(
            f"Article {article.get('id')} failed validation: {len(processed_content.split())} words (< 80 minimum)")

    # Return a dictionary with all necessary fields for database insertion
    return {
        'scraper_id': article.get('id'),
        'title': article.get('title', f"Article {article.get('id')}"),
        'pub_date': article.get('pub_date'),
        'domain': article.get('domain', ''),
        'content': content_to_store,
        'processed_at': processed_at
    }
