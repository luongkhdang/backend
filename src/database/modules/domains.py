"""
domains.py - Domain-related database operations

This module provides functions for managing domain data, including
retrieving domain goodness scores and domain statistics.

Exported functions:
- get_all_domain_goodness_scores(conn) -> Dict[str, float]
  Gets all domain goodness scores from the database

Related files:
- src/database/reader_db_client.py: Uses this module for domain operations
- src/steps/step3/__init__.py: Uses domain goodness scores for article prioritization
"""

import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)


def get_all_domain_goodness_scores(conn) -> Dict[str, float]:
    """
    Get all domain goodness scores from the database.

    Args:
        conn: Database connection

    Returns:
        Dict[str, float]: Dictionary mapping domains to their goodness scores
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT domain, domain_goodness_score
            FROM calculated_domain_goodness
        """)

        # Convert result to a dictionary
        domain_scores = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()

        return domain_scores
    except Exception as e:
        logger.error(f"Error retrieving domain goodness scores: {e}")
        return {}
