"""
domains.py - Domain-related database operations

This module provides functions for managing domain data, including
retrieving domain goodness scores and domain statistics.

Exported functions:
- get_all_domain_goodness_scores(conn) -> Dict[str, float]
  Gets all domain goodness scores from the database
- get_domain_goodness_scores(conn, domains: List[str]) -> Dict[str, float]
  Gets goodness scores for specific domains by name

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
            SELECT domain, goodness_score 
            FROM domain_statistics
        """)

        # Convert result to a dictionary
        domain_scores = {row[0]: row[1]
                         for row in cursor.fetchall() if row[1] is not None}
        cursor.close()

        return domain_scores
    except Exception as e:
        logger.error(f"Error retrieving domain goodness scores: {e}")
        return {}


def get_domain_goodness_scores(conn, domains: List[str]) -> Dict[str, float]:
    """
    Get goodness scores for specific domains by name.

    Args:
        conn: Database connection
        domains: List of domain names to fetch scores for

    Returns:
        Dict[str, float]: Dictionary mapping domains to their goodness scores (defaults to 0.5 if not found)
    """
    if not domains:
        return {}

    try:
        cursor = conn.cursor()

        # Create a placeholder string for the IN clause
        placeholders = ', '.join(['%s'] * len(domains))

        # Query goodness scores for the provided domains
        cursor.execute(f"""
            SELECT domain, goodness_score
            FROM domain_statistics
            WHERE domain IN ({placeholders})
        """, domains)

        # Build the result dictionary
        goodness_scores = {}
        for row in cursor.fetchall():
            domain, score = row
            # Use 0.5 as a reasonable default if score is None
            goodness_scores[domain] = float(
                score) if score is not None else 0.5

        cursor.close()

        # Add entries with 0.5 (neutral) for any domains not found in DB
        for domain in domains:
            if domain not in goodness_scores and domain is not None:
                goodness_scores[domain] = 0.5

        return goodness_scores

    except Exception as e:
        logger.error(f"Error fetching domain goodness scores: {e}")
        # Return dict with 0.5 (neutral) for all requested domains on error
        return {domain: 0.5 for domain in domains if domain is not None}
