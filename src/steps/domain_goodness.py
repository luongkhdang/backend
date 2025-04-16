#!/usr/bin/env python3
"""
domain_goodness.py - Domain Goodness Score Calculation Module

This module implements the full domain goodness calculation logic, analyzing domain statistics 
and reliability based on article clustering and hotness metrics.

Exported functions:
- calculate_domain_goodness_scores(db_client, min_entries, hot_threshold_percent, ...): Calculates goodness scores
  - Returns Dict[str, Any]: Statistics about the calculation process

Related files:
- src/database/reader_db_client.py: Used to interact with the database
- src/database/modules/domains.py: Contains module-level domain operations
- src/main.py: May call this module for scheduled domain goodness calculations
"""

import logging
import math
from typing import Dict, Any, List, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_domain_goodness_scores(db_client, min_entries: int = 5,
                                     hot_threshold_percent: float = 15.0,
                                     bonus_value: float = 0.1,
                                     domain_volume_weight: float = 0.6,
                                     domain_hotness_weight: float = 0.4) -> Dict[str, Any]:
    """
    Calculate goodness scores for all domains based on volume and hotness.

    Args:
        db_client: ReaderDBClient instance
        min_entries: Minimum entries required for scoring (unless 100% hot)
        hot_threshold_percent: Percentage threshold for consistency bonus
        bonus_value: Value of consistency bonus
        domain_volume_weight: Weight for normalized log volume component
        domain_hotness_weight: Weight for hotness percentage component

    Returns:
        Dict with calculation statistics
    """
    start_time = time.time()
    conn = None  # Initialize conn
    cursor = None  # Initialize cursor

    try:
        # Get database connection
        conn = db_client.get_connection()
        cursor = conn.cursor()

        # Step 1 & 2: Calculate domain statistics and goodness scores directly
        logger.info("Calculating domain statistics and goodness scores...")

        # Use a CTE to calculate domain statistics directly
        cursor.execute(f"""
            WITH domain_stats AS (
                SELECT
                    a.domain,
                    COUNT(*) as total_entries,
                    SUM(CASE WHEN c.hotness_score >= 0.7 THEN 1 ELSE 0 END) as hot_entries,
                    AVG(COALESCE(c.hotness_score, 0)) as average_cluster_hotness
                FROM articles a
                LEFT JOIN clusters c ON a.cluster_id = c.id
                WHERE a.domain IS NOT NULL AND a.domain != ''
                GROUP BY a.domain
            )
            SELECT domain, total_entries, hot_entries, average_cluster_hotness
            FROM domain_stats
            ORDER BY total_entries DESC;
        """)

        domains_data = cursor.fetchall()

        # Find max entries for normalization before the loop
        max_entries = max((row[1] for row in domains_data), default=1)
        logger.info(
            f"Found {len(domains_data)} domains with max {max_entries} entries")

        # Process each domain
        # Store full statistics for insertion later
        processed_domain_data = []
        domains_processed = 0
        domains_skipped = 0

        for domain, total_entries, hot_entries, avg_hotness in domains_data:
            # Skip domains with too few entries (unless all are hot)
            # Ensure hot_entries is not None before comparison
            hot_entries = hot_entries or 0
            if total_entries < min_entries and (hot_entries < total_entries):
                domains_skipped += 1
                continue

            # Calculate hot percentage
            hot_percentage = (
                hot_entries / total_entries) if total_entries > 0 else 0

            # Calculate log volume component (normalized)
            log_volume = math.log10(max(total_entries, 1))
            max_log_volume = math.log10(
                max_entries) if max_entries > 0 else 1  # Avoid log10(0)
            normalized_volume = log_volume / max_log_volume if max_log_volume > 0 else 0

            # Calculate base goodness score using weighted formula
            base_score = (
                domain_volume_weight * normalized_volume +
                domain_hotness_weight * hot_percentage
            )

            # Apply consistency bonus if hot percentage exceeds threshold
            consistency_bonus = bonus_value if hot_percentage >= (
                hot_threshold_percent / 100.0) else 0

            # Calculate final score (capped at 1.0)
            final_score = min(base_score + consistency_bonus, 1.0)

            # Append all calculated data for the domain
            # Ensure avg_hotness is not None
            processed_domain_data.append(
                (domain, total_entries, hot_entries, avg_hotness or 0.0, final_score))
            domains_processed += 1

        # Step 3: Store results in domain_statistics table
        logger.info(
            f"Storing calculated scores for {len(processed_domain_data)} domains...")

        # Clear existing scores from the correct table
        # Use TRUNCATE for speed, ensure permissions allow it
        cursor.execute("TRUNCATE TABLE domain_statistics;")

        # Insert new scores using correct columns
        insert_count = 0
        # Use the correct columns for domain_statistics
        insert_query = """
            INSERT INTO domain_statistics 
            (domain, total_entries, hot_entries, average_cluster_hotness, goodness_score, calculated_at)
            VALUES (%s, %s, %s, %s, %s, DEFAULT) 
        """

        for domain_data in processed_domain_data:
            # domain_data is (domain, total_entries, hot_entries, avg_hotness, final_score)
            cursor.execute(insert_query, domain_data)
            insert_count += 1

            # Commit in batches to avoid memory issues with very large datasets
            if insert_count % 100 == 0:
                conn.commit()
                logger.debug(
                    f"Committed batch of {insert_count} domain statistics...")

        # Final commit
        conn.commit()
        logger.info(f"Final commit of {insert_count} domain statistics.")

        runtime = time.time() - start_time

        # Get top domains using the updated _get_top_domains function
        top_domains_list = _get_top_domains(cursor, 5)  # Pass the cursor

        # Log summary statistics
        summary = {
            "domains_processed": domains_processed,
            "domains_skipped": domains_skipped,
            "scores_stored": insert_count,
            "runtime_seconds": runtime,
            "top_domains": top_domains_list
        }

        cursor.close()
        db_client.release_connection(conn)
        conn = None  # Mark as closed

        logger.info(
            f"Domain goodness calculation complete. Processed {domains_processed} domains in {runtime:.2f} seconds")
        return summary

    except Exception as e:
        logger.error(
            f"Error calculating domain goodness scores: {e}", exc_info=True)
        if cursor:
            cursor.close()
        if conn:
            db_client.release_connection(conn)

        return {
            "error": str(e),
            "domains_processed": 0,
            "runtime_seconds": time.time() - start_time
        }


def _get_top_domains(cursor, limit: int = 5) -> List[Dict[str, Any]]:
    """Get top domains by goodness score for reporting."""
    try:
        # Use correct table (domain_statistics) and column name (goodness_score)
        cursor.execute("""
            SELECT domain, goodness_score
            FROM domain_statistics
            ORDER BY goodness_score DESC
            LIMIT %s
        """, (limit,))

        # Return domain and score
        return [
            {"domain": row[0], "score": row[1]}
            for row in cursor.fetchall()
        ]
    except Exception as e:
        logger.error(f"Error fetching top domains: {e}", exc_info=True)
        return []


if __name__ == "__main__":
    # This allows running the module directly for testing or scheduled updates
    from src.database.reader_db_client import ReaderDBClient

    logger.info("Starting domain goodness calculation from command line")
    db_client = ReaderDBClient()

    try:
        result = calculate_domain_goodness_scores(db_client)
        logger.info(f"Calculation complete: {result}")

        if result.get("top_domains"):
            logger.info("Top domains by goodness score:")
            for domain in result["top_domains"]:
                logger.info(
                    f"  {domain['domain']}: {domain['score']:.4f}")
    finally:
        db_client.close()
