#!/usr/bin/env python3
"""
migrate_embeddings.py - One-time script to copy embeddings.

This script copies existing embeddings from the `embeddings` table 
to the `embedding` column in the `articles` table.

This is intended to be run once after adding the `articles.embedding` 
column to backfill data for existing articles.

Usage:
  (From the project root)
  python src/database/scripts/migrate_embeddings.py

  (Inside Docker container, if applicable)
  python /app/src/database/scripts/migrate_embeddings.py
"""

from psycopg2.extras import execute_batch  # For efficient batch updates
from src.database.reader_db_client import ReaderDBClient
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 500  # Process embeddings in batches


def migrate_embeddings():
    """Fetches embeddings and updates the articles table."""
    db_client = None
    conn = None
    cursor = None
    processed_count = 0
    failed_count = 0
    total_embeddings = 0

    try:
        logger.info("Initializing Reader DB client...")
        db_client = ReaderDBClient()
        conn = db_client.get_connection()
        if not conn:
            logger.critical("Failed to get database connection. Aborting.")
            return False

        cursor = conn.cursor()

        # Count total embeddings first for progress reporting
        cursor.execute(
            "SELECT COUNT(*) FROM embeddings WHERE embedding IS NOT NULL;")
        total_embeddings = cursor.fetchone()[0]
        logger.info(
            f"Found {total_embeddings} non-NULL embeddings to migrate.")

        if total_embeddings == 0:
            logger.info("No embeddings found to migrate.")
            return True

        # Fetch embeddings in batches
        cursor.execute(
            "SELECT article_id, embedding FROM embeddings WHERE embedding IS NOT NULL;")

        while True:
            batch_data = cursor.fetchmany(BATCH_SIZE)
            if not batch_data:
                break

            update_tuples = []
            valid_in_batch = 0
            for article_id, embedding_vector in batch_data:
                if article_id is not None and embedding_vector is not None:
                    update_tuples.append((embedding_vector, article_id))
                    valid_in_batch += 1
                else:
                    logger.warning(
                        f"Skipping record with NULL article_id or embedding: article_id={article_id}")
                    failed_count += 1  # Count as failure if data is invalid

            if not update_tuples:
                logger.info(
                    f"No valid data in current batch, skipping update.")
                continue

            # Perform batch update
            update_cursor = conn.cursor()  # Use a separate cursor for the update batch
            try:
                update_query = "UPDATE articles SET embedding = %s WHERE id = %s;"
                execute_batch(update_cursor, update_query, update_tuples)
                conn.commit()
                processed_count += len(update_tuples)
                logger.info(
                    f"Migrated batch of {len(update_tuples)}. Total processed: {processed_count}/{total_embeddings}")
            except Exception as batch_update_err:
                logger.error(
                    f"Error updating batch: {batch_update_err}", exc_info=True)
                conn.rollback()  # Rollback this batch on error
                failed_count += len(update_tuples)
                logger.warning(
                    f"Failed to update batch of {len(update_tuples)} articles.")
            finally:
                if update_cursor:
                    update_cursor.close()

        logger.info("--- Migration Summary ---")
        logger.info(f"Total Embeddings Found: {total_embeddings}")
        logger.info(f"Successfully Migrated:  {processed_count}")
        logger.info(f"Failed/Skipped:       {failed_count}")
        return failed_count == 0  # Return True if no failures

    except Exception as e:
        logger.critical(f"Migration script failed: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            db_client.release_connection(conn)
        if db_client:
            # db_client.close() # Don't close pool if other processes might use it
            pass
        logger.info("Migration script finished.")


if __name__ == "__main__":
    if migrate_embeddings():
        logger.info("Embedding migration completed successfully.")
        sys.exit(0)
    else:
        logger.error("Embedding migration finished with errors.")
        sys.exit(1)
