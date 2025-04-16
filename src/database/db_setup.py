#!/usr/bin/env python3

import logging
import os
import time
from src.database.reader_db_client import ReaderDBClient
from psycopg2 import ProgrammingError  # Import ProgrammingError

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Set up the database and test the connection."""
    client = None  # Initialize client to None
    try:
        logger.info("Initializing Reader DB client...")

        # Initialize the client with environment variables or defaults
        client = ReaderDBClient(
            host=os.getenv("READER_DB_HOST", "localhost"),
            port=int(os.getenv("READER_DB_PORT", "5432")),
            dbname=os.getenv("READER_DB_NAME", "reader_db"),
            user=os.getenv("READER_DB_USER", "postgres"),
            password=os.getenv("READER_DB_PASSWORD", "postgres"),
            max_retries=5,
            retry_delay=10
        )

        # Initialize tables (idempotent)
        logger.info("Initializing database tables (if necessary)...")
        if client.initialize_tables():
            logger.info("Tables initialized or already exist.")
        else:
            logger.error("Failed to initialize tables!")
            # Decide if you want to exit here or continue

        # Test the connection
        logger.info("Testing database connection...")
        connection_info = client.test_connection()

        # Log connection details
        logger.info(
            f"Connected to database version: {connection_info.get('version', 'Unknown')}")
        logger.info(
            f"Database name: {connection_info.get('database', 'Unknown')}")
        logger.info(
            f"Available tables: {list(connection_info.get('tables', {}).keys())}")
        logger.info(
            f"Record counts: {connection_info.get('record_counts', {})}")

        logger.info("Database setup and check completed successfully")

    except Exception as e:
        logger.error(f"Error during database setup: {e}", exc_info=True)
    finally:
        # Close the connection if client was initialized
        if client:
            client.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    # Add a delay to wait for the PostgreSQL container to be ready
    # This is useful in Docker Compose environments
    if os.getenv("WAIT_FOR_DB", "false").lower() == "true":
        wait_seconds = int(os.getenv("WAIT_SECONDS", "5"))
        logger.info(
            f"Waiting {wait_seconds} seconds for database to be ready...")
        time.sleep(wait_seconds)

    main()
