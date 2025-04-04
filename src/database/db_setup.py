#!/usr/bin/env python3

import logging
import os
import time
from src.database.reader_db_client import ReaderDBClient

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Set up the database and test the connection."""
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

        # Test the connection
        logger.info("Testing database connection...")
        connection_info = client.test_connection()

        # Log connection details
        logger.info(
            f"Connected to database version: {connection_info.get('version', 'Unknown')}")
        logger.info(
            f"Database name: {connection_info.get('database', 'Unknown')}")
        logger.info(f"Database user: {connection_info.get('user', 'Unknown')}")
        logger.info(
            f"Available tables: {', '.join(connection_info.get('tables', []))}")

        # Close the connection
        client.close()

        logger.info("Database setup completed successfully")

    except Exception as e:
        logger.error(f"Error during database setup: {e}")


if __name__ == "__main__":
    # Add a delay to wait for the PostgreSQL container to be ready
    # This is useful in Docker Compose environments
    if os.getenv("WAIT_FOR_DB", "false").lower() == "true":
        wait_seconds = int(os.getenv("WAIT_SECONDS", "5"))
        logger.info(
            f"Waiting {wait_seconds} seconds for database to be ready...")
        time.sleep(wait_seconds)

    main()
