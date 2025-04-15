#!/usr/bin/env python3

import logging
import os
import time
from src.database.reader_db_client import ReaderDBClient
from psycopg2 import ProgrammingError  # Import ProgrammingError

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_cluster_created_at(client: ReaderDBClient):
    """Migrate the clusters.created_at column from TIMESTAMP to DATE (Pacific Time)."""
    conn = None
    try:
        conn = client.get_connection()
        cursor = conn.cursor()

        # Check current data type of created_at
        cursor.execute("""
            SELECT data_type 
            FROM information_schema.columns
            WHERE table_name = 'clusters' AND column_name = 'created_at';
        """)
        result = cursor.fetchone()
        current_type = result[0] if result else None

        if current_type and 'timestamp' in current_type.lower():
            logger.info(
                "Migrating clusters.created_at from TIMESTAMP to DATE (Pacific Time)...")

            # 1. Add a temporary DATE column
            logger.info("Step 1: Adding temporary date column...")
            cursor.execute(
                "ALTER TABLE clusters ADD COLUMN IF NOT EXISTS created_at_date_temp DATE;")
            conn.commit()

            # 2. Populate the temporary column with converted dates (Pacific Time)
            logger.info(
                "Step 2: Populating temporary column with converted Pacific dates...")
            cursor.execute("""
                UPDATE clusters 
                SET created_at_date_temp = DATE(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')
                WHERE created_at IS NOT NULL;
            """)
            conn.commit()
            logger.info(f"Updated {cursor.rowcount} rows.")

            # 3. Drop the old timestamp column
            logger.info("Step 3: Dropping old timestamp column...")
            cursor.execute("ALTER TABLE clusters DROP COLUMN created_at;")
            conn.commit()

            # 4. Rename the temporary column to the final name
            logger.info("Step 4: Renaming temporary column...")
            cursor.execute(
                "ALTER TABLE clusters RENAME COLUMN created_at_date_temp TO created_at;")
            conn.commit()

            logger.info(
                "Migration of clusters.created_at completed successfully.")
        elif current_type == 'date':
            logger.info(
                "clusters.created_at is already DATE type. Migration not needed.")
        else:
            logger.warning(
                f"Could not determine type or unexpected type '{current_type}' for clusters.created_at. Skipping migration.")

        cursor.close()

    except ProgrammingError as pe:
        # Handle specific errors like column already exists if run multiple times partially
        logger.warning(
            f"Programming error during migration (might be expected if run partially before): {pe}")
        if conn:
            conn.rollback()
    except Exception as e:
        logger.error(f"Error during clusters.created_at migration: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            client.release_connection(conn)


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

        # Run Migrations
        logger.info("Running database migrations (if necessary)...")
        migrate_cluster_created_at(client)  # Run the new migration
        logger.info("Migrations complete.")

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
