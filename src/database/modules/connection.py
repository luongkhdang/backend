"""
connection.py - Database connection management

This module provides functions for creating and managing database connections
through a connection pool.

Exported functions:
- initialize_connection_pool(db_config: Dict[str, Any]) -> psycopg2.pool.ThreadedConnectionPool
  Creates and returns a threaded connection pool
- get_connection(pool) -> Connection
  Gets a connection from the pool
- release_connection(pool, conn) -> None
  Returns a connection to the pool
- close_pool(pool) -> None
  Closes all connections in the pool

Related modules:
- Used by ReaderDBClient for database connection management
"""

import logging
import time
from typing import Dict, Any, Optional

try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extensions import connection as Connection
    from pgvector.psycopg2 import register_vector
    PSYCOPG2_AVAILABLE = True
except ImportError:
    logging.warning(
        "psycopg2 or pgvector not available; database functionality will be limited")
    psycopg2 = None
    pool = None
    Connection = None
    register_vector = None
    PSYCOPG2_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Register the pgvector adapter globally if psycopg2 is available
if PSYCOPG2_AVAILABLE and register_vector:
    try:
        register_vector()
        logger.info(
            "Successfully registered pgvector adapter for psycopg2 globally.")
    except Exception as e:
        logger.error(f"Failed to register pgvector adapter globally: {e}")


def initialize_connection_pool(db_config: Dict[str, Any]) -> Optional[Any]:
    """
    Initialize a connection pool for PostgreSQL connections.

    Args:
        db_config: Dictionary containing database connection parameters
                  (host, port, dbname, user, password)

    Returns:
        ThreadedConnectionPool or None if initialization failed
    """
    if not PSYCOPG2_AVAILABLE:
        logger.error(
            "Cannot initialize connection pool: psycopg2 not available")
        return None

    max_retries = db_config.get('max_retries', 5)
    retry_delay = db_config.get('retry_delay', 10)
    min_conn = 1
    max_conn = 10
    connection_pool = None

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Initializing DB connection pool (attempt {attempt+1}/{max_retries})")
            # Create a pool of threaded connections
            connection_pool = pool.ThreadedConnectionPool(
                minconn=min_conn,
                maxconn=max_conn,
                host=db_config['host'],
                port=db_config['port'],
                dbname=db_config['dbname'],
                user=db_config['user'],
                password=db_config['password']
            )
            logger.info("Database connection pool initialized successfully")
            return connection_pool
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    "Maximum retry attempts reached; Failed to connect to database")
                return None


def get_connection(connection_pool) -> Optional[Connection]:
    """
    Get a connection from the pool.
    Also registers the pgvector type adapter for the connection.

    Args:
        connection_pool: The ThreadedConnectionPool to get a connection from

    Returns:
        Connection object or None if getting connection failed
    """
    if connection_pool is None:
        logger.error("Cannot get connection: connection pool is None")
        return None

    try:
        conn = connection_pool.getconn()
        if conn and register_vector:  # Ensure register_vector was imported
            try:
                # Register vector type for this specific connection
                register_vector(conn)
                logger.debug(
                    "pgvector adapter registered for retrieved connection.")
            except Exception as e:
                # Log error but still return the connection if obtained
                logger.error(
                    f"Failed to register pgvector adapter for connection: {e}")
        return conn
    except Exception as e:
        logger.error(f"Error getting connection from pool: {e}")
        return None


def release_connection(connection_pool, conn) -> None:
    """
    Release a connection back to the pool.

    Args:
        connection_pool: The ThreadedConnectionPool to return the connection to
        conn: The connection to release
    """
    if connection_pool is None or conn is None:
        return

    try:
        connection_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Error returning connection to pool: {e}")


def close_pool(connection_pool) -> None:
    """
    Close all connections in the pool.

    Args:
        connection_pool: The ThreadedConnectionPool to close
    """
    if connection_pool is None:
        return

    try:
        connection_pool.closeall()
        logger.info("Database connection pool closed")
    except Exception as e:
        logger.error(f"Error closing connection pool: {e}")
