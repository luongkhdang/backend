"""
schema.py - Database schema management

This module provides functions for initializing and managing the database schema,
including tables, indexes, and extensions.

Exported functions:
- initialize_tables(conn) -> None
  Creates tables if they don't exist
- ensure_column_exists(conn, table: str, column: str, column_def: str) -> bool
  Ensures a column exists in a table, adds it if not

Related modules:
- Used by ReaderDBClient for database schema management
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def initialize_tables(conn) -> bool:
    """
    Initialize database tables if they don't exist.

    Args:
        conn: Database connection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Create articles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            scraper_id INTEGER UNIQUE,
            title TEXT,
            url TEXT,
            content TEXT,
            pub_date TIMESTAMP,
            domain TEXT,
            author TEXT,
            is_valid BOOLEAN DEFAULT TRUE,
            is_processed BOOLEAN DEFAULT FALSE,
            is_hot BOOLEAN DEFAULT FALSE,
            cluster_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error TEXT
        );
        """)

        # Create entities table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE,
            entity_type TEXT,
            description TEXT,
            influence_score FLOAT DEFAULT 0.0,
            mentions INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create article_entities junction table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS article_entities (
            article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            mention_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (article_id, entity_id)
        );
        """)

        # Create embeddings table with pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
            embedding VECTOR(768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(article_id)
        );
        """)

        # Create clusters table with pgvector and additional metadata field
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id SERIAL PRIMARY KEY,
            centroid VECTOR(768),
            is_hot BOOLEAN DEFAULT FALSE,
            article_count INTEGER,
            metadata JSONB,
            created_at DATE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create essays table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS essays (
            id SERIAL PRIMARY KEY,
            title TEXT,
            content TEXT,
            summary TEXT,
            cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create essay_entities junction table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS essay_entities (
            essay_id INTEGER REFERENCES essays(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (essay_id, entity_id)
        );
        """)

        # Ensure required columns exist
        ensure_column_exists(conn, "entities", "entity_type", "TEXT")
        ensure_column_exists(
            conn, "entities", "influence_score", "FLOAT DEFAULT 0.0")
        ensure_column_exists(conn, "entities", "mentions", "INTEGER DEFAULT 0")
        ensure_column_exists(conn, "articles", "is_hot",
                             "BOOLEAN DEFAULT FALSE")
        ensure_column_exists(conn, "clusters", "metadata", "JSONB")

        # Create indexes to improve query performance
        create_indexes(conn)

        conn.commit()
        cursor.close()

        logger.info("Database tables initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing database tables: {e}")
        conn.rollback()
        return False


def ensure_column_exists(conn, table: str, column: str, column_def: str) -> bool:
    """
    Ensure a column exists in a table, adding it if it doesn't.

    Args:
        conn: Database connection
        table: Table name
        column: Column name
        column_def: Column definition (e.g. "TEXT", "INTEGER DEFAULT 0")

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.columns 
            WHERE table_name = '{table}' AND column_name = '{column}'
        )
        """)
        column_exists = cursor.fetchone()[0]

        if not column_exists:
            # Add column if it doesn't exist
            cursor.execute(f"""
            ALTER TABLE {table} 
            ADD COLUMN IF NOT EXISTS {column} {column_def};
            """)
            conn.commit()
            logger.info(f"Added column '{column}' to table '{table}'")

        cursor.close()
        return True

    except Exception as e:
        logger.warning(
            f"Error ensuring column {column} exists in {table}: {e}")
        return False


def create_indexes(conn) -> bool:
    """
    Create indexes on tables to improve query performance.

    Args:
        conn: Database connection

    Returns:
        bool: True if successful, False otherwise
    """
    indexes = [
        ("idx_articles_scraper_id", "articles", "scraper_id"),
        ("idx_articles_cluster_id", "articles", "cluster_id"),
        ("idx_articles_domain", "articles", "domain"),
        ("idx_articles_pub_date", "articles", "pub_date"),
        ("idx_entities_name", "entities", "name"),
        ("idx_entities_type", "entities", "entity_type"),
        ("idx_article_entities_article_id", "article_entities", "article_id"),
        ("idx_article_entities_entity_id", "article_entities", "entity_id"),
        ("idx_embeddings_article_id", "embeddings", "article_id"),
    ]

    success = True

    for idx_name, table, column in indexes:
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column});
            """)
            conn.commit()
            cursor.close()
        except Exception as e:
            logger.warning(f"Error creating index {idx_name}: {e}")
            success = False

    return success
