"""
schema.py - Database schema management

This module provides functions for initializing and managing the database schema,
including tables, indexes, and extensions.

The articles table includes a frame_phrases TEXT[] column which stores narrative framing phrases
extracted during entity extraction (Step 3). These represent the dominant narrative frames
present in each article.

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

        # Ensure pgvector extension is enabled
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create articles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id SERIAL PRIMARY KEY,
            scraper_id INTEGER UNIQUE,
            title TEXT,
            content TEXT,
            pub_date TIMESTAMP,
            domain TEXT,
            processed_at TIMESTAMP,
            extracted_entities BOOLEAN DEFAULT FALSE,
            is_hot BOOLEAN DEFAULT FALSE,
            cluster_id INTEGER,
            frame_phrases TEXT[] NULL
            -- Note: Foreign key to clusters cannot be added here due to potential cyclic dependency
            -- or if clusters table might not exist yet. It could be added separately if needed.
        );
        """)

        # Create entities table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name TEXT UNIQUE,
            entity_type TEXT,
            influence_score FLOAT DEFAULT 0.0,
            mentions INTEGER DEFAULT 0,
            first_seen TIMESTAMP,
            last_seen TIMESTAMP,
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

        # Add is_influential_context column to article_entities if it doesn't exist
        cursor.execute("""
        ALTER TABLE article_entities ADD COLUMN IF NOT EXISTS is_influential_context BOOLEAN DEFAULT FALSE;
        """)

        # Add frame_phrases column to articles table if it doesn't exist
        cursor.execute("""
        ALTER TABLE articles ADD COLUMN IF NOT EXISTS frame_phrases TEXT[] NULL;
        """)

        # Create embeddings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            article_id INTEGER UNIQUE REFERENCES articles(id) ON DELETE CASCADE,
            embedding VECTOR(768),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create clusters table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id SERIAL PRIMARY KEY,
            centroid VECTOR(768),
            is_hot BOOLEAN DEFAULT FALSE,
            article_count INTEGER,
            created_at DATE DEFAULT CURRENT_DATE, -- Set default to current date
            metadata JSONB,
            hotness_score FLOAT DEFAULT NULL
        );
        """)

        # Create essays table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS essays (
            id SERIAL PRIMARY KEY,
            type TEXT,
            article_id INTEGER, -- Not a foreign key based on schema? Consider implications.
            title TEXT,
            content TEXT,
            layer_depth INTEGER,
            cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags TEXT[]
        );
        """)

        # Create essay_entities junction table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS essay_entities (
            essay_id INTEGER REFERENCES essays(id) ON DELETE CASCADE,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (essay_id, entity_id)
        );
        """)

        # Create domain_statistics table (replacing calculated_domain_goodness)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS domain_statistics (
            domain TEXT PRIMARY KEY,
            total_entries INTEGER DEFAULT 0,
            hot_entries INTEGER DEFAULT 0,
            average_cluster_hotness FLOAT DEFAULT 0.0,
            goodness_score FLOAT DEFAULT 0.0,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_domain_statistics_score ON domain_statistics(goodness_score DESC);
        """)

        # Add influence_calculated_at column to entities table
        cursor.execute("""
        ALTER TABLE entities 
        ADD COLUMN IF NOT EXISTS influence_calculated_at TIMESTAMP;
        """)

        # Create entity influence factors table for transparency
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_influence_factors (
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            base_mention_score FLOAT,
            source_quality_score FLOAT,
            content_context_score FLOAT,
            temporal_score FLOAT,
            raw_data JSONB,  -- Store raw inputs for auditability
            PRIMARY KEY (entity_id, calculation_timestamp)
        );
        """)

        # Create entity type weights table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_type_weights (
            entity_type TEXT PRIMARY KEY,
            weight FLOAT NOT NULL DEFAULT 1.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Initialize entity type weights with default values
        cursor.execute("""
        INSERT INTO entity_type_weights (entity_type, weight) VALUES
        ('PERSON', 1.0),
        ('ORGANIZATION', 1.1),
        ('GOVERNMENT_AGENCY', 1.2),
        ('LOCATION', 0.8),
        ('GEOPOLITICAL_ENTITY', 1.3),
        ('CONCEPT', 1.0),
        ('LAW_OR_POLICY', 1.1),
        ('EVENT', 0.9),
        ('OTHER', 0.7)
        ON CONFLICT (entity_type) DO NOTHING;
        """)

        # Create entity snippets table for storing supporting text
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_snippets (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
            snippet TEXT NOT NULL,
            is_influential BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Create entity_snippets indexes
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_entity_snippets_entity_id ON entity_snippets(entity_id);
        CREATE INDEX IF NOT EXISTS idx_entity_snippets_article_id ON entity_snippets(article_id);
        """)

        # Create clusters_fundamental table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters_fundamental (
            id SERIAL PRIMARY KEY,
            centroid VECTOR(768), -- Assuming same dimension as other vectors
            metadata JSONB
        );
        """)

        # Create indexes to improve query performance
        create_indexes(conn)

        conn.commit()
        cursor.close()

        logger.info(
            "Database tables initialized successfully according to provided schema")
        return True

    except Exception as e:
        # Add exc_info for more details
        logger.error(f"Error initializing database tables: {e}", exc_info=True)
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
