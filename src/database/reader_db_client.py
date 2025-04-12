import psycopg2
from psycopg2 import pool
import logging
import os
import time
import random
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# For date parsing
try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReaderDBClient:
    """
    Client for interactions with the reader-db database.

    This client provides a direct connection to the reader-db PostgreSQL database
    and handles connection pooling, retries, and query execution.

    Exported functions:
    - test_connection(): Tests database connection, returns connection details
    - get_connection(): Gets a connection from the connection pool
    - release_connection(conn): Releases a connection back to the pool
    - initialize_tables(): Initializes required database tables
    - insert_article(article): Inserts a single article into reader-db
    - batch_insert_articles(articles): Inserts multiple articles in batches
    - insert_scraper_ids(article_ids): Inserts minimal article records with just scraper_ids
    - get_article_by_scraper_id(scraper_id): Retrieves an article by its scraper_id
    - get_error_articles(): Retrieves articles with null or error content
    - get_articles_needing_embedding(max_char_count): Gets articles without embeddings under character limit
    - insert_entity(entity): Inserts an entity into the database
    - link_article_entity(article_id, entity_id, mention_count): Links an article to an entity
    - insert_embedding(article_id, embedding): Inserts an embedding for an article
    - insert_cluster(centroid, is_hot): Inserts a new cluster
    - update_article_cluster(article_id, cluster_id): Updates an article's cluster assignment
    - insert_essay(essay): Inserts an essay into the database
    - link_essay_entity(essay_id, entity_id): Links an essay to an entity
    - get_similar_articles(embedding, limit): Finds articles with similar embeddings
    - get_hot_articles(limit): Gets articles marked as 'hot'
    - get_entities_by_influence(limit): Gets entities sorted by influence score
    - close(): Closes the connection pool

    Related files:
    - src/main.py: Uses this client to store processed articles
    - src/database/news_api_client.py: Fetches articles for processing
    - src/gemini/gemini_client.py: Generates embeddings for articles
    """

    def __init__(self,
                 host=os.getenv("READER_DB_HOST", "localhost"),
                 port=int(os.getenv("READER_DB_PORT", "5432")),
                 dbname=os.getenv("READER_DB_NAME", "reader_db"),
                 user=os.getenv("READER_DB_USER", "postgres"),
                 password=os.getenv("READER_DB_PASSWORD", "postgres"),
                 max_retries=5,
                 retry_delay=10):
        """Initialize the Reader DB client.

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database username
            password: Database password
            max_retries: Maximum number of connection retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_pool = None

        logger.info(
            f"Initializing connection to Reader DB at {host}:{port} with user '{user}' and database '{dbname}'")

        self._initialize_connection_pool()

        # Initialize all required tables
        self.initialize_tables()

    def _initialize_connection_pool(self):
        """Initialize the connection pool with retry logic."""
        retries = 0
        while retries < self.max_retries:
            try:
                self.connection_pool = pool.ThreadedConnectionPool(
                    1, 20,
                    host=self.host,
                    port=self.port,
                    dbname=self.dbname,
                    user=self.user,
                    password=self.password,
                    connect_timeout=10
                )
                logger.info(
                    f"Successfully connected to Reader DB at {self.host}:{self.port}")
                return
            except Exception as e:
                retries += 1
                wait_time = self.retry_delay * (1 + random.random())
                logger.error(f"Failed to connect to Reader DB: {e}")

                if retries >= self.max_retries:
                    raise

                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)

    def initialize_tables(self):
        """Initialize all required database tables if they don't exist."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check if any tables exist already
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]

            # Check if we need to migrate an existing articles table
            migrate_articles = 'articles' in existing_tables

            # Install pgvector extension if not already installed
            cursor.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
            """)

            # 1. Create or update articles table
            if migrate_articles:
                logger.info("Checking for articles table migration needs")

                # Check if it has the new schema
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'articles' 
                    AND table_schema = 'public';
                """)
                columns = [row[0] for row in cursor.fetchall()]

                # If missing required columns, rename old table and create new one
                if 'scraper_id' not in columns or 'is_hot' not in columns:
                    logger.info("Migrating articles table to new schema")

                    # Rename old table
                    cursor.execute("""
                        ALTER TABLE articles RENAME TO articles_old;
                    """)

                    # Create new table
                    cursor.execute("""
                        CREATE TABLE articles (
                            id SERIAL PRIMARY KEY,
                            scraper_id INTEGER UNIQUE,
                            title TEXT NOT NULL,
                            content TEXT NOT NULL,
                            pub_date TIMESTAMP,
                            domain TEXT,
                            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            is_hot BOOLEAN DEFAULT FALSE,
                            cluster_id INTEGER
                        );
                    """)

                    # Migrate data
                    try:
                        cursor.execute("""
                            INSERT INTO articles (
                                scraper_id, title, content, pub_date, domain
                            )
                            SELECT 
                                original_id as scraper_id,
                                title,
                                content,
                                pub_date,
                                domain
                            FROM articles_old
                            WHERE title IS NOT NULL AND content IS NOT NULL;
                        """)
                        logger.info(
                            "Data migration from old articles table completed")
                    except Exception as e:
                        logger.error(f"Error migrating article data: {e}")

                    conn.commit()
            else:
                logger.info("Creating articles table")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS articles (
                        id SERIAL PRIMARY KEY,
                        scraper_id INTEGER UNIQUE,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        pub_date TIMESTAMP,
                        domain TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_hot BOOLEAN DEFAULT FALSE,
                        cluster_id INTEGER
                    );
                """)

            # 2. Create entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    influence_score FLOAT DEFAULT 0.0,
                    mentions INTEGER DEFAULT 0,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP,
                    UNIQUE (name, type)
                );
            """)

            # 3. Create article_entities junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS article_entities (
                    article_id INTEGER REFERENCES articles(id),
                    entity_id INTEGER REFERENCES entities(id),
                    mention_count INTEGER DEFAULT 1,
                    PRIMARY KEY (article_id, entity_id)
                );
            """)

            # 4. Create embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER UNIQUE REFERENCES articles(id),
                    embedding VECTOR(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 5. Create clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id SERIAL PRIMARY KEY,
                    centroid VECTOR(768),
                    article_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_hot BOOLEAN DEFAULT FALSE
                );
            """)

            # 6. Create essays table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS essays (
                    id SERIAL PRIMARY KEY,
                    type TEXT NOT NULL,
                    article_id INTEGER REFERENCES articles(id),
                    content TEXT NOT NULL,
                    layer_depth INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT[]
                );
            """)

            # 7. Create essay_entities junction table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS essay_entities (
                    essay_id INTEGER REFERENCES essays(id),
                    entity_id INTEGER REFERENCES entities(id),
                    PRIMARY KEY (essay_id, entity_id)
                );
            """)

            conn.commit()
            logger.info(
                "Successfully initialized all required database tables")

        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.release_connection(conn)

    def get_connection(self):
        """Get a connection from the pool."""
        if not self.connection_pool:
            self._initialize_connection_pool()
        return self.connection_pool.getconn()

    def release_connection(self, conn):
        """Release a connection back to the pool."""
        if self.connection_pool:
            self.connection_pool.putconn(conn)

    def test_connection(self):
        """Test the database connection and return connection details."""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Test connection with simple query
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # Get database info
            cursor.execute("SELECT current_database(), current_user;")
            db_info = cursor.fetchone()

            # Get schema info
            cursor.execute(
                "SELECT schema_name FROM information_schema.schemata;")
            schemas = [row[0] for row in cursor.fetchall()]

            # Get table info from public schema
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]

            cursor.close()

            return {
                "version": version,
                "database": db_info[0],
                "user": db_info[1],
                "schemas": schemas,
                "tables": tables,
                "connection_string": f"{self.user}@{self.host}:{self.port}/{self.dbname}"
            }

        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return {"error": str(e)}

        finally:
            if conn:
                self.release_connection(conn)

    def insert_article(self, article: Dict[str, Any]) -> Optional[int]:
        """Insert an article into the reader-db.

        Args:
            article: The article data to insert

        Returns:
            The new article ID if successful, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Insert the article with the new schema
            logger.info(
                f"Inserting article: {article.get('title', '')[:50]}...")

            cursor.execute("""
                INSERT INTO articles (
                    scraper_id, title, content, 
                    pub_date, domain
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (scraper_id) DO UPDATE
                SET title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    domain = EXCLUDED.domain,
                    pub_date = EXCLUDED.pub_date,
                    processed_at = CURRENT_TIMESTAMP
                RETURNING id;
            """, (
                article.get('id') or article.get('scraper_id'),
                article.get('title', ''),
                article.get('content', ''),
                article.get('pub_date'),
                article.get('domain', '')
            ))

            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id

        except Exception as e:
            logger.error(f"Error inserting article: {e}")
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                self.release_connection(conn)

    def insert_scraper_ids(self, article_ids: List[int]) -> int:
        """
        Insert article IDs into reader-db as scraper_id with minimal required fields

        Args:
            article_ids: List of article IDs to insert as scraper_id

        Returns:
            int: Number of successfully inserted records
        """
        if not article_ids:
            logger.info("No article IDs to insert")
            return 0

        logger.info(
            f"Attempting to insert {len(article_ids)} article IDs into reader-db...")

        successful_inserts = 0
        failed_inserts = 0

        try:
            # First verify connection and tables
            test_result = self.test_connection()
            if "error" in test_result:
                logger.error(
                    f"Cannot connect to reader-db: {test_result['error']}")
                return 0

            logger.info(
                f"Successfully connected to reader-db, available tables: {', '.join(test_result.get('tables', []))}")

            # Process and insert IDs in batches
            batch_size = 100
            total_batches = (len(article_ids) + batch_size - 1) // batch_size

            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(article_ids))
                current_batch = article_ids[start_idx:end_idx]

                logger.info(
                    f"Processing batch {batch_num+1}/{total_batches} ({len(current_batch)} articles)")

                for article_id in current_batch:
                    # Create minimal article object with required fields
                    article = {
                        'scraper_id': article_id,
                        # Minimal required field
                        'title': f"Article {article_id}",
                        # Minimal required field
                        'content': f"Content for article {article_id}",
                    }

                    # Insert article
                    new_id = self.insert_article(article)

                    if new_id:
                        successful_inserts += 1
                        if successful_inserts % 100 == 0:
                            logger.info(
                                f"Inserted {successful_inserts} articles so far")
                    else:
                        failed_inserts += 1

            logger.info(
                f"Insertion complete: {successful_inserts} successful, {failed_inserts} failed")
            return successful_inserts

        except Exception as e:
            logger.error(f"Error inserting articles into reader-db: {e}")
            return successful_inserts

    def get_article_by_scraper_id(self, scraper_id: int) -> Optional[Dict[str, Any]]:
        """Get an article by its scraper_id.

        Args:
            scraper_id: The scraper_id of the article to retrieve

        Returns:
            Optional[Dict[str, Any]]: The article data or None if not found
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            query = "SELECT * FROM articles WHERE scraper_id = %s"
            cursor.execute(query, (scraper_id,))

            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()

            if row:
                article = dict(zip(columns, row))
                return article

            return None

        except Exception as e:
            logger.error(f"Error retrieving article by scraper_id: {e}")
            return None

        finally:
            if conn:
                self.release_connection(conn)

    def insert_entity(self, entity: Dict[str, Any]) -> Optional[int]:
        """Insert an entity into the database.

        Args:
            entity: Dictionary containing entity data with keys:
                   name, type, and optionally influence_score, mentions

        Returns:
            The new entity ID if successful, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Insert the entity with ON CONFLICT to handle duplicates
            cursor.execute("""
                INSERT INTO entities (
                    name, type, influence_score, mentions, last_seen
                )
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (name, type) DO UPDATE
                SET influence_score = 
                    CASE 
                        WHEN entities.influence_score < EXCLUDED.influence_score 
                        THEN EXCLUDED.influence_score 
                        ELSE entities.influence_score 
                    END,
                    mentions = entities.mentions + EXCLUDED.mentions,
                    last_seen = CURRENT_TIMESTAMP
                RETURNING id;
            """, (
                entity.get('name'),
                entity.get('type'),
                entity.get('influence_score', 0.0),
                entity.get('mentions', 1)
            ))

            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id

        except Exception as e:
            logger.error(f"Error inserting entity: {e}")
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                self.release_connection(conn)

    def link_article_entity(self, article_id: int, entity_id: int, mention_count: int = 1) -> bool:
        """Link an article to an entity in the article_entities junction table.

        Args:
            article_id: ID of the article
            entity_id: ID of the entity
            mention_count: Number of times the entity is mentioned in the article

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO article_entities (article_id, entity_id, mention_count)
                VALUES (%s, %s, %s)
                ON CONFLICT (article_id, entity_id) DO UPDATE
                SET mention_count = article_entities.mention_count + EXCLUDED.mention_count;
            """, (article_id, entity_id, mention_count))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error linking article to entity: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.release_connection(conn)

    # Embedding methods
    def insert_embedding(self, article_id: int, embedding: List[float]) -> Optional[int]:
        """Insert an embedding for an article.

        Args:
            article_id: ID of the article
            embedding: Vector of embedding values (768 dimensions)

        Returns:
            The new embedding ID if successful, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO embeddings (article_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (article_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    created_at = CURRENT_TIMESTAMP
                RETURNING id;
            """, (article_id, embedding))

            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id

        except Exception as e:
            logger.error(f"Error inserting embedding: {e}")
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                self.release_connection(conn)

    # Cluster methods
    def insert_cluster(self, centroid: List[float], is_hot: bool = False) -> Optional[int]:
        """Insert a new cluster.

        Args:
            centroid: Vector representing the cluster centroid (768 dimensions)
            is_hot: Whether this is a "hot" cluster

        Returns:
            The new cluster ID if successful, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO clusters (centroid, is_hot)
                VALUES (%s, %s)
                RETURNING id;
            """, (centroid, is_hot))

            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id

        except Exception as e:
            logger.error(f"Error inserting cluster: {e}")
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                self.release_connection(conn)

    def update_article_cluster(self, article_id: int, cluster_id: int) -> bool:
        """Update an article's cluster assignment.

        Args:
            article_id: ID of the article
            cluster_id: ID of the cluster

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Update the article
            cursor.execute("""
                UPDATE articles
                SET cluster_id = %s
                WHERE id = %s;
            """, (cluster_id, article_id))

            # Update the cluster's article count
            cursor.execute("""
                UPDATE clusters
                SET article_count = article_count + 1
                WHERE id = %s;
            """, (cluster_id,))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating article cluster: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.release_connection(conn)

    # Essay methods
    def insert_essay(self, essay: Dict[str, Any]) -> Optional[int]:
        """Insert an essay into the database.

        Args:
            essay: Dictionary containing essay data with keys:
                  type, content, and optionally article_id, layer_depth, tags

        Returns:
            The new essay ID if successful, None otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO essays (
                    type, article_id, content, layer_depth, tags
                )
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                essay.get('type'),
                essay.get('article_id'),
                essay.get('content'),
                essay.get('layer_depth'),
                essay.get('tags', [])
            ))

            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id

        except Exception as e:
            logger.error(f"Error inserting essay: {e}")
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                self.release_connection(conn)

    def link_essay_entity(self, essay_id: int, entity_id: int) -> bool:
        """Link an essay to an entity in the essay_entities junction table.

        Args:
            essay_id: ID of the essay
            entity_id: ID of the entity

        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO essay_entities (essay_id, entity_id)
                VALUES (%s, %s)
                ON CONFLICT (essay_id, entity_id) DO NOTHING;
            """, (essay_id, entity_id))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error linking essay to entity: {e}")
            if conn:
                conn.rollback()
            return False

        finally:
            if conn:
                self.release_connection(conn)

    # Query methods for advanced searches
    def get_similar_articles(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Find articles with similar embeddings using cosine similarity.

        Args:
            embedding: Vector to compare against (768 dimensions)
            limit: Maximum number of results to return

        Returns:
            List of articles sorted by similarity
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT a.*, 1 - (e.embedding <=> %s) as similarity
                FROM articles a
                JOIN embeddings e ON a.id = e.article_id
                ORDER BY similarity DESC
                LIMIT %s;
            """, (embedding, limit))

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except Exception as e:
            logger.error(f"Error finding similar articles: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def get_hot_articles(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get articles marked as 'hot'.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of hot articles
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM articles
                WHERE is_hot = TRUE
                ORDER BY processed_at DESC
                LIMIT %s;
            """, (limit,))

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except Exception as e:
            logger.error(f"Error retrieving hot articles: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def get_entities_by_influence(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get entities sorted by influence score.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of entities
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM entities
                ORDER BY influence_score DESC
                LIMIT %s;
            """, (limit,))

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            return results

        except Exception as e:
            logger.error(f"Error retrieving entities by influence: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def close(self):
        """Close the connection pool."""
        if hasattr(self, 'connection_pool') and self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Closed all Reader DB connections")

    def batch_insert_articles(self, articles: List[Dict[str, Any]]) -> int:
        """
        Insert multiple processed articles into reader-db in batches.

        Args:
            articles: List of prepared article dictionaries with fields:
                     scraper_id, title, pub_date, domain, content, processed_at

        Returns:
            int: Number of successfully inserted records
        """
        if not articles:
            logger.info("No articles to insert")
            return 0

        logger.info(
            f"Preparing to insert {len(articles)} articles into reader-db in batches...")

        successful_inserts = 0
        failed_inserts = 0

        # Process and insert articles in batches
        batch_size = 100
        total_batches = (len(articles) + batch_size - 1) // batch_size

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(articles))
            current_batch = articles[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_num+1}/{total_batches} ({len(current_batch)} articles)")

            conn = None
            try:
                conn = self.get_connection()
                cursor = conn.cursor()

                for article in current_batch:
                    # Handle processed_at field
                    processed_at = article.get('processed_at')

                    # Convert processed_at to proper format if it's a string
                    if isinstance(processed_at, str):
                        try:
                            processed_at = dateutil_parser.parse(processed_at)
                        except:
                            # If parsing fails, use current timestamp
                            processed_at = datetime.now()

                    # Insert with content and processed_at fields
                    cursor.execute("""
                        INSERT INTO articles (
                            scraper_id, title, pub_date, domain, content, processed_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (scraper_id) DO UPDATE
                        SET title = EXCLUDED.title,
                            pub_date = EXCLUDED.pub_date,
                            domain = EXCLUDED.domain,
                            content = EXCLUDED.content,
                            processed_at = EXCLUDED.processed_at
                        RETURNING id;
                    """, (
                        article.get('scraper_id'),
                        article.get(
                            'title', f"Article {article.get('scraper_id')}"),
                        article.get('pub_date'),
                        article.get('domain', ''),
                        article.get('content', ''),
                        processed_at  # Use the processed value
                    ))

                    result = cursor.fetchone()
                    if result and result[0]:
                        successful_inserts += 1
                    else:
                        failed_inserts += 1

                conn.commit()
                if successful_inserts % 100 == 0 and successful_inserts > 0:
                    logger.info(
                        f"Inserted {successful_inserts} articles so far")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num+1}: {e}")
                if conn:
                    conn.rollback()
            finally:
                if conn:
                    self.release_connection(conn)

        logger.info(
            f"Batch insertion complete: {successful_inserts} successful, {failed_inserts} failed")
        return successful_inserts

    def get_error_articles(self) -> List[Dict[str, Any]]:
        """
        Get all articles with null or error content.

        Returns:
            List[Dict[str, Any]]: List of articles with null or error content
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, scraper_id, title, content, pub_date, domain, processed_at
                FROM articles
                WHERE content IS NULL OR content = 'ERROR' OR LENGTH(TRIM(content)) = 0
                ORDER BY processed_at DESC;
            """)

            columns = [desc[0] for desc in cursor.description]
            error_articles = []

            for row in cursor.fetchall():
                error_articles.append(dict(zip(columns, row)))

            logger.info(
                f"Found {len(error_articles)} articles with null/error content")
            return error_articles

        except Exception as e:
            logger.error(
                f"Error retrieving articles with null/error content: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)

    def get_articles_needing_embedding(self, max_char_count: int = 8000) -> List[Dict[str, Any]]:
        """
        Get articles that need embeddings and have a content length under the specified character count.

        This method selects articles with valid character counts instead of truncating them,
        as there's a future plan for handling long articles separately.

        Args:
            max_char_count: Maximum number of characters in the article content (default: 8000)
                           to stay within Gemini embedding model's token limit (~2048 tokens at 4 chars/token)

        Returns:
            List[Dict[str, Any]]: List of articles needing embeddings, each containing 'id' and 'content'
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Query for articles that have valid content and don't have embeddings yet
            # This query retrieves all articles first, then we filter by length in Python
            cursor.execute("""
                SELECT a.id, a.content 
                FROM articles a
                LEFT JOIN embeddings e ON a.id = e.article_id
                WHERE 
                    a.content IS NOT NULL 
                    AND a.content != 'ERROR'
                    AND LENGTH(TRIM(a.content)) > 0
                    AND e.article_id IS NULL;
            """)

            # Fetch all results
            results = cursor.fetchall()
            total_articles = len(results)

            # Filter based on character count
            filtered_articles = []
            skipped_articles = 0

            for article_id, content in results:
                # Check character count
                char_count = len(content)
                if char_count <= max_char_count:
                    filtered_articles.append({
                        'id': article_id,
                        'content': content
                    })
                else:
                    skipped_articles += 1

            # More detailed logging to track skipped articles
            logger.info(f"Found {len(filtered_articles)} articles suitable for embedding "
                        f"(skipped {skipped_articles} articles exceeding {max_char_count} characters)")
            return filtered_articles

        except Exception as e:
            logger.error(f"Error retrieving articles needing embeddings: {e}")
            return []

        finally:
            if conn:
                self.release_connection(conn)
