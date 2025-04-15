"""
reader_db_client.py - Client for interacting with the Reader database

This module provides a client class for interacting with the Reader database,
which stores articles, embeddings, clusters, and related data.

Exported classes:
- ReaderDBClient: Main client class for database operations
  - __init__(self, host, port, dbname, user, password, max_retries, retry_delay)
  - initialize_tables(self) -> bool
  - get_connection(self) -> Connection
  - release_connection(self, conn) -> None
  - test_connection(self) -> Dict[str, Any]
  - close(self) -> None
  Plus many domain-specific methods that delegate to the appropriate modules

Related files:
- All modules in src/database/modules/ for specific operation implementations
- src/steps/step1.py: Uses this client for article processing
- src/steps/step2.py: Uses this client for clustering operations
"""

import logging
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from datetime import datetime

# Import the modules
from src.database.modules import connection
from src.database.modules import schema
from src.database.modules import articles
from src.database.modules import entities
from src.database.modules import embeddings
from src.database.modules import clusters
from src.database.modules import dimensionality
from src.database.modules import essays

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReaderDBClient:
    """Client for interacting with the Reader database."""

    def __init__(self,
                 host=os.getenv("READER_DB_HOST", "localhost"),
                 port=int(os.getenv("READER_DB_PORT", "5432")),
                 dbname=os.getenv("READER_DB_NAME", "reader_db"),
                 user=os.getenv("READER_DB_USER", "postgres"),
                 password=os.getenv("READER_DB_PASSWORD", "postgres"),
                 max_retries=5,
                 retry_delay=10):
        """
        Initialize the Reader database client.

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database user
            password: Database password
            max_retries: Number of connection retry attempts
            retry_delay: Delay between retry attempts
        """
        self.db_config = {
            'host': host,
            'port': port,
            'dbname': dbname,
            'user': user,
            'password': password,
            'max_retries': max_retries,
            'retry_delay': retry_delay
        }

        # Initialize connection pool
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        """Initialize the database connection pool."""
        self.connection_pool = connection.initialize_connection_pool(
            self.db_config)
        if not self.connection_pool:
            logger.critical("Failed to initialize connection pool")
            raise ConnectionError("Could not connect to database")

    def initialize_tables(self) -> bool:
        """
        Initialize database tables if they don't exist.

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                result = schema.initialize_tables(conn)
                return result
            finally:
                self.release_connection(conn)
        return False

    def get_connection(self):
        """
        Get a connection from the pool.

        Returns:
            Connection: Database connection
        """
        return connection.get_connection(self.connection_pool)

    def release_connection(self, conn):
        """
        Release a connection back to the pool.

        Args:
            conn: Database connection
        """
        connection.release_connection(self.connection_pool, conn)

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the database connection and return information about the database.

        Returns:
            Dict[str, Any]: Connection details and database information
        """
        result = {
            "status": "error",
            "message": "Connection test failed",
            "database": self.db_config["dbname"],
            "host": self.db_config["host"],
            "port": self.db_config["port"]
        }

        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()

                # Test basic query
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]

                # Get table counts
                cursor.execute("""
                    SELECT table_name, 
                           (SELECT count(*) FROM information_schema.columns 
                            WHERE table_name = t.table_name) as column_count
                    FROM (
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    ) t;
                """)
                tables = {row[0]: row[1] for row in cursor.fetchall()}

                # Get record counts for main tables
                counts = {}
                main_tables = ["articles", "embeddings",
                               "clusters", "entities"]
                for table in main_tables:
                    if table in tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table};")
                            counts[table] = cursor.fetchone()[0]
                        except:
                            counts[table] = "error"

                cursor.close()

                # Update result
                result.update({
                    "status": "connected",
                    "message": "Connection successful",
                    "version": version,
                    "tables": tables,
                    "record_counts": counts
                })

            except Exception as e:
                result["error"] = str(e)
            finally:
                self.release_connection(conn)

        return result

    # Article operations
    def insert_article(self, article: Dict[str, Any]) -> Optional[int]:
        """
        Insert an article into the database.

        Args:
            article: The article data

        Returns:
            int or None: The new article ID if successful, None otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.insert_article(conn, article)
            finally:
                self.release_connection(conn)
        return None

    def insert_scraper_ids(self, article_ids: List[int]) -> int:
        """
        Insert minimal article records just from IDs.

        This is useful for storing article IDs that we know about but
        haven't processed yet.

        Args:
            article_ids: List of article scraper IDs to insert

        Returns:
            int: Number of successfully inserted article IDs
        """
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()
                inserted_count = 0

                for article_id in article_ids:
                    try:
                        cursor.execute("""
                            INSERT INTO articles (scraper_id, title)
                            VALUES (%s, %s)
                            ON CONFLICT (scraper_id) DO NOTHING;
                        """, (article_id, f"Article {article_id}"))

                        if cursor.rowcount > 0:
                            inserted_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error inserting scraper ID {article_id}: {e}")

                conn.commit()
                cursor.close()
                return inserted_count
            finally:
                self.release_connection(conn)
        return 0

    def batch_insert_articles(self, articles_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Insert multiple articles in a batch.

        Args:
            articles_data: List of article dictionaries

        Returns:
            Dict[str, int]: Dictionary with success and failure counts
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.batch_insert_articles(conn, articles_data)
            finally:
                self.release_connection(conn)
        return {"success": 0, "failure": 0}

    def get_article_by_id(self, article_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an article by its ID.

        Args:
            article_id: The article ID

        Returns:
            Dict or None: The article data or None if not found
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_article_by_id(conn, article_id)
            finally:
                self.release_connection(conn)
        return None

    def get_article_by_scraper_id(self, scraper_id: int) -> Optional[Dict[str, Any]]:
        """
        Get an article by its scraper_id.

        Args:
            scraper_id: The scraper_id of the article

        Returns:
            Dict or None: The article data or None if not found
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_article_by_scraper_id(conn, scraper_id)
            finally:
                self.release_connection(conn)
        return None

    def get_articles_needing_embedding(self, max_char_count: int = 8000) -> List[Dict[str, Any]]:
        """
        Get articles that need embedding generation.

        Args:
            max_char_count: Maximum character count for articles to process

        Returns:
            List[Dict[str, Any]]: List of articles that need embeddings
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_articles_needing_embedding(conn, max_char_count)
            finally:
                self.release_connection(conn)
        return []

    def get_articles_needing_summarization(self, min_char_count: int = 8000) -> List[Dict[str, Any]]:
        """
        Get articles that need summarization because their content length exceeds the specified limit.

        Args:
            min_char_count: Minimum character count for an article to require summarization

        Returns:
            List[Dict[str, Any]]: List of articles needing summarization
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_articles_needing_summarization(conn, min_char_count)
            finally:
                self.release_connection(conn)
        return []

    def get_error_articles(self) -> List[Dict[str, Any]]:
        """
        Get all articles with null or error content.

        Returns:
            List[Dict[str, Any]]: List of articles with errors
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_error_articles(conn)
            finally:
                self.release_connection(conn)
        return []

    def get_hot_articles(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get articles marked as 'hot'.

        Args:
            limit: Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of hot articles
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_hot_articles(conn, limit)
            finally:
                self.release_connection(conn)
        return []

    def get_sample_titles_for_articles(self, article_ids: List[int], sample_size: int) -> List[str]:
        """
        Get a sample of titles for a list of article IDs.

        Args:
            article_ids: List of article IDs
            sample_size: Maximum number of titles to return

        Returns:
            List[str]: Sample list of article titles
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.get_sample_titles_for_articles(conn, article_ids, sample_size)
            finally:
                self.release_connection(conn)
        return []

    def get_publication_dates_for_articles(self, article_ids: List[int]) -> Dict[int, Optional[datetime]]:
        """
        Get publication dates for a list of article IDs.

        This is the implementation required by the step2 refactoring to delegate 
        DB operations to the ReaderDBClient.

        Args:
            article_ids: List of article IDs to fetch publication dates for

        Returns:
            Dict[int, Optional[datetime]]: Dictionary mapping article IDs to their publication dates
                                          (None for articles without a date or not found)
        """
        conn = self.get_connection()
        if conn:
            try:
                # Explicitly call the get_publication_dates_for_articles function from the articles module
                return articles.get_publication_dates_for_articles(conn, article_ids)
            finally:
                self.release_connection(conn)
        return {}

    # Entity operations
    def insert_entity(self, entity: Dict[str, Any]) -> Optional[int]:
        """
        Insert an entity into the database.

        Args:
            entity: The entity data

        Returns:
            int or None: The new entity ID if successful, None otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return entities.insert_entity(conn, entity)
            finally:
                self.release_connection(conn)
        return None

    def link_article_entity(self, article_id: int, entity_id: int, mention_count: int = 1) -> bool:
        """
        Link an article to an entity with mention count.

        Args:
            article_id: ID of the article
            entity_id: ID of the entity
            mention_count: Number of mentions of the entity in the article

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return entities.link_article_entity(conn, article_id, entity_id, mention_count)
            finally:
                self.release_connection(conn)
        return False

    def get_entities_by_influence(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get entities sorted by influence score.

        Args:
            limit: Maximum number of entities to return

        Returns:
            List[Dict[str, Any]]: List of entities with influence scores
        """
        conn = self.get_connection()
        if conn:
            try:
                return entities.get_entities_by_influence(conn, limit)
            finally:
                self.release_connection(conn)
        return []

    def get_entity_influence_for_articles(self, article_ids: List[int]) -> Dict[int, float]:
        """
        Get entity influence scores for a list of articles.

        Args:
            article_ids: List of article IDs

        Returns:
            Dict[int, float]: Dictionary mapping article IDs to influence scores
        """
        conn = self.get_connection()
        if conn:
            try:
                return entities.get_entity_influence_for_articles(conn, article_ids)
            finally:
                self.release_connection(conn)
        return {}

    # Embedding operations
    def insert_embedding(self, article_id: int, embedding_data: Dict[str, Any]) -> bool:
        """
        Insert an embedding for an article.

        Args:
            article_id: ID of the article
            embedding_data: Data containing the embedding vector and metadata

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return embeddings.insert_embedding(conn, article_id, embedding_data)
            finally:
                self.release_connection(conn)
        return False

    def batch_insert_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Insert multiple embeddings into the database in batches.

        Args:
            embeddings_data: List of dictionaries with article_id and embedding data

        Returns:
            Dict[str, int]: Dictionary with success and failure counts
        """
        conn = self.get_connection()
        if conn:
            try:
                return embeddings.batch_insert_embeddings(conn, embeddings_data)
            finally:
                self.release_connection(conn)
        return {"success": 0, "failure": 0}

    def get_all_embeddings(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all embeddings from the database.

        Args:
            limit: Optional limit on the number of embeddings to return

        Returns:
            List[Dict[str, Any]]: List of embedding records with article_id and vector
        """
        conn = self.get_connection()
        if conn:
            try:
                return embeddings.get_all_embeddings(conn, limit)
            finally:
                self.release_connection(conn)
        return []

    def get_all_embeddings_with_pub_date(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all embeddings with their corresponding article publication dates.

        Args:
            limit: Optional limit on the number of embeddings to return

        Returns:
            List[Dict[str, Any]]: List of embedding records with article_id, vector, and pub_date
        """
        conn = self.get_connection()
        if conn:
            try:
                return embeddings.get_all_embeddings_with_pub_date(conn, limit)
            finally:
                self.release_connection(conn)
        return []

    def get_embedding_for_article(self, article_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the embedding for a specific article.

        Args:
            article_id: ID of the article

        Returns:
            Dict[str, Any] or None: The embedding data or None if not found
        """
        conn = self.get_connection()
        if conn:
            try:
                return embeddings.get_embedding_for_article(conn, article_id)
            finally:
                self.release_connection(conn)
        return None

    def get_similar_articles(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get articles similar to the provided embedding.

        Args:
            embedding: Embedding vector to compare
            limit: Maximum number of similar articles to return

        Returns:
            List[Dict[str, Any]]: List of similar articles with similarity scores
        """
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()

                # Use pgvector to find similar articles
                cursor.execute("""
                    SELECT a.id, a.title, a.pub_date, a.domain, 
                           1 - (e.vector <=> %s) as similarity
                    FROM embeddings e
                    JOIN articles a ON e.article_id = a.id
                    ORDER BY e.vector <=> %s ASC
                    LIMIT %s;
                """, (embedding, embedding, limit))

                similar_articles = []
                for row in cursor.fetchall():
                    similar_articles.append({
                        'id': row[0],
                        'title': row[1],
                        'pub_date': row[2],
                        'domain': row[3],
                        'similarity': row[4]
                    })

                cursor.close()
                return similar_articles
            finally:
                self.release_connection(conn)
        return []

    # Cluster operations
    def insert_cluster(self, name: str = None, description: Optional[str] = None,
                       centroid: Optional[List[float]] = None, is_hot: bool = False) -> Optional[int]:
        """
        Insert a new cluster.

        Args:
            name: Name of the cluster (optional, stored in metadata)
            description: Optional description (stored in metadata)
            centroid: Optional centroid vector of the cluster
            is_hot: Whether the cluster is hot

        Returns:
            int or None: Cluster ID if successful, None otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                # Generate a name if only centroid is provided
                if centroid is not None and name is None:
                    name = f"Cluster-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                elif name is None:
                    # Ensure name is not None if centroid is also None, provide a default
                    name = f"Cluster-NoCentroid-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

                # Call the module function to handle the creation
                cluster_id = clusters.create_cluster(
                    conn, name, description, centroid, is_hot)
                return cluster_id
            except Exception as e:
                # Log the error originating from the module or the client call itself
                logger.error(
                    f"Error in insert_cluster operation: {e}", exc_info=True)
                return None  # Ensure None is returned on error
            finally:
                self.release_connection(conn)
        return None

    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """
        Get all clusters.

        Returns:
            List[Dict[str, Any]]: List of all clusters
        """
        conn = self.get_connection()
        if conn:
            try:
                return clusters.get_all_clusters(conn)
            finally:
                self.release_connection(conn)
        return []

    def update_article_cluster(self, article_id: int, cluster_id: Optional[int], is_hot: bool = False) -> bool:
        """
        Update an article's cluster assignment.

        Args:
            article_id: ID of the article
            cluster_id: ID of the cluster (None for no cluster)
            is_hot: Whether the article is hot

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.update_article_cluster(conn, article_id, cluster_id, is_hot)
            finally:
                self.release_connection(conn)
        return False

    def batch_update_article_clusters(self, assignments: List[Tuple[int, Optional[int], bool]]) -> Dict[str, int]:
        """
        Update multiple article cluster assignments in batch.

        Args:
            assignments: List of tuples (article_id, cluster_id, is_hot)

        Returns:
            Dict[str, int]: Dictionary with success and failure counts
        """
        conn = self.get_connection()
        if conn:
            try:
                return articles.batch_update_article_clusters(conn, assignments)
            finally:
                self.release_connection(conn)
        return {"success": 0, "failure": 0}

    def get_articles_by_cluster(self, cluster_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get articles belonging to a specific cluster.

        Args:
            cluster_id: ID of the cluster
            limit: Optional limit on the number of articles to return

        Returns:
            List[Dict[str, Any]]: List of articles in the cluster
        """
        conn = self.get_connection()
        if conn:
            try:
                return clusters.get_articles_by_cluster(conn, cluster_id, limit)
            finally:
                self.release_connection(conn)
        return []

    def update_cluster_metadata(self, cluster_id: int, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a cluster.

        Args:
            cluster_id: ID of the cluster to update
            metadata: Dictionary of metadata to update

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()

                # Get existing metadata
                cursor.execute("""
                    SELECT metadata FROM clusters WHERE id = %s
                """, (cluster_id,))

                row = cursor.fetchone()
                if not row:
                    logger.error(f"Cluster {cluster_id} not found")
                    cursor.close()
                    return False

                existing_metadata = row[0] if row[0] else {}
                if isinstance(existing_metadata, str):
                    try:
                        existing_metadata = json.loads(existing_metadata)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Could not decode existing metadata for cluster {cluster_id}. Overwriting.")
                        existing_metadata = {}

                # Merge existing with new metadata
                updated_metadata = {**existing_metadata, **metadata}

                # Update the cluster, serializing the metadata dictionary to JSON
                # Also remove updated_at to avoid potential column absence issues
                cursor.execute("""
                    UPDATE clusters
                    SET metadata = %s
                    WHERE id = %s
                    RETURNING id;
                """, (json.dumps(updated_metadata), cluster_id))

                result = cursor.fetchone() is not None
                conn.commit()
                cursor.close()
                return result
            except Exception as e:
                logger.error(f"Error updating cluster metadata: {e}")
                conn.rollback()
                return False
            finally:
                self.release_connection(conn)
        return False

    def update_cluster_article_count(self, cluster_id: int, article_count: int) -> bool:
        """
        Update the article count for a cluster.

        Args:
            cluster_id: ID of the cluster
            article_count: New article count

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT metadata FROM clusters WHERE id = %s
                """, (cluster_id,))

                row = cursor.fetchone()
                if not row:
                    logger.error(f"Cluster {cluster_id} not found")
                    return False

                existing_metadata = row[0] if row[0] else {}

                # Update article count in metadata
                existing_metadata['article_count'] = article_count

                # Update the cluster metadata
                cursor.execute("""
                    UPDATE clusters
                    SET metadata = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id;
                """, (existing_metadata, cluster_id))

                result = cursor.fetchone() is not None
                conn.commit()
                cursor.close()
                return result
            except Exception as e:
                logger.error(f"Error updating cluster article count: {e}")
                conn.rollback()
                return False
            finally:
                self.release_connection(conn)
        return False

    # Essay operations
    def insert_essay(self, essay: Dict[str, Any]) -> Optional[int]:
        """
        Insert an essay into the database.

        Args:
            essay: The essay data

        Returns:
            int or None: The new essay ID if successful, None otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return essays.insert_essay(conn, essay)
            finally:
                self.release_connection(conn)
        return None

    def link_essay_entity(self, essay_id: int, entity_id: int) -> bool:
        """
        Link an essay to an entity.

        Args:
            essay_id: ID of the essay
            entity_id: ID of the entity

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return essays.link_essay_entity(conn, essay_id, entity_id)
            finally:
                self.release_connection(conn)
        return False

    # Dimensionality reduction operations
    def store_coordinates(self, article_id: int, coordinates: Dict[str, Any], method: str) -> bool:
        """
        Store dimensionality reduction coordinates for an article.

        Args:
            article_id: ID of the article
            coordinates: Dictionary containing the coordinates and metadata
            method: Dimensionality reduction method (e.g., 'umap', 'tsne', 'pca')

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return dimensionality.store_coordinates(conn, article_id, coordinates, method)
            finally:
                self.release_connection(conn)
        return False

    def batch_store_coordinates(self, coordinates_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Store multiple sets of coordinates in batches.

        Args:
            coordinates_data: List of dictionaries containing article_id, coordinates, method

        Returns:
            Dict[str, int]: Dictionary with counts of successful and failed operations
        """
        conn = self.get_connection()
        if conn:
            try:
                return dimensionality.batch_store_coordinates(conn, coordinates_data)
            finally:
                self.release_connection(conn)
        return {"success_count": 0, "error_count": 0}

    def get_coordinates(self, article_id: int, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get dimensionality reduction coordinates for an article.

        Args:
            article_id: ID of the article
            method: Optional method filter (e.g., 'umap', 'tsne', 'pca')

        Returns:
            List[Dict[str, Any]]: List of coordinate data for the article
        """
        conn = self.get_connection()
        if conn:
            try:
                return dimensionality.get_coordinates(conn, article_id, method)
            finally:
                self.release_connection(conn)
        return []

    def update_coordinates_config(self, config: Dict[str, Any], method: str) -> bool:
        """
        Update configuration metadata for a dimensionality reduction method.

        Args:
            config: Configuration dictionary
            method: Dimensionality reduction method

        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.get_connection()
        if conn:
            try:
                return dimensionality.update_coordinates_config(conn, config, method)
            finally:
                self.release_connection(conn)
        return False

    def get_coordinates_config(self, method: str) -> Optional[Dict[str, Any]]:
        """
        Get the configuration for a dimensionality reduction method.

        Args:
            method: Dimensionality reduction method

        Returns:
            Dict[str, Any] or None: The configuration data or None if not found
        """
        conn = self.get_connection()
        if conn:
            try:
                return dimensionality.get_coordinates_config(conn, method)
            finally:
                self.release_connection(conn)
        return None

    def close(self):
        """Close the database connection pool."""
        if hasattr(self, 'connection_pool') and self.connection_pool:
            connection.close_pool(self.connection_pool)
