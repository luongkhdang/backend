"""
haystack_client.py - Haystack RAG Pipeline Client

This module provides an interface to Haystack 2.x components for building and running
retrieval-augmented generation pipelines. It handles document retrieval, ranking,
and integration with the Gemini API for essay generation.

Exported functions:
- get_document_store(): Returns configured PgvectorDocumentStore
- get_text_embedder(): Returns GeminiTextEmbedder
- get_embedding_retriever(document_store): Returns PgvectorEmbeddingRetriever
- get_ranker(): Returns configured Ranker component
- get_prompt_builder(template_path): Returns PromptBuilder component
- get_gemini_generator(): Returns GoogleAIGeminiGenerator component
- build_retrieval_pipeline(embedder, retriever): Builds retrieval pipeline
- build_retrieval_ranking_pipeline(embedder, retriever, ranker): Builds retrieval+ranking pipeline
- run_article_retrieval(query_text): Runs retrieval pipeline for a query
- run_article_retrieval_and_ranking(query_text): Runs retrieval+ranking pipeline for a query

Related files:
- src/database/modules/haystack_db.py: Database integration module
- src/gemini/gemini_client.py: Gemini API client for model access
- src/prompts/haystack_prompt.txt: Template for essay generation
- src/steps/step5.py: Main script that uses this client
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Haystack imports
from haystack import Pipeline, component
from haystack.utils import Secret
from haystack.dataclasses import Document
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders import PromptBuilder

# Haystack integrations
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

# Import GeminiClient for the custom embedder
from src.gemini.gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Haystack Component for Gemini Embeddings


@component
class GeminiTextEmbedder:
    """
    Custom Haystack component to generate text embeddings using the Gemini API via GeminiClient.
    """

    def __init__(self, client: Optional[GeminiClient] = None, model_name: Optional[str] = None, task_type: Optional[str] = None):
        """
        Initializes the GeminiTextEmbedder.

        Args:
            client: An initialized GeminiClient instance (optional, will be created if None).
            model_name: The specific Gemini embedding model name (optional, defaults from client/env).
            task_type: The task type for embedding (optional, defaults from client/env).
        """
        self.client = client or GeminiClient()
        # Use provided model/task or let GeminiClient use its defaults from env
        self.model_name = model_name or self.client.embedding_model
        self.task_type = task_type or self.client.default_task_type
        logger.info(
            f"GeminiTextEmbedder initialized with model {self.model_name}, task {self.task_type}")

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        Generates the embedding for the input text.

        Args:
            text: The text to embed.

        Returns:
            A dictionary with the key "embedding" containing the generated embedding list.
        """
        if not text or not isinstance(text, str):
            logger.warning("GeminiTextEmbedder received invalid text input.")
            # Return zero vector of appropriate dimension as fallback? Or raise error?
            # Returning empty list might cause issues downstream. Let's return None and handle upstream if needed.
            # For now, let's return an empty dict which might signal failure better in Haystack.
            # Or maybe just return an empty list, but log it clearly.
            # Let's try returning an empty embedding list
            logger.warning(
                "Returning empty embedding list due to invalid input.")
            return {"embedding": []}  # Return empty list

        embedding = self.client.generate_embedding(
            text=text,
            task_type=self.task_type
            # Let generate_embedding handle retries internally
        )

        if embedding is None:
            logger.error(
                f"Failed to generate embedding for text: {text[:100]}...")
            return {"embedding": []}  # Return empty list on failure

        return {"embedding": embedding}


def get_document_store() -> PgvectorDocumentStore:
    """
    Initialize and return the PgvectorDocumentStore to connect to your 
    PostgreSQL database with pgvector extension.

    Returns:
        PgvectorDocumentStore: Configured document store instance
    """
    # Get connection string from environment variable
    pg_conn_str = os.getenv(
        "PG_CONN_STR", "postgresql://postgres:postgres@postgres:5432/reader_db")

    # Initialize and return document store
    document_store = PgvectorDocumentStore(
        connection_string=pg_conn_str,
        table_name="articles",  # Table containing articles with embedding column
        embedding_dimension=768,  # Dimension of text-embedding-004 embeddings
        vector_function="cosine_similarity",  # Use cosine similarity for search
        recreate_table=False,  # Don't recreate the table, assume it exists
        embedding_field="embedding"  # Name of the vector column in articles table
    )

    logger.info("Initialized PgvectorDocumentStore")
    return document_store


def get_text_embedder() -> GeminiTextEmbedder:
    """
    Initialize and return the custom GeminiTextEmbedder.

    Returns:
        GeminiTextEmbedder: Configured text embedder instance
    """
    # Initialize the custom embedder (it will use env vars via GeminiClient)
    embedder = GeminiTextEmbedder()

    logger.info(f"Initialized GeminiTextEmbedder")
    return embedder


def get_embedding_retriever(document_store: PgvectorDocumentStore) -> PgvectorEmbeddingRetriever:
    """
    Initialize and return a retriever for the specified document store.

    Args:
        document_store: Initialized PgvectorDocumentStore

    Returns:
        PgvectorEmbeddingRetriever: Configured retriever instance
    """
    # Initialize the retriever with the document store
    retriever = PgvectorEmbeddingRetriever(
        document_store=document_store,
        top_k=50  # Retrieve top 50 documents initially
    )

    logger.info("Initialized PgvectorEmbeddingRetriever")
    return retriever


def get_ranker() -> TransformersSimilarityRanker:
    """
    Initialize and return a ranker for reordering retrieved documents.

    Returns:
        TransformersSimilarityRanker: Configured ranker instance
    """
    # Initialize the ranker
    ranker = TransformersSimilarityRanker(
        # Good general-purpose reranking model
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k=20  # Return top 20 documents after ranking
    )

    logger.info("Initialized TransformersSimilarityRanker")
    return ranker


def get_prompt_builder(template_path: str = "src/prompts/haystack_prompt.txt") -> PromptBuilder:
    """
    Load a prompt template from the specified path and initialize a PromptBuilder.

    Args:
        template_path: Path to the prompt template file

    Returns:
        PromptBuilder: Configured prompt builder instance
    """
    try:
        # Load the template from file
        template_file = Path(template_path)
        if not template_file.exists():
            logger.error(f"Prompt template file not found: {template_path}")
            raise FileNotFoundError(
                f"Prompt template file not found: {template_path}")

        template = template_file.read_text(encoding='utf-8')

        # Initialize the prompt builder
        prompt_builder = PromptBuilder(template=template)

        logger.info(
            f"Initialized PromptBuilder with template from: {template_path}")
        return prompt_builder
    except Exception as e:
        logger.error(f"Error initializing PromptBuilder: {e}")
        raise


def get_gemini_generator() -> GoogleAIGeminiGenerator:
    """
    Initialize and return a GoogleAIGeminiGenerator for text generation.

    Returns:
        GoogleAIGeminiGenerator: Configured generator instance
    """
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Get model name from environment or use default
    model_name = os.getenv("GEMINI_FLASH_THINKING_MODEL",
                           "models/gemini-2.0-flash-thinking-exp-01-21")

    # Initialize the generator
    generator = GoogleAIGeminiGenerator(
        api_key=api_key,
        model=model_name,
        generation_kwargs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 8192
        }
    )

    logger.info(
        f"Initialized GoogleAIGeminiGenerator with model: {model_name}")
    return generator


def build_retrieval_pipeline(embedder, retriever) -> Pipeline:
    """
    Build and return a Haystack pipeline for document retrieval.

    Args:
        embedder: Initialized text embedder (GeminiTextEmbedder)
        retriever: Initialized document retriever

    Returns:
        Pipeline: Configured Haystack pipeline
    """
    pipeline = Pipeline()

    # Add components
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)

    # Connect components
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    logger.info("Built retrieval pipeline")
    return pipeline


def build_retrieval_ranking_pipeline(embedder, retriever, ranker) -> Pipeline:
    """
    Build and return a Haystack pipeline for document retrieval with ranking.

    Args:
        embedder: Initialized text embedder (GeminiTextEmbedder)
        retriever: Initialized document retriever
        ranker: Initialized document ranker

    Returns:
        Pipeline: Configured Haystack pipeline
    """
    pipeline = Pipeline()

    # Add components
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("ranker", ranker)

    # Connect components
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")

    logger.info("Built retrieval and ranking pipeline")
    return pipeline


def run_article_retrieval(query_text: str) -> List[Document]:
    """
    Run the retrieval pipeline to get relevant documents for a query.

    Args:
        query_text: The query text to search for

    Returns:
        List[Document]: Retrieved documents
    """
    try:
        # Initialize components
        document_store = get_document_store()
        embedder = get_text_embedder()
        retriever = get_embedding_retriever(document_store)

        # Build pipeline
        pipeline = build_retrieval_pipeline(embedder, retriever)

        # Run pipeline
        result = pipeline.run({"embedder": {"text": query_text}})

        # Return documents
        documents = result["retriever"]["documents"]
        logger.info(f"Retrieved {len(documents)} documents for query")
        return documents

    except Exception as e:
        logger.error(f"Error in article retrieval: {e}", exc_info=True)
        return []


def run_article_retrieval_and_ranking(query_text: str) -> List[Document]:
    """
    Run the retrieval and ranking pipeline to get relevant documents for a query.

    Args:
        query_text: The query text to search for

    Returns:
        List[Document]: Retrieved and ranked documents
    """
    try:
        # Initialize components
        document_store = get_document_store()
        embedder = get_text_embedder()
        retriever = get_embedding_retriever(document_store)
        ranker = get_ranker()

        # Build pipeline
        pipeline = build_retrieval_ranking_pipeline(
            embedder, retriever, ranker)

        # Run pipeline
        result = pipeline.run({
            "embedder": {"text": query_text},
            "ranker": {"query": query_text}
        })

        # Return documents
        documents = result["ranker"]["documents"]
        logger.info(
            f"Retrieved and ranked {len(documents)} documents for query")
        return documents

    except Exception as e:
        logger.error(
            f"Error in article retrieval and ranking: {e}", exc_info=True)
        return []
