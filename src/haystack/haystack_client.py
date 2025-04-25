"""
haystack_client.py - Haystack 2.x RAG Pipeline Client (Refactored)

This module provides an interface primarily focused on using Haystack 2.x components
(target: haystack-ai==2.13.1) for **ranking** retrieved documents and potentially
preparing prompts for essay generation. It leverages the `google-ai-haystack` integration
package (target: >=5.1.0) for the Gemini generator.

It assumes **retrieval** is handled externally (e.g., via ReaderDBClient) based on custom logic.

Exported functions:
- get_ranker(): Returns configured Ranker component (TransformersSimilarityRanker).
- get_prompt_builder(template_path): Returns PromptBuilder component.
- get_gemini_generator(): Returns GoogleAIGeminiGenerator component.
- run_article_retrieval_and_ranking(query_text, article_ids): Implements custom retrieval (via ReaderDBClient) and ranking (via Haystack Ranker).

Related files:
- src/database/reader_db_client.py: Database client, responsible for retrieval.
- src/gemini/gemini_client.py: Gemini API client (potentially used alongside or instead of GoogleAIGeminiGenerator).
- src/prompts/haystack_prompt.txt: Template for essay generation.
- src/steps/step5.py: Main script that uses this client.
- requirements.txt: Specifies haystack-ai and google-ai-haystack versions.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np  # Added for embedding averaging

# Haystack imports (corrected for Haystack 2.x)
# Document location confirmed in 2.x
from haystack.dataclasses import Document
# Component locations confirmed in 2.x
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.builders.prompt_builder import PromptBuilder  # More specific path

# Haystack integrations (Assuming path is correct for google-ai-haystack >= 5.1.0)
# Linter will verify this
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator

# Import ReaderDBClient for custom retrieval
from src.database.reader_db_client import ReaderDBClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Removed GeminiTextEmbedder custom component as it's no longer used


# Removed get_document_store function


# Removed get_text_embedder function


# Removed get_embedding_retriever function


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
    (Kept for potential use in Step 5, though GeminiClient might be used directly)

    Returns:
        GoogleAIGeminiGenerator: Configured generator instance
    """
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Get model name from environment or use default, ensuring SHORT ID is used
    model_full_name = os.getenv("GEMINI_FLASH_THINKING_MODEL",
                                # Default to a known good ID if env var is missing
                                "gemini-1.5-flash-latest")
    # Extract short ID (e.g., gemini-1.5-flash-latest) in case prefix was included in env var
    model_short_id = model_full_name.split('/')[-1]

    # Initialize the generator using the short model ID
    generator = GoogleAIGeminiGenerator(
        api_key=api_key,
        model=model_short_id,  # Pass short ID
        generation_kwargs={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 8192
        }
    )

    logger.info(
        f"Initialized GoogleAIGeminiGenerator with model: {model_short_id}")
    return generator


# Removed build_retrieval_pipeline function


# Removed build_retrieval_ranking_pipeline function


# Removed run_article_retrieval function


def run_article_retrieval_and_ranking(query_text: str, article_ids: List[int]) -> List[Document]:
    """
    Implements the new custom retrieval and ranking strategy:
    1. Fetches embeddings for the given article_ids from the database.
    2. Calculates the average embedding for the group.
    3. Uses the average embedding to find similar articles via ReaderDBClient.
    4. Converts the results to Haystack Document objects.
    5. Ranks the resulting documents using TransformersSimilarityRanker based on the query_text.

    Args:
        query_text: The original query text (e.g., group rationale) used for ranking relevance.
        article_ids: List of article IDs forming the current group, used to calculate the average embedding.

    Returns:
        List[Document]: Retrieved and ranked documents.
    """
    db_client = None
    try:
        logger.info(
            f"Starting custom retrieval and ranking for {len(article_ids)} articles.")

        # 1. & 2. Fetch and Average Group Embeddings
        db_client = ReaderDBClient()  # Initialize DB client
        logger.debug(f"Fetching embeddings for article IDs: {article_ids}")
        # Assuming get_embeddings_for_articles returns a list of dicts like [{'article_id': id, 'embedding': [..]}, ...]
        # Use the newly added method from ReaderDBClient
        group_embeddings_data = db_client.get_embeddings_for_articles(
            article_ids)

        if not group_embeddings_data:
            logger.warning(
                f"No embeddings found for article IDs: {article_ids}. Cannot perform retrieval.")
            return []

        # Safely extract valid embeddings, checking type and iterating over dictionary values
        valid_embeddings = []
        # group_embeddings_data is a Dict[int, Dict[str, Any]]
        for data in group_embeddings_data.values():  # Iterate over the dictionary values
            embedding = data.get('embedding') if isinstance(
                data, dict) else None
            # Check existence and non-emptiness explicitly
            if embedding is not None and len(embedding) > 0:
                valid_embeddings.append(embedding)
            else:
                # Handle cases where data might not be a dict or embedding is invalid
                if isinstance(data, dict):
                    # Data is a dict, but embedding is invalid/missing
                    logger.warning(
                        # Added note about missing ID
                        f"Skipping invalid or missing embedding data for article {data.get('article_id', 'UNKNOWN - data has no ID')}: type={type(embedding)}")
                else:
                    # Data is not a dict, log its type and value
                    logger.warning(
                        # Log first 100 chars
                        f"Skipping unexpected item in group_embeddings_data values. Expected dict, got {type(data)}: {str(data)[:100]}")

        if not valid_embeddings:
            logger.warning(
                f"No *valid* embeddings found for article IDs: {article_ids} after filtering. Cannot perform retrieval.")
            return []

        logger.info(
            f"Found {len(valid_embeddings)} valid embeddings for averaging.")
        # Calculate average embedding
        average_embedding = np.mean(valid_embeddings, axis=0).tolist()
        logger.debug(
            f"Calculated average embedding (first 3 dims): {average_embedding[:3]}")

        # 3. Perform Similarity Search using DB Client
        # find_similar_articles should query embeddings table and JOIN articles
        logger.info("Performing similarity search using average embedding...")
        retrieved_articles_data = db_client.find_similar_articles(
            embedding=average_embedding, limit=50)  # Use 'limit' instead of 'top_k'

        if not retrieved_articles_data:
            logger.info(
                "No similar articles found via custom database retrieval.")
            return []

        logger.info(
            f"Retrieved {len(retrieved_articles_data)} candidate articles from DB.")

        # 4. Convert to Haystack Documents, filtering out ERROR articles
        retrieved_haystack_docs: List[Document] = []
        skipped_error_count = 0
        for item in retrieved_articles_data:
            article_content = item.get('content', '')
            # Skip articles marked as ERROR
            if article_content == 'ERROR':
                skipped_error_count += 1
                continue

            # Construct meta, carefully handling missing keys
            meta_data = {
                'id': item.get('id'),
                'title': item.get('title'),
                'domain': item.get('domain'),
                'pub_date': str(item.get('pub_date')) if item.get('pub_date') else None,
                'frame_phrases': item.get('frame_phrases'),
                # Add other relevant metadata from articles table if needed
            }
            # Create Document, ensuring content is a string
            doc = Document(
                content=str(article_content),  # Use the variable
                meta=meta_data,
                score=item.get('similarity')  # Similarity score from DB query
            )
            retrieved_haystack_docs.append(doc)

        if skipped_error_count > 0:
            logger.info(
                f"Skipped {skipped_error_count} articles with 'ERROR' content.")

        logger.debug(
            f"Converted {len(retrieved_haystack_docs)} valid results to Haystack Documents.")

        # 5. Rank documents using Haystack Ranker
        ranker = get_ranker()
        logger.info("Warming up the ranker...")
        ranker.warm_up()
        logger.info(
            f"Ranking {len(retrieved_haystack_docs)} documents with query: '{query_text[:50]}...'")
        # The ranker component's run method expects kwargs matching its input sockets
        ranking_result = ranker.run(
            query=query_text, documents=retrieved_haystack_docs)

        # Extract ranked documents
        ranked_documents = ranking_result.get('documents', [])

        logger.info(
            f"Ranking complete. Returning {len(ranked_documents)} documents.")
        return ranked_documents

    except Exception as e:
        logger.error(
            f"Error in custom article retrieval and ranking: {e}", exc_info=True)
        return []
    finally:
        # Ensure DB client connection is closed/released if it was opened
        if db_client:
            db_client.close()
            logger.debug("Closed ReaderDBClient connection.")
