Haystack 2.x Guide for RAG Essay Generation ProjectThis guide provides an overview of the key Haystack 2.x components and concepts you'll use to build your RAG pipeline, based on information from Haystack documentation and examples.1. Core Concepts (Haystack 2.x)Components: The fundamental building blocks. In 2.x, previous "Nodes" are now generally referred to as "Components". Each performs a specific task (e.g., retrieving, embedding, generating). You'll use components like PgvectorDocumentStore (as a backend), SentenceTransformersTextEmbedder, PgvectorEmbeddingRetriever, Rankers, PromptBuilder, and GoogleAIGeminiGenerator.Pipelines: Directed Acyclic Graphs (DAGs) defining data flow. In 2.x, building pipelines is a two-step process:Add components using pipeline.add_component().Explicitly connect component outputs to inputs using pipeline.connect().2. Document Store (PgvectorDocumentStore)Purpose: Provides the interface to your PostgreSQL database enabled with the pgvector extension. It's not a pipeline component itself in 2.x but is used by components like Retrievers and Writers.Integration: Uses the pgvector-haystack library (pip install pgvector-haystack).Initialization: Requires connection details and configuration matching your database setup.# Within reader_db_client.py or a config module
import os
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from haystack.utils import Secret

# Ensure PG_CONN_STR is set, e.g., "postgresql://user:password@host:port/database"

pg_conn_str = Secret.from_env_var("PG_CONN_STR")

# Initialize the DocumentStore instance

document_store = PgvectorDocumentStore(
connection_string=pg_conn_str,
table_name="articles", # Your table with documents + embeddings
embedding_dimension=768, # Matches your model/schema [cite: 78-89]
vector_function="cosine_similarity", # Or other supported function
recreate_table=False, # Assume table exists
search_strategy="hnsw", # Or other strategy if configured
embedding_field="embedding" # Name of the vector column in 'articles'
)
Handling Embeddings:The search results confirm PgvectorDocumentStore expects content and embeddings in the same table.Recommendation: Stick with the plan to add an embedding VECTOR(768) column to your articles table and populate it. Configure PgvectorDocumentStore to use this table and column.Usage: Pass the initialized document_store instance to components that need it, like PgvectorEmbeddingRetriever.3. Embedding Query (SentenceTransformersTextEmbedder)Purpose: Transforms a single input string (your query) into a vector embedding using a Sentence Transformers model. Use this for the query, not SentenceTransformersDocumentEmbedder (which is for indexing multiple documents).Model Matching: Crucial: Use the exact same Sentence Transformers model name here as was used to generate the embeddings stored in your embeddings table [cite: 78-89] / articles table.Initialization:# Within haystack_client.py
from haystack.components.embedders import SentenceTransformersTextEmbedder

# Use the exact model name matching your stored embeddings

text_embedder = SentenceTransformersTextEmbedder(
model="your-embedding-model-name" # e.g., model="sentence-transformers/all-mpnet-base-v2" # prefix="Represent this sentence for searching relevant passages:" # Add if required by your specific model (e.g., BGE models)
)
text_embedder.warm_up() # Optional: Pre-load the model
Input/Output: Takes text (string), outputs embedding (list of floats).4. Retrieval (PgvectorEmbeddingRetriever)Purpose: A pipeline component that queries a PgvectorDocumentStore to find documents whose embeddings are most similar to an input query embedding.Initialization: Requires the initialized PgvectorDocumentStore instance.# Within haystack_client.py
from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever

# Assuming 'document_store' is your initialized PgvectorDocumentStore instance

retriever = PgvectorEmbeddingRetriever(
document_store=document_store,
top_k=50 # Retrieve more initially for ranking
)
Input/Output: Takes query_embedding (list of floats), outputs documents (list of Haystack Document objects).5. Ranking (Ranker Components)Purpose: Pipeline components that re-order a list of documents based on specific criteria, typically placed after a Retriever.Available Rankers (Examples from Search Results):TransformersSimilarityRanker: Cross-encoder model for relevance (accurate but slower).CohereRanker, JinaRanker, NvidiaRanker: Vendor-specific reranking models.MetaFieldRanker: Sorts by a metadata field value (e.g., pub_date).LostInTheMiddleRanker: Reorders to potentially improve LLM focus on key documents.SentenceTransformersDiversityRanker: Aims for diversity in ranked results.Initialization & Usage:# Within haystack_client.py
from haystack.components.rankers import MetaFieldRanker # Example

# Example: Rank by publication date (assuming 'pub_date' is in meta)

# Note: MetaFieldRanker might require meta values to be parsed correctly (e.g., as dates)

ranker = MetaFieldRanker(meta_field="pub_date", top_k=20, sort_order="descending")

# --- Or using LostInTheMiddleRanker ---

# from haystack.components.rankers import LostInTheMiddleRanker

# ranker = LostInTheMiddleRanker(top_k=20)

# In Pipeline definition:

# pipe.add_component("ranker", ranker)

# pipe.connect("retriever.documents", "ranker.documents")

Input/Output: Takes documents (list), outputs documents (re-ordered list).6. Prompt Engineering (PromptBuilder)Purpose: A pipeline component that constructs the final prompt string using a Jinja2 template and input variables.Initialization: Takes the template string.# Within haystack_client.py
from haystack.components.builders import PromptBuilder

# Example template using Jinja2 syntax

prompt_template = """
Based on the following context, please write an essay addressing the key question.

Historical Articles:
{% for doc in documents %}
Article Title: {{ doc.meta.get('title', 'N/A') }} (Published: {{ doc.meta.get('pub_date', 'N/A') }})
Content Snippet: {{ doc.content | truncate(300) }} {# Example: Limit content length #}

---

{% endfor %}

Related Historical Data:
{% for item in structured_summaries %}

- {{ item }}
  {% else %}
  No additional structured data provided.
  {% endfor %}

Key Question/Rationale: {{ query }}

Essay:
"""

prompt_builder = PromptBuilder(template=prompt_template)
Input/Output: Takes variables matching the template placeholders (e.g., documents, structured_summaries, query) as keyword arguments during pipeline.run(), outputs prompt (string).7. LLM Integration (GoogleAIGeminiGenerator)Purpose: A pipeline component that interacts with the Google Gemini API for text generation.Integration: Uses the google-ai-haystack library (pip install google-ai-haystack).Initialization: Requires the Gemini model name and API key.# Within haystack_client.py OR gemini_client.py (if using directly)
import os
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack.utils import Secret

# Ensure GOOGLE_API_KEY is set

gemini_api_key = Secret.from_env_var("GOOGLE_API_KEY")

gemini_generator = GoogleAIGeminiGenerator(
api_key=gemini_api_key,
model="gemini-1.5-pro" # Specify desired Gemini model # generation_kwargs={"temperature": 0.7, "top_p": 0.9} # Optional LLM params
)
Input/Output: Takes prompt (string), outputs replies (list of strings). Supports multimodal inputs via the parts argument if needed (though your use case seems text-focused).8. Pipelines (Haystack 2.x)Purpose: Orchestrate the flow between components.Construction:# Within haystack_client.py
from haystack import Pipeline

def build_essay_pipeline(text_embedder, retriever, ranker, prompt_builder, generator):
pipe = Pipeline()

    # 1. Add components
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("ranker", ranker)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    # 2. Connect components explicitly
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "ranker.documents")
    pipe.connect("ranker.documents", "prompt_builder.documents")
    # Note: 'query' and 'structured_summaries' for prompt_builder
    # will likely come from the pipeline run input, not other components.
    pipe.connect("prompt_builder.prompt", "generator.prompt")

    return pipe

# Running the pipeline (in step5.py)

# haystack_pipeline = haystack_client.build_essay_pipeline(...)

# result = haystack_pipeline.run({

# "text_embedder": {"text": group_query_text}, # Input for the embedder

# "prompt_builder": { # Inputs specifically for prompt_builder

# "query": group_rationale,

# "structured_summaries": formatted_structured_data

# }

# })

# essay = result["generator"]["replies"][0]

9. Custom Components (Optional)Purpose: Create reusable components for logic not covered by built-in ones (like your structured data retrieval if not handled in step5.py).Requirements (Haystack 2.x):Use the @component decorator on the class.Define a run() method.Define output types using @component.output_types(...) decorator on the run method.Inputs can be defined via run() method arguments or set_input_type(s).This guide provides a starting point based on the documentation search results. Always refer to the latest official Haystack documentation (https://docs.haystack.deepset.ai/) for the most current and detailed information on component parameters and behavior.
