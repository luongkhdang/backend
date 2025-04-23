RAG Essay Generation Implementation Plan
This plan outlines the steps to implement the Haystack-based RAG pipeline for generating essays with historical context using the Gemini API and your defined PostgreSQL schema.

Phase 1: Setup & Foundation

Environment Setup:
Configuration:

Modify essays / essay_entities Tables: Use reader_db_client.py (or direct SQL scripts) to apply the recommended ALTER TABLE statements (add group_id, source_article_ids, context columns, etc.).

Phase 2: Core Haystack Pipeline - Article Retrieval (Est. 1 Day)
Embedder Component:

Choose and initialize your text embedder (gemini 'models/text-embedding-004'). Ensure the model matches the one used to generate the embeddings stored in the database.

Retriever Component:

Initialize PgvectorEmbeddingRetriever, linking it to the configured DocumentStore. Set an initial top_k (e.g., 50).

Basic Retrieval Pipeline:

Create a simple Haystack Pipeline.

Add the embedder and retriever components.

Connect embedder.embedding to retriever.query_embedding.

Phase 3: Structured Data Retrieval - Custom Logic
Database Query Implementation (within reader_db_client.py):

Implement methods within reader_db_client.py corresponding to the functions previously listed:

get_key_entities_for_group(article_ids: list[int], top_n=10) -> list[int]

get_related_events(entity_ids: list[int], limit=15) -> list[dict]

get_related_policies(entity_ids: list[int], limit=15) -> list[dict]

get_related_relationships(entity_ids: list[int], limit=20) -> list[dict]

(Optional) Cluster-based filtering methods.

These methods handle the actual SQL execution using the established DB connection.

Result Formatting (within reader_db_client.py or a utility module):

Implement helper methods (potentially within reader_db_client.py or a separate utility) to format the raw dictionary results into concise text strings.

Include logic to parse the metadata JSONB from relationships.

Integration Strategy:
The step5.py script will call methods from reader_db_client.py before running the Haystack pipeline managed by haystack_client.py.

Phase 4: Ranking & Context Assembly
Ranker Component (within haystack_client.py):

Initialize and add a Haystack Ranker to the retrieval pipeline definition in haystack_client.py.

Context Selection Logic (step5.py or haystack_client.py):

Implement Python logic (likely within step5.py or potentially a dedicated function called by it) that:

Gets ranked historical articles from the Haystack pipeline result (managed by haystack_client.py).

Gets formatted structured data summaries by calling reader_db_client.py methods (if using pre-processing/Option B) or from the custom component result (if Option A).

Selects the final top N items.

PromptBuilder Implementation (within haystack_client.py):

Define the prompt template string.

Initialize PromptBuilder within haystack_client.py.

Context Assembly Logic (step5.py or haystack_client.py):

Implement code (likely within step5.py or a function it calls) to gather all context pieces:

Group info (from group.json, loaded by step5.py).

Current article metadata/snippets (calling methods in reader_db_client.py).

Selected historical articles (from Haystack results).

Selected structured summaries.

Format these into the structure expected by the PromptBuilder.

Phase 5: Generation & Output
Generation Strategy:
Full Pipeline Construction / Execution:
step5.py calls a method in haystack_client.py that assembles and runs a Haystack pipeline up to the PromptBuilder. step5.py gets the final prompt from the result, then calls generator.py.generate_essay(prompt).
Group Iteration Script (step5.py):

Loads group.json.

Iterates through each group.

Fetches group info (article_ids, group_rationale).

Calls reader_db_client.py for structured data (if using pre-processing/Option B).

Prepares input data for haystack_client.py.

Runs the generation process using the chosen strategy (Option A or B).

Extracts the generated essay text.

Saving Results (step5.py calling reader_db_client.py):

Construct the data dictionary for the new essays record within step5.py.

Call a method in reader_db_client.py (e.g., save_essay(essay_data: dict)) to execute the INSERT statement.
