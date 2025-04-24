# RAG Essay Generation Implementation Plan

This plan outlines the steps to implement the Haystack-based RAG pipeline for generating essays with historical context using the Gemini API and your defined PostgreSQL schema.

**Phase 1: Setup & Foundation (Est. 0.5 Day)**

1.  **Environment Setup:**
    - **Dependencies:** Add `google-ai-haystack`, `pgvector-haystack`, `sentence-transformers`, and potentially `cohere-haystack` (if using CohereRanker) to `requirements.txt`.
    - **Docker:** Rebuild the base image (`docker build -t article-transfer-base:latest -f base.Dockerfile .`) and the main image (`docker-compose build article-transfer`) to include the new dependencies.
2.  **Configuration:**
    - **`.env` / `docker-compose.yml`:** Ensure `GOOGLE_API_KEY` is correctly set. Add `PG_CONN_STR="postgresql://postgres:postgres@postgres:5432/reader_db"` (adjust if needed) to the `environment` section of the `article-transfer` service in `docker-compose.yml` so Haystack's `PgvectorDocumentStore` can connect. Verify other necessary ENV VARs (like embedding model names) are present.
3.  **Database Schema Update:**
    - **Essay Table:** Review the revised `essays` table proposal in `hayhay.md` (including `group_id`, `source_article_ids`, `model_name`, `generation_settings`, etc.) and apply necessary `ALTER TABLE` commands via a migration script (e.g., `src/database/migrations/001_update_essays_schema.sql`) or by updating `src/database/modules/schema.py`.
    - **Articles Table (Embedding):** Add an `embedding VECTOR(768)` column to the `articles` table. This is required for `PgvectorDocumentStore`. Populate this column for historical articles intended for retrieval.
    - **Migration Runner:** Ensure `db_setup.py` or a dedicated script runs the schema updates.

**Phase 2: Core Haystack Components (Implemented in `src/haystack/haystack_client.py`) (Est. 0.5 Day)**

1.  **Document Store Initialization (`get_document_store`)**: Confirmed implemented. Initializes `PgvectorDocumentStore` pointing to the `articles` table and the `embedding` column.
2.  **Custom Text Embedder (`GeminiTextEmbedder`, `get_text_embedder`)**: Confirmed implemented. Uses a custom Haystack component (`GeminiTextEmbedder`) that wraps `src.gemini.gemini_client.GeminiClient` to generate query embeddings with the model specified in the environment (`models/text-embedding-004`).
3.  **Retriever Initialization (`get_embedding_retriever`)**: Confirmed implemented. Initializes `PgvectorEmbeddingRetriever` using the document store, `top_k=50`.
4.  **Ranker Initialization (`get_ranker`)**: Confirmed implemented. Initializes `TransformersSimilarityRanker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`), `top_k=20`.
5.  **Prompt Builder Initialization (`get_prompt_builder`)**: Confirmed implemented. Initializes `PromptBuilder` using a template file path (default: `src/prompts/haystack_prompt.txt`).
6.  **Generator Initialization (`get_gemini_generator`)**: Confirmed implemented. Initializes `GoogleAIGeminiGenerator` using the API key and model from environment (`models/gemini-2.0-flash-thinking-exp-01-21`).
7.  **Pipeline Definitions (`build_retrieval_pipeline`, `build_retrieval_ranking_pipeline`)**: Confirmed implemented. Defines pipelines connecting embedder -> retriever and embedder -> retriever -> ranker.
8.  **Pipeline Execution Wrappers (`run_article_retrieval`, `run_article_retrieval_and_ranking`)**: Confirmed implemented. Provides functions to easily run the retrieval or retrieval+ranking pipelines.

**Phase 3: Structured Data Retrieval (Implemented in `src/database/modules/haystack_db.py`) (Est. 0.5 Day)**

1.  **Implement DB Query Functions (in `haystack_db.py`)**: Confirmed implemented. Contains standalone functions:
    - `get_key_entities_for_group(conn, article_ids, top_n)`
    - `get_related_events(conn, entity_ids, limit)`
    - `get_related_policies(conn, entity_ids, limit)`
    - `get_related_relationships(conn, entity_ids, limit)`
      These functions take a connection object and parameters, execute SQL, and return raw results.
2.  **Implement Result Formatting Helpers (in `haystack_db.py`)**: Confirmed implemented. Contains private helper functions:
    - `_format_event(event_dict)`
    - `_format_policy(policy_dict)`
    - `_format_relationship(relationship_dict)` (includes parsing metadata for snippets)
3.  **Implement Wrapper Methods (in `ReaderDBClient`)**: Confirmed implemented in `src/database/reader_db_client.py`. Wrapper methods like `get_formatted_related_events`, `get_formatted_related_policies`, etc., handle connection management, call the corresponding functions in `haystack_db.py`, and utilize the formatting helpers to return lists of strings.

**Phase 4: Context Assembly & Prompting (Est. 1 Day)**

1.  **Context Selection Logic** (To be implemented within `src/steps/step5.py`):
    - After getting ranked historical docs from `haystack_client.run_article_retrieval_and_ranking` and formatted structured data summaries from `reader_db_client` (via its wrappers), implement logic to select the top N (e.g., 20) items total. This logic can prioritize item types (e.g., ensure at least 5 historical articles, 5 relationships) or use a simple combined ranking.
2.  **Define Prompt Template** (Create/Populate `src/prompts/haystack_prompt.txt`):
    - Create the detailed prompt structure as discussed in `hayhay.md`, including placeholders like `{group_rationale}`, `{key_questions}`, `{current_articles_summary}`, `{historical_articles}`, `{related_events}`, `{related_policies}`, `{entity_relationships}`. Define the final essay writing instruction clearly.
3.  **Implement Context Assembly Logic** (To be implemented within `src/steps/step5.py`):
    - Create a dedicated function `assemble_final_context(group_data, current_article_details, selected_historical_docs, formatted_structured_data) -> str`.
    - This function fetches all necessary pieces:
      - Group info (`group_rationale`, angles/theories derived from `frame_phrases` if needed) from `group_data`.
      - Current article summaries (metadata + top snippets fetched via `reader_db_client`).
      - Selected historical article content (from `selected_historical_docs`).
      - Formatted structured data strings (events, policies, relationships).
    - Formats everything into a single large string, using the structure defined in `haystack_prompt.txt`. Manage length carefully (summarization/truncation might be needed).

**Phase 5: Orchestration & Output (Est. 1 Day)**

1.  **Modify/Confirm Generator Module (`src/gemini/modules/generator.py`)**: Existing `analyze_articles_with_prompt` seems suitable. Confirm it handles large prompts correctly with the specified model and returns the raw text response.
2.  **Add Gemini Client Method (`src/gemini/gemini_client.py`)**: Add `generate_essay_from_prompt(self, full_prompt_text: str, ...)` method as planned, likely calling `generator.analyze_articles_with_prompt`.
3.  **Implement Essay Saving (`src/database/modules/haystack_db.py` & `ReaderDBClient`)**: Confirmed implemented. `haystack_db.save_essay` handles the SQL INSERT, and `ReaderDBClient.save_essay` acts as the wrapper handling connections.
4.  **Implement Group Iteration Script (`src/steps/step5.py`)**:
    - Create `src/steps/step5.py`.
    - Load `group.json`.
    - Initialize `ReaderDBClient()` and `GeminiClient()`. Instantiate `HaystackClient` components as needed (or create a simple `HaystackClient` class).
    - Loop through each group in `group.json`.
    - Inside loop:
      - Get `article_ids`, `group_rationale`.
      - **Fetch Current Data:** Call `reader_db_client` for current article metadata/snippets.
      - **Identify Key Entities:** Call `reader_db_client.get_key_entities_for_group`.
      - **Fetch Structured Data:** Call `reader_db_client.get_formatted_related_events`, etc.
      - **Fetch Historical Articles:** Call `haystack_client.run_article_retrieval_and_ranking`.
      - **Select Context:** Apply selection logic (Phase 4.1).
      - **Assemble Context:** Call `assemble_final_context` (Phase 4.3).
      - **Generate Essay:** Call `gemini_client.generate_essay_from_prompt`.
      - **Extract & Format:** Process the generator's response (essay text, maybe title).
      - **Save Result:** Prepare `essay_data` dict and call `reader_db_client.save_essay`.

**Phase 6: Testing & Refinement (Ongoing)**

- Test each phase incrementally.
- Validate database queries and formatting.
- Test Haystack pipeline execution and results.
- Analyze generated essay quality and context relevance.
- Refine prompts, ranking, context selection, and summarization strategies based on results.
- Monitor API costs and token usage.

This detailed plan provides a clearer roadmap for implementation, assigning responsibilities to specific modules and outlining the data flow between `step5.py`, `reader_db_client.py`, `haystack_client.py`, and `gemini_client.py`/`generator.py`.

```

```
