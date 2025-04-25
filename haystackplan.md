# Haystack Integration Refactoring Plan (Step 5)

## 1. Problem Statement

The current implementation in `step5.py` attempts to use Haystack's `PgvectorDocumentStore` and `PgvectorEmbeddingRetriever` to find relevant historical articles for RAG context. However, these components expect the article content and its vector embedding to reside in the _same_ database table (specified by `table_name` in `PgvectorDocumentStore`).

Our database schema (`schema.md`) is normalized:

- `articles` table: Stores article content and metadata.
- `embeddings` table: Stores vector embeddings, linked via `article_id`.

This mismatch leads to errors:

- `UndefinedColumn`: When Haystack queries `articles` for an `embedding` column that isn't configured correctly or doesn't exist there initially.
- `TypeError`: When trying to pass unsupported arguments (`embedding_table`, `embedding_column_name`, etc.) to the `PgvectorDocumentStore` constructor in an attempt to bridge the two tables.
- `KeyError`: When Haystack's internal data converter (`_from_pg_to_haystack_documents`) tries to create `Document` objects from the retrieved rows and expects specific columns (`blob_data`, `blob_meta`) that don't exist in our `articles` table.

Adding columns like `embedding`, `blob_data`, `blob_meta` to the `articles` table is a workaround that modifies our schema solely to fit the default component's assumptions and adds redundancy/unused columns.

## 2. Chosen Strategy: Hybrid Approach

We will adapt our Haystack usage by implementing a hybrid approach:

1.  **Use Haystack for:**
    - Generating query embeddings (`GeminiTextEmbedder` in `haystack_client.py`).
    - Re-ranking retrieved documents (`TransformersSimilarityRanker` in `haystack_client.py`).
2.  **Use Custom Logic (`ReaderDBClient`) for:**
    - Performing the vector similarity search directly against our `embeddings` table.
    - Fetching the corresponding article content and metadata from the `articles` table based on the search results.
    - Constructing Haystack `Document` objects from the fetched data.
3.  **Bypass:** We will no longer use `PgvectorDocumentStore` or `PgvectorEmbeddingRetriever` for the _retrieval_ step within the `run_article_retrieval_and_ranking` function.

This approach respects our existing database schema while leveraging Haystack's strengths for embedding and ranking.

## 3. Implementation Steps

### Step 3.1: Enhance `ReaderDBClient` (`src/database/reader_db_client.py`)

- **Create New Method:** Add a method `find_similar_articles(self, embedding: List[float], top_k: int = 50) -> List[Dict[str, Any]]`.
  - **Functionality:**
    - Takes a query embedding vector and `top_k` as input.
    - Executes a SQL query using `pgvector` operators (`<=>`) against the `embeddings` table.
    - The query should `JOIN` `embeddings` with `articles` on `embeddings.article_id = articles.id`.
    - Select necessary fields from `articles` (e.g., `id`, `title`, `content`, `pub_date`, `domain`) and the similarity score `(1 - (embeddings.embedding <=> %s)) as similarity`.
    - Order the results by similarity (`embeddings.embedding <=> %s ASC`).
    - Limit results by `top_k`.
    - Fetch results and format them into a list of dictionaries. Each dictionary should represent a retrieved article and include its content, metadata, and similarity score.

### Step 3.2: Refactor `HaystackClient` (`src/haystack/haystack_client.py`)

- **Modify `run_article_retrieval_and_ranking`:**
  - Keep the initialization of `GeminiTextEmbedder` (`embedder`) and `TransformersSimilarityRanker` (`ranker`).
  - Keep the step where the query embedding is generated: `query_embedding = embedder.run(text=query_text)['embedding']`.
  - **Remove** the initialization of `PgvectorDocumentStore` and `PgvectorEmbeddingRetriever` within this function.
  - **Remove** the construction of the Haystack `Pipeline` that uses the retriever.
  - Instantiate `ReaderDBClient`: `db_client = ReaderDBClient()`.
  - Call the new method: `retrieved_articles_data = db_client.find_similar_articles(embedding=query_embedding, top_k=50)`. (Use a `top_k` sufficient for the ranker, e.g., 50).
  - **Convert DB Results to Haystack Documents:** Iterate through `retrieved_articles_data`:
    - For each dictionary, create a Haystack `Document` object:
      - `content`: Map from the `content` field.
      - `meta`: Populate with other relevant fields (`id`, `title`, `pub_date`, `domain`, etc.).
      - `score`: Assign the `similarity` score from the database query result.
    - Store these `Document` objects in a list (e.g., `retrieved_haystack_docs`).
  - Call the ranker: `ranking_result = ranker.run(query=query_text, documents=retrieved_haystack_docs)`.
  - Return the ranked documents: `return ranking_result['documents']`.
  - Add appropriate error handling around the database call and document conversion.
  - Close the `ReaderDBClient` connection if opened within the function scope (or ensure pool management handles it).
- **Simplify/Comment `get_document_store`:**
  - Revert its configuration to the simplest form (pointing to `articles`, assuming `embedding` column exists).
  - Add comments explaining that this function is **not** used by the `run_article_retrieval_and_ranking` function for retrieval anymore but might be used for writing documents via Haystack elsewhere (if applicable).
- **Cleanup (Optional):** Remove unused functions if they are no longer called (e.g., `get_embedding_retriever`, `build_retrieval_pipeline`, `build_retrieval_ranking_pipeline`, `run_article_retrieval`).

### Step 3.3: Review `Step 5` (`src/steps/step5.py`)

- Ensure the call signature used for `run_article_retrieval_and_ranking` matches the refactored function in `haystack_client.py`.
- Confirm that the code correctly processes the list of Haystack `Document` objects returned by the refactored function. (Likely no changes needed here).

## 4. Considerations & Follow-up

- **Schema Redundancy:** The `articles.embedding`, `articles.blob_data`, and `articles.blob_meta` columns are now technically unnecessary for _this specific retrieval flow_. They can potentially be removed in a future cleanup phase **if and only if** no other process depends on them.
- **Synchronization Logic:** The logic added to `ReaderDBClient.insert_embedding` and `batch_insert_embeddings` to update `articles.embedding` could also be removed _if_ the `articles.embedding` column is removed later. For now, keep the sync logic.
- **Migration Script:** The `migrate_embeddings.py` script remains useful for ensuring the `articles.embedding` column is populated if needed for other reasons or if we revisit Haystack retrieval in the future.
- **Performance:** Monitor the performance of the direct database query compared to the previous Haystack retriever approach.
- **Error Handling:** Implement robust error handling in the new database query method and the refactored `haystack_client.py` function.

This plan focuses on adapting our code to work reliably with Haystack where it fits our use case, rather than forcing schema changes to accommodate default component limitations.
