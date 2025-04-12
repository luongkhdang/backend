# Plan for Step 1.6: Embedding Generation

This plan outlines the steps to implement Step 1.6 in `src/steps/step1.py`, which involves generating text embeddings for articles using the Gemini API and storing them in the database.

**Goal:** Generate embeddings for articles with content length suitable for the Gemini `text-embedding-004` model (less than 1450 words / ~2048 tokens) and store them in the `reader-db`.

**Assumptions:**

- `GEMINI_API_KEY` is available as an environment variable in the `.env` file.
- The `google-generativeai` library is listed in `requirements.txt`.

**Steps:**

1.  **Update `src/database/reader_db_client.py`:**

    - Define a new method: `get_articles_needing_embedding(max_word_count: int = 1450) -> List[Dict[str, Any]]`
      - This method should query the `articles` table.
      - It should select `articles.id` and `articles.content`.
      - It should filter articles where:
        - `content` is not NULL, not 'ERROR', and not empty.
        - The article does NOT already have an entry in the `embeddings` table (using `LEFT JOIN embeddings ON articles.id = embeddings.article_id WHERE embeddings.article_id IS NULL`).
      - Fetch the results (`id`, `content`).
      - In Python, filter the fetched results based on word count: `len(article['content'].split()) < max_word_count`.
      - Return a list of dictionaries, each containing `{'id': article_id, 'content': article_content}` for articles meeting all criteria.
      - Add logging for the number of articles found.
    - Verify the existing `insert_embedding(article_id: int, embedding: List[float]) -> Optional[int]` method is suitable for storing the results.

2.  **Implement `src/gemini/gemini_client.py`:**

    - Add the standard file header comment as required by `general.mdc`, outlining purpose, exported functions/classes, and related files.
    - Import necessary libraries: `google.generativeai` (as `genai`), `os`, `dotenv`, `logging`, `typing`, `time`, `random`.
    - Create a `GeminiClient` class:
      - `__init__(self)`:
        - Load `.env` variables using `load_dotenv()`.
        - Retrieve `GEMINI_API_KEY` from `os.getenv()`. Raise an error or log critically if missing.
        - Configure `genai.configure(api_key=...)`.
        - Initialize a logger instance.
        - Store the embedding model name (`text-embedding-004`).
      - `generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT", retries: int = 3, initial_delay: float = 1.0) -> Optional[List[float]]`:
        - Takes text content, task type, and retry parameters.
        - Uses `genai.embed_content` with `model=self.embedding_model`, `content=text`, and `task_type=task_type`.
        - Implement a retry loop with exponential backoff (using `time.sleep` and `random.random`) to handle potential transient API errors or rate limits.
        - Include robust error handling (e.g., `try...except` for API errors) and detailed logging for both success and failure scenarios.
        - Return the embedding list (`result['embedding']`) or `None` if embedding fails after retries.
      - Consider a `close(self)` method if needed, though likely not required for this API client library.

3.  **Update `src/steps/step1.py`:**

    - Import `GeminiClient` from `src.gemini.gemini_client`.
    - Import `ReaderDBClient` type hint if not already present.
    - Modify the `run` function:
      - After Step 1.5 (error reporting / `output_error_articles_json`), add the logic for Step 1.6.
      - Log the start of Step 1.6.
      - Use the _existing_ `reader_db` instance (initialized earlier in `run`) to call `reader_db.get_articles_needing_embedding()`.
      - Check if the returned list of articles is empty. If so, log that no articles require embedding and conclude Step 1.6.
      - If articles are found:
        - Log the number of articles queued for embedding.
        - Initialize `GeminiClient`. Handle potential initialization errors (e.g., missing API key) by logging an error and skipping Step 1.6.
        - Set up `concurrent.futures.ThreadPoolExecutor` with `max_workers`.
        - Define an inner worker function `process_embedding_task(article_data: Dict[str, Any]) -> Tuple[int, bool]`:
          - Takes `{'id': article_id, 'content': ...}`.
          - Calls `gemini_client.generate_embedding(article_data['content'])`.
          - If embedding is successful (not `None`):
            - Calls `reader_db.insert_embedding(article_data['id'], embedding)` using the _persistent_ `reader_db` instance. Check the return value for success.
            - Return `(article_data['id'], True)` if insertion succeeds.
          - Return `(article_data['id'], False)` if embedding generation or DB insertion fails. Log the specific error.
        - Submit all `article_data` items to the executor using `executor.submit(process_embedding_task, article)`.
        - Use `concurrent.futures.as_completed` to process results. Track success/failure counts.
        - Log progress periodically (e.g., every 50 articles).
        - Log the final summary: total articles processed for embedding, number successful, number failed.
      - Log the completion or skipping of Step 1.6.

4.  **Update `requirements.txt`:**

    - Add `google-generativeai` to the file.
    - Ensure `python-dotenv` is present (it should be).

5.  **Update `.env` file:**

    - Ensure the line `GEMINI_API_KEY="YOUR_API_KEY_HERE"` exists and contains a valid API key.

6.  **Testing:**
    - Prepare test data in the `reader-db` (articles with and without embeddings, varying content lengths).
    - Run `src/main.py`.
    - Verify:
      - Correct articles are selected based on content length and existing embeddings.
      - Logs indicate the start, progress, and end of Step 1.6.
      - API calls are made to Gemini (monitor quotas if possible).
      - Embeddings are correctly stored in the `embeddings` table linked to the right `article_id`.
      - Error handling (API errors, DB errors) is logged appropriately.
      - The process completes gracefully even if some embeddings fail.
