# Step 2 Database Interaction Refactoring Analysis

This document outlines the necessary changes within the `src/steps/step2/` modules to ensure all database interactions are consistently handled via the `ReaderDBClient`, adhering to the project's data access pattern.

## Summary of Required Changes

1.  **`src/steps/step2/database.py`**:

    - **Function**: `batch_update_article_cluster_assignments`
    - **Current Issue**: Implements direct SQL `UPDATE` query using `unnest` and manages connection/cursor.
    - **Required Change**: Refactor the function to call `reader_client.batch_update_article_clusters(assignments)`. The detailed SQL implementation using `unnest` should reside within the `batch_update_article_clusters` method in `src/database/modules/articles.py`.

2.  **`src/steps/step2/data_fetching.py`**:

    - **Function**: `get_all_embeddings`
    - **Current Issue**: Executes direct SQL `SELECT` query, manages connection/cursor, and performs data type conversions.
    - **Required Change**: Simplify the function to call `reader_client.get_all_embeddings_with_pub_date()`. The SQL query, connection management, and necessary type conversions (e.g., handling string representations of embeddings) should be encapsulated within the client method in `src/database/modules/embeddings.py`.

3.  **`src/steps/step2/interpretation.py`**:
    - **Function**: `extract_cluster_keywords`
    - **Current Issue**: Accepts generic `db_client` parameter; executes direct SQL `SELECT content FROM articles WHERE id = %s`.
    - **Required Change**:
      - Modify signature: `reader_client: ReaderDBClient`.
      - Replace direct SQL: Use `reader_client.get_article_by_id(article_id)` inside the loop to fetch article data and extract the `content`.
      - Add error handling for cases where `get_article_by_id` returns `None` or the content is missing.
    - **Function**: `interpret_clusters`
    - **Current Issue**: Accepts generic `db_client` parameter; executes direct SQL `SELECT published_at FROM articles WHERE id IN %s`.
    - **Required Change**:
      - Modify signature: `reader_client: ReaderDBClient`.
      - Replace direct SQL: Call a _new_ `ReaderDBClient` method, e.g., `get_publication_dates_for_articles(article_ids: List[int]) -> Dict[int, Optional[datetime]]`.
      - Implement this new method (`get_publication_dates_for_articles`) in `src/database/modules/articles.py`.
      - Ensure `extract_cluster_keywords` is called correctly with `reader_client`.

## Implementation Notes

- When moving SQL logic into `ReaderDBClient` methods (specifically within the `src/database/modules/` files), ensure proper connection handling (getting/releasing connections from the pool) and error handling (try/except blocks, rollbacks if necessary).
- The new `get_publication_dates_for_articles` method should efficiently fetch publication dates for a list of article IDs, potentially using `WHERE id = ANY(%s)` or similar constructs.

