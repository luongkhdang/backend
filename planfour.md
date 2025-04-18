# Implementation Plan for `step4.py`

**Version:** 1.0
**Date:** 2024-07-26

## 1. Objective

Create `step4.py` to query recently processed articles (published yesterday or today) along with their domain goodness scores and top 10 associated entities (by influence/mentions), and save this consolidated data to a timestamped JSON file in the `src/output/` directory.

## 2. Dependencies

- `src.database.reader_db_client.ReaderDBClient`
- `os` (for path manipulation, checking directory existence)
- `json` (for saving output)
- `datetime` (for generating timestamped filenames and filtering dates)
- `logging`

## 3. File Structure (`src/steps/step4.py`)

- Include a standard file header comment explaining the purpose, exports, and related files, adhering to the project's conventions.
- Define a `run()` function as the main entry point.
- Define helper functions if needed (e.g., for structuring data).

## 4. Core Logic (`run()` function)

1.  Initialize `ReaderDBClient`.
2.  Call a new method `db_client.get_recent_day_processed_articles_with_details()` (to be created) to fetch articles processed yesterday or today. This method should return a list of dictionaries, each containing `article_id`, `title`, `domain`, `goodness_score`, `pub_date`, and `cluster_id`.
3.  Initialize an empty list `output_data`.
4.  Iterate through the fetched articles:
    - For each `article_id`, call `db_client.get_top_entities_for_article(article_id, limit=10)` (to be created) to get the names of the top 10 entities.
    - Combine the article details, goodness score, and entity names into a structured dictionary.
    - Append the structured dictionary to `output_data`.
5.  Define the output directory path: `output_dir = "src/output/"`.
6.  Ensure the output directory exists: `os.makedirs(output_dir, exist_ok=True)`.
7.  Generate a timestamped output filename: `timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")`, `filepath = os.path.join(output_dir, f"step4_output_{timestamp}.json")`.
8.  Write the `output_data` list to the JSON file: `with open(filepath, 'w') as f: json.dump(output_data, f, indent=2, default=str)` (use `default=str` for `datetime` objects).
9.  Close the `ReaderDBClient` connection using a `try...finally` block or context manager if `ReaderDBClient` supports it.
10. Log success, including the number of articles processed and the output file path.
11. Return a status dictionary: `{"success": True, "articles_processed": len(output_data), "output_file": filepath}`.

## 5. Database Client Modifications (`reader_db_client.py` and modules)

- **In `src/database/modules/articles.py`:**
  - Create `get_recent_day_processed_articles_with_details(conn, limit=None)`:
    - Query `articles` (`a`).
    - Filter: `a.processed_at IS NOT NULL` AND `a.pub_date >= CURRENT_DATE - INTERVAL '1 day'`.
    - Join with `domain_statistics` (`ds`) on `a.domain = ds.domain` (use `LEFT JOIN` to include articles even if domain stats are missing).
    - Select: `a.id`, `a.title`, `a.domain`, `ds.goodness_score`, `a.pub_date`, `a.cluster_id`.
    - Apply `limit` if provided.
    - Return list of dictionaries.
- **In `src/database/modules/entities.py`:**
  - Create `get_top_entities_for_article(conn, article_id: int, limit: int = 10)`:
    - Query `article_entities` (`ae`).
    - Join with `entities` (`e`) on `ae.entity_id = e.id`.
    - Filter by `ae.article_id = article_id`.
    - Order by `e.influence_score DESC`, `e.mentions DESC`.
    - Limit by `limit`.
    - Select `e.name`.
    - Return list of entity names.
- **In `src/database/reader_db_client.py`:**
  - Add wrapper methods `get_recent_day_processed_articles_with_details()` and `get_top_entities_for_article()` that get a connection and call the respective functions in the `articles` and `entities` modules.

## 6. Integration (`main.py`)

- Import `run as run_step4` from `src.steps.step4`.
- Add an environment variable check (e.g., `RUN_STEP4=true`).
- Inside the `main` function, after Step 3, add conditional execution block for Step 4:
  ```python
  if os.getenv("RUN_STEP4", "false").lower() == "true":
      logger.info("========= STARTING STEP 4: DATA EXPORT ========")
      try:
          step4_status = run_step4() # Assumes synchronous
          logger.info("Step 4 Summary:")
          logger.debug(json.dumps(step4_status, indent=2))
          if step4_status.get("success", False):
              logger.info(f"Step 4 successful: Exported {step4_status.get('articles_processed', 0)} articles to {step4_status.get('output_file')}")
          else:
              logger.warning(f"Step 4 completed with issues: {step4_status.get('error', 'Unknown error')}")
      except Exception as e:
          logger.error(f"Step 4 failed with error: {e}", exc_info=True)
      finally:
          logger.info("========= STEP 4 COMPLETE ========")
  else:
      logger.info("Skipping Step 4: Data Export (RUN_STEP4 not true)")
  ```

## 7. Docker Configuration

- **`Dockerfile`:** Add `RUN mkdir -p /app/src/output && chmod 777 /app/src/output` before the `CMD` or `ENTRYPOINT`.
- **`docker-compose.yml`:**
  - Add `RUN_STEP4: ${RUN_STEP4:-false}` environment variable to the `article-transfer` service.
  - _Optional:_ Add a volume mount for `src/output` to persist output on the host:
    ```yaml
    services:
      article-transfer:
        # ... other config ...
        volumes:
          # ... existing volumes ...
          - ./src/output:/app/src/output
    ```

## 8. Error Handling

- Implement `try...except` blocks around database calls and file I/O within `step4.py`.
- Log errors using the `logging` module, including tracebacks where helpful (`exc_info=True`).
- Ensure `run()` function returns a status dictionary with `success: False` and an `error` message upon failure.
- Ensure `ReaderDBClient` connection is closed reliably using `try...finally`.

## 9. Refinement & Sophistication

- Leverages `ReaderDBClient` for maintainability.
- Uses efficient SQL queries.
- Timestamped output prevents accidental overwrites.
- Conditional execution enhances flexibility.
- Clear separation of concerns is maintained.
