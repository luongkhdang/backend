# Step 3 Implementation Plan (part1.md)

This plan outlines the implementation steps for `src/steps/step3.py`, based on the objectives in `plan1.md`.

**Overall Goal:** Process recent, unprocessed articles daily to extract entities, store per-article findings, and update basic global entity statistics.

**Core Modules Used:**

- `src.database.reader_db_client.ReaderDBClient`: For all database operations.
- `src.gemini.gemini_client.GeminiClient`: For calls to the Gemini API (entity extraction).
- `src.utils.rate_limit.RateLimiter`: Used internally by `GeminiClient`.

**Implementation Steps:**

1.  **Setup and Initialization (`step3.py`)**

    - Create the file `src/steps/step3.py`.
    - Add necessary imports: `logging`, `os`, `json`, `datetime`, `ReaderDBClient`, `GeminiClient`.
    - Define the main function `run() -> Dict[str, Any]`.
    - Initialize `ReaderDBClient` and `GeminiClient` within `run()` or helper functions.
    - Define constants: `ARTICLE_LIMIT = 2000`, `DAYS_LOOKBACK = 2`, Tier thresholds (`TIER0_COUNT = 150`, `TIER1_COUNT = 350`), scoring weights (`CLUSTER_HOTNESS_WEIGHT = 0.65`, `DOMAIN_GOODNESS_WEIGHT = 0.35`).
    - **Tier-to-Model mapping:** Define a mapping dictionary:
      ```python
      TIER_MODEL_MAP = {
          0: 'models/gemini-2.0-flash-thinking-exp-01-21',  # Highest quality for top tier
          1: 'models/gemini-2.0-flash-exp',                 # Mid-tier
          2: 'models/gemini-2.0-flash'                      # Base tier
      }
      FALLBACK_MODEL = 'models/gemini-2.0-flash-lite'
      ```

2.  **Domain Goodness Scores (Helper Function in `step3.py`)**

    - Define a helper function `_get_domain_goodness_scores(db_client: ReaderDBClient) -> Dict[str, float]`.
    - **Schema Update:** Create `calculated_domain_goodness` table if it doesn't exist:
      ```sql
      CREATE TABLE IF NOT EXISTS calculated_domain_goodness (
          domain TEXT PRIMARY KEY,
          domain_goodness_score FLOAT NOT NULL,
          calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );
      ```
    - **New Function:** Implement `db_client.get_all_domain_goodness_scores()` in `reader_db_client.py` (and optionally in `modules/domains.py`):
      ```python
      def get_all_domain_goodness_scores(self) -> Dict[str, float]:
          """Get all domain goodness scores."""
          try:
              with self.get_connection() as conn:
                  cursor = conn.cursor()
                  cursor.execute("SELECT domain, domain_goodness_score FROM calculated_domain_goodness")
                  return {row[0]: row[1] for row in cursor.fetchall()}
          except Exception as e:
              logger.error(f"Error fetching domain goodness scores: {e}")
              return {}
      ```
    - **Initial Fallback:** Return a `defaultdict(lambda: 0.5)` if table doesn't exist.

3.  **Fetch and Prioritize Articles (Helper Function in `step3.py`)**

    - Define `_prioritize_articles(db_client: ReaderDBClient, domain_scores: Dict[str, float]) -> List[Dict[str, Any]]`.
    - **New Function:** Implement `db_client.get_recent_unprocessed_articles(days, limit)` in `reader_db_client.py` and `modules/articles.py`:
      ```python
      def get_recent_unprocessed_articles(conn, days: int = 2, limit: int = 2000) -> List[Dict[str, Any]]:
          """
          Get recent unprocessed articles.
          Args:
              conn: Database connection
              days: Number of days to look back
              limit: Maximum number of articles to return
          Returns:
              List of article dictionaries
          """
          try:
              cursor = conn.cursor()
              cursor.execute("""
                  SELECT id, domain, cluster_id, content
                  FROM articles
                  WHERE is_processed = FALSE
                  AND created_at >= (CURRENT_DATE - INTERVAL %s DAY)
                  ORDER BY created_at DESC
                  LIMIT %s
              """, (days, limit))

              columns = [desc[0] for desc in cursor.description]
              articles = [dict(zip(columns, row)) for row in cursor.fetchall()]
              cursor.close()
              return articles
          except Exception as e:
              logger.error(f"Error fetching recent unprocessed articles: {e}")
              return []
      ```
    - Fetch cluster hotness scores: Call `db_client.get_all_clusters()` and create a `cluster_id -> hotness_score` map (defaulting to 0.0 if `hotness_score` is missing).
    - Iterate through fetched articles:
      - Get `domain_goodness_score` from `domain_scores` (default 0.0 if missing).
      - Get `cluster_hotness_score` from the cluster map.
      - Normalize scores (0-1 range, handle potential division by zero if max score in batch is 0).
      - Calculate `combined_priority_score` using defined weights.
    - Sort articles by `combined_priority_score` (descending).
    - Assign `priority_rank` and `processing_tier` based on rank and tier counts.
    - Return the list of article dictionaries, now including scoring and tier information.

4.  **Extract Entities via API (Helper Function in `step3.py`)**

    - Define `_extract_entities(gemini_client: GeminiClient, articles: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]`.
    - **Model Selection Enhancement:** Pass the processing tier to the Gemini client and use it to select the appropriate model:

      ```python
      # Use the tier-to-model mapping in step3.py
      model_name = TIER_MODEL_MAP.get(article['processing_tier'], TIER_MODEL_MAP[2])  # Default to tier 2 model

      # Then when calling gemini_client
      response = gemini_client.generate_text_with_prompt(
          article_content=article['content'],
          processing_tier=article['processing_tier'],
          model_override=model_name  # Add this parameter to gemini_client if needed
      )
      ```

    - Initialize an empty dictionary `results = {}` to store `article_id -> list_of_extracted_entities`.
    - Iterate through the prioritized `articles` list.
    - For each `article`:
      - Call `gemini_client.generate_text_with_prompt(article_content=article['content'], processing_tier=article['processing_tier'])`.
        - **Note:** Confirm/Update `gemini_client` to correctly use `processing_tier` for model selection per `plan1.md`. If not yet implemented, proceed with current logic and add a TODO.
      - Handle the response:
        - If successful (returns a string): Try parsing the string as JSON.
          - If JSON parsing succeeds: Store `results[article['id']] = parsed_json['extracted_entities']`.
          - If JSON parsing fails: Log error, store `results[article['id']] = {'error': 'JSON parsing failed', 'raw_response': response_string}`.
        - If unsuccessful (returns None): Log error, store `results[article['id']] = {'error': 'API call failed'}`.
      - Include error handling (e.g., `try...except` blocks for API calls and JSON parsing).
    - Return the `results` dictionary.

5.  **Store Entity Data and Update Status (Helper Function in `step3.py`)**

    - Define `_store_results(db_client: ReaderDBClient, entity_results: Dict[int, List[Dict[str, Any]]]) -> Dict[str, int]`.
    - **New Function:** Implement `db_client.find_or_create_entity(name, entity_type)` in `reader_db_client.py` and `modules/entities.py`:
      ```python
      def find_or_create_entity(conn, name: str, entity_type: str) -> Optional[int]:
          """
          Find an entity by name or create it if it doesn't exist.
          Args:
              conn: Database connection
              name: Entity name
              entity_type: Entity type
          Returns:
              Entity ID or None if error
          """
          try:
              cursor = conn.cursor()
              cursor.execute("""
                  INSERT INTO entities (name, entity_type)
                  VALUES (%s, %s)
                  ON CONFLICT (name) DO UPDATE
                  SET entity_type = EXCLUDED.entity_type,
                      updated_at = CURRENT_TIMESTAMP
                  RETURNING id;
              """, (name, entity_type))

              entity_id = cursor.fetchone()[0]
              conn.commit()
              cursor.close()
              return entity_id
          except Exception as e:
              logger.error(f"Error finding/creating entity {name}: {e}")
              conn.rollback()
              return None
      ```
    - **New Function:** Implement `db_client.increment_entity_mentions(entity_id, count)` in `reader_db_client.py` and `modules/entities.py`:
      ```python
      def increment_entity_mentions(conn, entity_id: int, count: int = 1) -> bool:
          """
          Increment the mentions count for an entity.
          Args:
              conn: Database connection
              entity_id: Entity ID
              count: Amount to increment by
          Returns:
              True if successful, False otherwise
          """
          try:
              cursor = conn.cursor()
              cursor.execute("""
                  UPDATE entities
                  SET mentions = COALESCE(mentions, 0) + %s,
                      updated_at = CURRENT_TIMESTAMP
                  WHERE id = %s
              """, (count, entity_id))

              success = cursor.rowcount > 0
              conn.commit()
              cursor.close()
              return success
          except Exception as e:
              logger.error(f"Error incrementing mentions for entity {entity_id}: {e}")
              conn.rollback()
              return False
      ```
    - **New Function:** Implement `db_client.mark_article_processed(article_id)` in `reader_db_client.py` and `modules/articles.py`:
      ```python
      def mark_article_processed(conn, article_id: int) -> bool:
          """
          Mark an article as processed.
          Args:
              conn: Database connection
              article_id: Article ID
          Returns:
              True if successful, False otherwise
          """
          try:
              cursor = conn.cursor()
              cursor.execute("""
                  UPDATE articles
                  SET is_processed = TRUE,
                      updated_at = CURRENT_TIMESTAMP
                  WHERE id = %s
              """, (article_id,))

              success = cursor.rowcount > 0
              conn.commit()
              cursor.close()
              return success
          except Exception as e:
              logger.error(f"Error marking article {article_id} as processed: {e}")
              conn.rollback()
              return False
      ```
    - Initialize counters: `processed_count = 0`, `entity_links_created = 0`, `errors = 0`.
    - Iterate through `entity_results.items()`, getting `article_id` and `extracted_data`.
    - If `extracted_data` indicates an error (e.g., contains an 'error' key), increment `errors` and continue to the next article.
    - If `extracted_data` is a list of entities:
      - Iterate through each `entity` dict in the list.
      - **Action:** Call a **new** function `db_client.find_or_create_entity(name=entity['entity_name'], type=entity['entity_type'])` -> returns `entity_id`.
        - _(Needs implementation in `src/database/modules/entities.py` and `reader_db_client.py`. Should handle `INSERT...ON CONFLICT...RETURNING id`)._
      - **Action:** Call `db_client.link_article_entity(article_id, entity_id, mention_count=entity['mention_count_article'])`. Check return value for success/failure.
      - **Action:** Call a **new** function `db_client.increment_entity_mentions(entity_id, count=entity['mention_count_article'])`.
        - _(Needs implementation in `src/database/modules/entities.py` and `reader_db_client.py`. Updates the global `mentions` count in the `entities` table)._
        - Increment `entity_links_created` based on success.
      - **Action:** After processing all entities for an article, call a **new** function `db_client.mark_article_processed(article_id)`.
        - _(Needs implementation in `src/database/modules/articles.py` and `reader_db_client.py`. Updates the `is_processed` flag or status)._
      - Increment `processed_count`.
    - Return summary counters: `{'processed': processed_count, 'links': entity_links_created, 'errors': errors}`.

6.  **Main `run()` Function Orchestration (`step3.py`)**

    - Implement the `run()` function.
    - Call `_get_domain_goodness_scores()`.
    - Call `_prioritize_articles()`.
    - Call `_extract_entities()`.
    - Call `_store_results()`.
    - Log summary information.
    - Return a final status dictionary combining results from helper functions.

7.  **Integrate into Main Pipeline (`src/main.py`)**
    - Import `run as run_step3` from `src.steps.step3`.
    - Add a section after Step 2 to call `run_step3`, potentially guarded by an environment variable (`RUN_STEP3`).
    - Log the status returned by `run_step3`.

**Future Enhancements (For Next Iteration):**

1. **Global Influence Score Calculation**

   - Current implementation only updates mention counts via `increment_entity_mentions()`
   - Future: Implement the full scoring algorithm from `plan1.md` step 5:
     ```python
     def calculate_entity_influence_score(conn, entity_id: int) -> float:
         """
         Calculate comprehensive influence score for an entity based on:
         - Total mentions across articles
         - Proportion of influential context mentions
         - Source domain quality
         - Associated cluster hotness
         - Recency of mentions
         - Entity type weighting
         """
         # Implementation to follow in next iteration
     ```

2. **Domain Goodness Calculation Logic**

   - Currently assumed to be pre-calculated and stored
   - Future: Implement the full calculation logic from `plan1.md` step 1 in a separate module/scheduled job

3. **Storage of Supporting Snippets**
   - Current implementation doesn't handle the `supporting_snippets` from entity extraction
   - Future: Add new table and functionality to store these for UI display:
     ```sql
     CREATE TABLE IF NOT EXISTS entity_snippets (
         id SERIAL PRIMARY KEY,
         entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
         article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
         snippet TEXT NOT NULL,
         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
     );
     ```

**Schema and Function Verification Status:**

| Component                           | Status     | Notes                                                                |
| ----------------------------------- | ---------- | -------------------------------------------------------------------- |
| `calculated_domain_goodness` table  | ❌ Missing | Needs to be created                                                  |
| `articles.is_processed` field       | ✅ Exists  | Already in schema.py                                                 |
| `entities` table                    | ✅ Exists  | Contains `name`, `entity_type`, `mentions`, `influence_score`        |
| `article_entities` table            | ✅ Exists  | Contains `article_id`, `entity_id`, `mention_count`                  |
| `get_all_domain_goodness_scores()`  | ❌ Missing | Needs to be implemented                                              |
| `get_recent_unprocessed_articles()` | ❌ Missing | Needs to be implemented                                              |
| `find_or_create_entity()`           | ❌ Missing | Needs to be implemented, partial logic exists in `insert_entity()`   |
| `increment_entity_mentions()`       | ❌ Missing | Needs to be implemented                                              |
| `mark_article_processed()`          | ❌ Missing | Needs to be implemented                                              |
| Gemini Client Tier Selection        | ⚠️ Partial | `generate_text_with_prompt()` exists but needs tier-to-model mapping |
