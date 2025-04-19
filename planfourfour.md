# Implementation Plan: Enhance Step 4 Article Export (`planfourfour.md`)

**Goal:** Modify `src/steps/step4.py` to export richer article data, including frame phrases, influential entity context, content previews for hot articles, and related entity snippets.

**Current Export Fields:**

- `article_id`
- `title`
- `domain`
- `goodness_score`
- `pub_date`
- `cluster_id`
- `top_entities` (List of entity names)

**Requested Enhancements:**

1.  **Add `frame_phrases`:** Include the `frame_phrases` array for each article.
2.  **Enhance `top_entities`:**
    - Change from `List[str]` to `List[Dict[str, Any]]`.
    - Each dictionary should contain at least `entity_id`, `name`, and `is_influential` (derived from `article_entities.is_influential_context`).
3.  **Add `content_preview`:** If `articles.is_hot` is `True`, add a `content_preview` field containing the first 100 words of `articles.content`.
4.  **Add `influential_snippets`:** If `articles.is_hot` is `True` _and_ an entity in `top_entities` has `is_influential` as `True`, fetch all `entity_snippets.snippet` for that specific `article_id` and `entity_id` and add them as a list under the key `influential_snippets` within that entity's dictionary in the `top_entities` list.

**Implementation Steps:**

1.  **Database Layer (`src/database/modules/` & `src/database/reader_db_client.py`):**

    - **Modify `modules/articles.py`:** Update `get_recent_day_processed_articles_with_details` to select and return `a.is_hot`, `a.content`, and `a.frame_phrases` in addition to the current fields.
    - **Modify `modules/entities.py`:** Create a new function `get_top_entities_with_influence_flag(conn, article_id: int, limit: int = 10) -> List[Dict[str, Any]]`. This should query `article_entities` joined with `entities`, filter by `article_id`, order appropriately (e.g., by `e.influence_score DESC`, `ae.mention_count DESC`), limit the results, and return a list of dictionaries containing `entity_id`, `name`, and `is_influential_context`.
    - **Modify `reader_db_client.py`:**
      - Ensure the signature/docstring for `get_recent_day_processed_articles_with_details` reflects the added return fields (`is_hot`, `content`, `frame_phrases`).
      - Add a new client method `get_top_entities_with_influence_flag` that calls the corresponding function in `entities.py`.
      - Verify that `get_article_entity_snippets(article_id, entity_id)` exists and functions as needed.

2.  **Application Layer (`src/steps/step4.py`):**
    - **Update Header:** Modify the file header comment to reflect the new functionality and dependencies.
    - **Fetch Enhanced Data:**
      - Use the updated `db_client.get_recent_day_processed_articles_with_details()` to fetch articles.
    - **Process Articles:**
      - Inside the loop iterating through fetched articles:
        - Retrieve `is_hot`, `content`, `frame_phrases`, and other necessary fields from the article dictionary.
        - Call the new `db_client.get_top_entities_with_influence_flag(article_id)` to get the top entities with their influence context.
        - Construct the base `article_data` dictionary.
        - Add `frame_phrases` directly to `article_data`.
        - Process the result from `get_top_entities_with_influence_flag` into the desired `top_entities` list structure (list of dicts). Rename the `is_influential_context` key to `is_influential` for clarity in the output.
        - **Conditional Logic for `is_hot`:**
          - If `is_hot` is `True`:
            - Calculate `content_preview`: Take the first 100 words from `content`. Handle potential `None` content gracefully. Add `content_preview` to `article_data`.
            - Iterate through the processed `top_entities` list:
              - For each `entity_dict` where `entity_dict['is_influential']` is `True`:
                - Fetch snippets: `snippets_result = db_client.get_article_entity_snippets(article_id, entity_dict['entity_id'])`.
                - Extract snippet texts: `influential_snippets = [s['snippet'] for s in snippets_result if 'snippet' in s]`.
                - Add `influential_snippets` list to the `entity_dict`.
        - Append the fully constructed `article_data` to `output_data`.
    - **Update Output:** Ensure the final JSON structure matches the new `article_data` format.
    - **Logging:** Update logging statements to be informative about the new data being processed and exported.

**Schema Guidance (`schema.md`):**

- Confirm column names: `articles.is_hot`, `articles.content`, `articles.frame_phrases`.
- Confirm junction table `article_entities` has `article_id`, `entity_id`, `is_influential_context`.
- Confirm table `entity_snippets` has `article_id`, `entity_id`, `snippet`.

**Testing:**

- Run the pipeline with `RUN_STEP4=true`.
- Verify the structure and content of the output JSON file in `src/output/`.
- Check edge cases: articles that are not hot, hot articles with no influential entities, articles with `None` content or `None` frame phrases.
