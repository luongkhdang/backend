# Implementation Plan: Gemini API Optimization and Frame Phrase Extraction

This plan outlines the steps to optimize the Gemini API response format for token efficiency, incorporate the extraction of narrative `frame_phrases`, and store this new data in the `articles` table.

**Goal:** Reduce API token usage while enriching article analysis with narrative framing information.

**References:**

- `plan9.md`: Initial token optimization analysis.
- `prompt9.md`: Revised Gemini API prompt with shorter keys and `frame_phrases` request.
- `src/steps/step3/__init__.py`: Primary logic for Step 3 processing.
- `src/gemini/gemini_client.py`: API interaction client.
- `src/database/reader_db_client.py` & modules: Database interaction layer.

**Implementation Steps:**

1.  **Database Schema Update (`src/database/modules/schema.py`):**

    - **Action:** Add a `frame_phrases TEXT[] NULL` column to the `articles` table definition within the `initialize_tables` function.
    - **Rationale:** Create storage for the extracted narrative frames. Allow NULL as processing might fail or frames might not be found.
    - **Verification:** Ensure `initialize_tables` runs successfully and the column is added (e.g., via pgAdmin or DB inspection).

2.  **Database Client Update (`src/database/reader_db_client.py` & `src/database/modules/articles.py`):**

    - **Action (articles.py):** Create a new function `update_article_frames_and_mark_processed(conn, article_id: int, frame_phrases: Optional[List[str]]) -> bool`. This function should update the `articles` table, setting `frame_phrases` to the provided list and `extracted_entities` to `TRUE` for the given `article_id`. It should handle `frame_phrases` being `None` gracefully (setting the column to NULL or skipping the update for that column).
    - **Action (reader_db_client.py):** Add a corresponding method `update_article_frames_and_mark_processed` that calls the function created in `articles.py`.
    - **Rationale:** Provide a dedicated mechanism to store the new data and update the processing status atomically.
    - **Consideration:** Update relevant article retrieval functions (e.g., `get_article_by_id`) if `frame_phrases` need to be fetched elsewhere later. Initially, this might not be required.

3.  **Gemini Client Update (`src/gemini/gemini_client.py`):**

    - **Action:** Modify `_load_prompt_template` or the prompt loading mechanism to use the content of `prompt9.md` as the template for `generate_text_with_prompt` (and async version).
    - **Action:** Enhance the response handling in `generate_text_with_prompt` (and async version). After extracting the text between the first `{` and last `}`, attempt to parse it using `json.loads`. Wrap this in a `try...except json.JSONDecodeError`.
    - **Action:** If parsing succeeds, return the _parsed JSON object_ (dict) instead of just the string. If parsing fails, log an error and return `None`.
    - **Rationale:** Ensures the client uses the optimized prompt and validates the structure of the compact JSON response before passing it downstream. Returning a parsed object simplifies downstream processing.

4.  **Step 3 Processing Logic Update (`src/steps/step3/__init__.py`):**

    - **Action:** Modify the code that calls `gemini_client.generate_text_with_prompt...`. It should now expect a Python dictionary (parsed JSON) or `None`.
    - **Action:** If a dictionary is received, validate its structure: check for top-level keys `'ents'` and `'fr'`. Validate that `'fr'` is a list of strings and `'ents'` is a list of dictionaries, each containing the expected short keys (`'en'`, `'et'`, `'mc'`, `'ic'`, `'ss'`) with appropriate data types.
    - **Action:** Extract the `frame_phrases` list from the `'fr'` key.
    - **Action:** Extract the entity list from the `'ents'` key and process entities as before (using the new short keys).
    - **Action:** After processing entities (e.g., calling `db_client.find_or_create_entity`, `db_client.link_article_entity`, `db_client.store_entity_snippet`), call the new `db_client.update_article_frames_and_mark_processed(article_id, frame_phrases)` function instead of the old `db_client.mark_article_processed(article_id)`. Pass the extracted `frame_phrases` list.
    - **Action:** Implement robust error handling for JSON validation failures or missing keys.
    - **Rationale:** Adapts Step 3 to the new API response format, extracts the framing information, and ensures it's stored correctly in the database along with the processing status update.

5.  **Documentation:**

    - **Action:** Update header comments in `schema.py`, `articles.py`, `reader_db_client.py`, `gemini_client.py`, and `src/steps/step3/__init__.py` to reflect the addition of `frame_phrases`, the use of the new prompt/JSON format, and any new/modified functions.
    - **Rationale:** Keep documentation consistent with code changes.

6.  **Testing:**
    - **Action:** Test the end-to-end flow: Ensure articles are processed, entities and frames are extracted correctly, the compact JSON is handled, and data (including `frame_phrases` and `extracted_entities=TRUE`) is stored accurately in the database. Test edge cases like API errors, JSON validation failures, and articles with no identifiable frames.
    - **Rationale:** Verify the correctness and robustness of the implementation.
