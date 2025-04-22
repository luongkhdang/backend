# Plan for Step 4 Data Formatting Change (plan12.md)

**Goal:** Modify `src/steps/step4.py` to format the `articles_for_analysis` list using a positional compressed format (list of lists) instead of a list of dictionaries, matching the structure in `plannn.md`.

**Steps:**

1.  **Locate Data Preparation Loop:** In `src/steps/step4.py`, find the `for article in articles_raw:` loop where `analysis_input_item` is constructed.
2.  **Modify `analysis_input_item` Construction:**
    - Change the assignment from `analysis_input_item = { ... }` to `analysis_input_item = []`.
    - Populate this list with items in the exact order specified by `plannn.md`:
      - `article_id`
      - `title` (use `.get(..., '')` for safety)
      - `domain` (use `.get(..., '')` for safety)
      - `pub_date` (formatted as 'YYYY-MM-DD' string)
      - `frame_phrases` (use `.get(..., [])` for safety)
      - `entities_for_prompt` (which will now be a list of lists)
3.  **Modify Inner `entities_for_prompt` Construction:**
    - Inside the loop iterating through `top_entities_detail`, locate the construction of `entity_data`.
    - Change the assignment from `entity_data = { ... }` to `entity_data = []`.
    - Populate this inner list with items in the exact order specified by `plannn.md`:
      - `name` (use `.get('name')`)
      - `entity_type` (use `.get('entity_type')`)
      - `snippets` (the list of snippet strings already being generated)
    - Append this `entity_data` list to the `entities_for_prompt` list.
4.  **Verify Data Types:** Ensure all elements added to the lists have the correct data types (integer, string, list of strings, list of lists).
5.  **Test:** (Manual Step) Run the pipeline and verify the format of the `step4_input_data_[timestamp].json` debug file matches the `plannn.md` structure.
