# Plan: Add Hotness Score to Cluster Metadata

**Goal:** Store the calculated numerical hotness score within the `metadata` JSONB column for each cluster, ideally near the beginning of the JSON object for easy viewing.

**Analysis:**

1.  **Score Calculation:** The final numerical hotness score for each cluster label is calculated in `src/steps/step2/hotness.py` within the `calculate_hotness_factors` function and stored locally in the `hotness_scores` dictionary. However, this function currently only returns the `hotness_map` (boolean `is_hot` status based on Top N selection).
2.  **Metadata Update:** The enhanced metadata (keywords, date range, etc.) is generated and updated in `src/steps/step2/metadata_generation.py` via the `generate_and_update_cluster_metadata` function, which ultimately calls `reader_db_client.update_cluster_metadata`. This happens _after_ initial cluster creation and assignment updates.
3.  **Integration Point:** The most logical place to add the score to the metadata is within the `generate_and_update_cluster_metadata` function, as it already handles constructing and updating the `metadata` JSONB field. This requires passing the calculated scores from `core.py` into this function.

**Implementation Plan:**

1.  **Modify `src/steps/step2/hotness.py` (`calculate_hotness_factors`):**

    - Change the function's return signature to return both the boolean `hotness_map` _and_ the numerical `hotness_scores` dictionary.
      - Current: `-> Dict[int, bool]`
      - New: `-> Tuple[Dict[int, bool], Dict[int, float]]`
    - Update the `return` statement to return `hotness_map, hotness_scores`.
    - Update the function's docstring accordingly.

2.  **Modify `src/steps/step2/core.py` (`run` function):**

    - Update the call to `calculate_hotness_factors` to receive both returned values (the map and the scores dictionary).
      - Example: `cluster_hotness_map, cluster_hotness_scores = calculate_hotness_factors(...)`
    - In the loop where `generate_and_update_cluster_metadata` is called (Step 10):
      - Retrieve the specific `hotness_score` for the current `label` from the `cluster_hotness_scores` dictionary.
      - Pass this score as a new argument to `generate_and_update_cluster_metadata`.

3.  **Modify `src/steps/step2/metadata_generation.py` (`generate_and_update_cluster_metadata`):**

    - Add a new parameter `hotness_score: Optional[float]` to the function signature.
    - Update the function's docstring.
    - When constructing the `enhanced_metadata` dictionary:
      - Add the `hotness_score` field. Format it (e.g., round to 4 decimal places) and handle `None` cases.
      - Place this field near the beginning of the dictionary definition for better readability in the final JSONB (e.g., `{'hotness_score': ..., 'top_keywords': ...}`).

4.  **Modify `src/database/reader_db_client.py` (`update_cluster_metadata`):**

    - _No changes needed here._ This function already merges the provided dictionary with existing metadata, so adding the score to the dictionary in the previous step is sufficient.

5.  **Documentation:** Update relevant header comments and docstrings in modified files (`hotness.py`, `core.py`, `metadata_generation.py`).
