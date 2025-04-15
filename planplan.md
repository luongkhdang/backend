# Implementation Plan for Enhanced Hotness Calculation (plan.md -> planplan.md)

This plan outlines the steps to implement the refined hotness calculation logic described in `plan.md` into the existing `src/steps/step2/` module structure.

**Goal:** Enhance the `calculate_hotness_factors` function in `src/steps/step2/hotness.py` to include Tiered Persistence and Fading Penalty scores, using historical cluster data.

**Phase 1: Database Layer Modifications**

1.  **`src/database/modules/clusters.py`**:

    - Add a new function: `get_historical_clusters_by_days_ago(conn, days_ago: int) -> List[Dict[str, Any]]`.
      - **Purpose:** Retrieve cluster data (id, centroid, is_hot, article_count, metadata) from the most recent run on a specific past day (relative to the current Pacific Time date).
      - **SQL Logic:**
        - Calculate the target past date in 'America/Los_Angeles' timezone: `DATE(NOW() AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles' - INTERVAL '%s days')`.
        - Filter `clusters` table where `DATE(created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')` equals the target date.
        - Select the necessary columns (`id`, `centroid`, `is_hot`, `article_count`, `metadata`).
        - Order by `created_at DESC` and potentially `LIMIT 1` per original cluster if multiple runs happened on the same day (or handle aggregation if needed, but keep it simple first - maybe just get all clusters from that day). _Initial thought: Just get all clusters from that target date._
      - Return a list of dictionaries containing the cluster data.

2.  **`src/database/reader_db_client.py`**:
    - Add a new method: `get_historical_clusters(self, days_ago: int) -> List[Dict[str, Any]]`.
    - This method should call `clusters.get_historical_clusters_by_days_ago(conn, days_ago)` using a connection from the pool.

**Phase 2: Configuration**

1.  **`.env` / `docker-compose.yml`**:
    - Add the following new environment variables with appropriate default values:
      - `W_DAILY_P` (e.g., "0.05")
      - `W_WEEKLY_P` (e.g., "0.03")
      - `W_MONTHLY_P` (e.g., "0.02")
      - `W_FADING_PENALTY` (e.g., "-0.05")
      - `PERSISTENCE_SIMILARITY_THRESHOLD` (e.g., "0.90")
      - `DOWNWARD_TREND_FACTOR` (e.g., "0.75")
    - Verify/update existing weight variables (`W_SIZE`, `W_RECENCY`, `W_INFLUENCE`, `W_RELEVANCE`, `TARGET_HOT_CLUSTERS`) if their defaults need adjustment based on the new factors.

**Phase 3: Hotness Calculation Logic (`src/steps/step2/hotness.py`)**

1.  **`calculate_hotness_factors` Function:**
    - **Update Signature:** Modify the function signature to accept the historical cluster data fetched in `core.py`:
      ```python
      def calculate_hotness_factors(
          cluster_data: Dict[int, List[int]], # Current: label -> list of article indices
          article_ids: List[int],             # Full list of article IDs corresponding to indices
          pub_dates: List[Optional[datetime]],# Full list of pub dates
          reader_client: ReaderDBClient,
          nlp: Optional[Any],
          # --- New Parameters ---
          historical_clusters_1d: List[Dict[str, Any]],
          historical_clusters_7d: List[Dict[str, Any]],
          historical_clusters_30d: List[Dict[str, Any]],
          # Need current centroids to compare with historical
          current_centroids: Dict[int, List[float]]
      ) -> Dict[int, bool]:
      ```
    - **Load New Weights:** Read the new persistence and penalty weights from environment variables within the function or pass them from `core.py`.
    - **Pre-compute Entity Influence:** Keep the existing logic to fetch influence scores efficiently.
    - **Per-Cluster Score Calculation Loop:**
      - Inside the loop iterating through `cluster_data.keys()` (current cluster labels):
        - Get the current cluster's centroid from `current_centroids`.
        - **Persistence Scores (Daily, Weekly, Monthly):**
          - Compare the current centroid with centroids in `historical_clusters_1d`, `historical_clusters_7d`, `historical_clusters_30d` using cosine similarity (`1 - cosine_distance`).
          - If similarity > `PERSISTENCE_SIMILARITY_THRESHOLD` for any historical cluster in a period, assign a score of 1 for that period (e.g., `raw_scores[label]["daily_persistence"] = 1`), else 0.
        - **Fading Penalty Score:**
          - Find the most similar cluster (if any) in `historical_clusters_7d` exceeding the threshold.
          - Check if that historical cluster `was_hot` (using `is_hot` field).
          - Compare the current cluster's article count (`len(cluster_data[label])`) with the historical cluster's `article_count`.
          - If a similar hot cluster existed 7 days ago and the current size is less than `historical_count * DOWNWARD_TREND_FACTOR`, assign a score of 1 for the penalty (e.g., `raw_scores[label]["fading_penalty"] = 1`), else 0.
        - Keep existing calculations for Size, Recency, Influence, Relevance.
    - **Normalization:** Keep normalization for Size, Recency, Influence. Persistence and Fading Penalty scores are binary (0 or 1) and don't strictly need normalization in the same way, but ensure they are handled correctly in the weighted sum.
    - **Final Weighted Score:** Update the weighted sum calculation to include the new scores multiplied by their respective weights (`W_DAILY_P`, `W_WEEKLY_P`, `W_MONTHLY_P`, `W_FADING_PENALTY`). Remember the penalty weight is negative. Ensure the final score is capped at a minimum of 0.
      ```python
      final_score = (
          w_size * norm_scores["size"].get(label, 0) +
          w_recency * norm_scores["recency"].get(label, 0) +
          w_influence * norm_scores["influence"].get(label, 0) +
          w_relevance * raw_scores[label].get("relevance", 0) + # Assuming relevance is 0/1
          w_daily_p * raw_scores[label].get("daily_persistence", 0) +
          w_weekly_p * raw_scores[label].get("weekly_persistence", 0) +
          w_monthly_p * raw_scores[label].get("monthly_persistence", 0) +
          w_fading_penalty * raw_scores[label].get("fading_penalty", 0) # Penalty weight is negative
      )
      hotness_scores[label] = max(0, final_score) # Ensure score >= 0
      ```
    - **Ranking & Output:** Keep the existing logic for ranking by score and selecting the top `TARGET_HOT_CLUSTERS`.
    - **Helper Function:** Consider adding a helper function for calculating cosine similarity between centroids if not already available easily (e.g., using `sklearn.metrics.pairwise.cosine_similarity`).

**Phase 4: Orchestration (`src/steps/step2/core.py`)**

1.  **`run` Function:**
    - **Load New Config:** Load the new environment variables related to persistence/penalty weights and thresholds.
    - **Fetch Historical Data:** After initializing `reader_client` and _before_ calling `calculate_hotness_factors`:
      - Call `reader_client.get_historical_clusters(days_ago=1)`
      - Call `reader_client.get_historical_clusters(days_ago=7)`
      - Call `reader_client.get_historical_clusters(days_ago=30)`
    - **Pass Data to Hotness Function:** Modify the call to `calculate_hotness_factors`:
      - Pass the fetched historical cluster data lists.
      - Pass the `centroids` dictionary calculated in Step 5.
      - Ensure the `article_ids` list (mapping indices to DB IDs) is correctly passed. `cluster_data` currently maps label -> list of _indices_, so `article_ids` list is needed alongside it.

**Phase 5: Testing and Refinement**

1.  **Unit Tests (Optional but Recommended):** Add tests for `get_historical_clusters_by_days_ago` and the new logic within `calculate_hotness_factors`.
2.  **Integration Tests:** Run the full Step 2 pipeline with varying historical data (or mock data) to ensure persistence and fading penalties are calculated correctly.
3.  **Tuning:** Monitor the logs and adjust weights/thresholds in the environment configuration based on observed results.

**Notes:**

- This plan leverages the existing modular structure.
- Focus on implementing the core logic first. Optimization (e.g., more complex SQL for historical data retrieval) can come later if needed.
- Ensure robust handling of potentially missing historical data (e.g., if the pipeline didn't run exactly 7 days ago). The current plan fetches clusters _from_ that specific day, which might return empty lists if no run occurred. The hotness calculation should handle empty lists gracefully (resulting in zero scores for persistence/penalty for that period).
- Pay attention to data types (e.g., numpy ints vs Python ints) when using dictionary keys. The existing code seems to handle some of this.
- Use `sklearn.metrics.pairwise.cosine_similarity` for comparing centroids. Remember similarity = `1 - distance`.

