# Implementation Plan: Top N Hot Clusters (planfour.md)

## 1. Goal

Modify the cluster hotness determination in `src/steps/step2/hotness.py` to select a target number of top-scoring clusters as "hot", instead of relying on a fixed score threshold. Aim for approximately 15 hot clusters by default.

## 2. Analysis

The previous approach used a fixed `HOTNESS_THRESHOLD` (0.6), which resulted in only 1 cluster being marked as hot. The calculated scores for other clusters were significantly lower. Trying to perfectly calibrate weights and a threshold can be complex and may need frequent readjustment. Selecting the top N clusters provides a more stable and predictable way to identify the _most relatively interesting_ clusters based on the defined factors.

## 3. Chosen Approach: Top N Selection

1.  **Calculate Weighted Score:** Continue calculating the weighted hotness score for all non-noise clusters using the existing factors (Size, Recency, Influence, Relevance) and weights (`W_SIZE`, `W_RECENCY`, `W_INFLUENCE`, `W_RELEVANCE`). Normalization (min-max) will still be used before weighting to balance factor scales for ranking.
2.  **Target Number:** Introduce a new environment variable `TARGET_HOT_CLUSTERS` (Default: 15) to specify how many top clusters should be marked as hot.
3.  **Rank and Select:** Rank all calculated clusters based on their final weighted score in descending order.
4.  **Mark Top N:** Mark the top `TARGET_HOT_CLUSTERS` in the ranking as `is_hot = True`. All other clusters will be `is_hot = False`.
5.  **Remove Threshold:** The `HOTNESS_THRESHOLD` environment variable and its associated logic will be removed.
6.  **Enhance Logging:** Improve logging within `calculate_hotness_factors` to output raw scores, normalized scores (if applicable), and the final weighted score for _all_ clusters, facilitating future analysis and potential weight tuning.

## 4. Detailed Implementation Steps

**4.1. Modify `calculate_hotness_factors` Function (`src/steps/step2/hotness.py`)**

- **Inputs:** Signature remains the same: `calculate_hotness_factors(cluster_data, article_ids, pub_dates, reader_client, nlp=None)`.
- **Internal Logic:**
  1.  Calculate raw scores for Size, Recency, Influence, and Relevance for all non-noise clusters (as currently implemented).
  2.  Normalize scores for Size, Recency, and Influence using min-max scaling (as currently implemented). Relevance remains 0 or 1.
  3.  Get weight parameters (`W_SIZE`, `W_RECENCY`, `W_INFLUENCE`, `W_RELEVANCE`) from environment variables.
  4.  Calculate the final weighted `hotness_score` for each non-noise cluster label. Store these scores (e.g., in a dictionary `hotness_scores: Dict[int, float]`).
  5.  Get the target number of hot clusters from the `TARGET_HOT_CLUSTERS` environment variable (default to 15).
  6.  Sort the clusters by their `hotness_score` in descending order.
  7.  Create the `hotness_map: Dict[int, bool]`. Iterate through the sorted clusters:
      - Mark the first `TARGET_HOT_CLUSTERS` as `True`.
      - Mark the rest as `False`.
  8.  **Logging:** Enhance the logging section to include a summary of raw scores (min/max/avg) per factor, normalized scores (min/max/avg), and the final weighted scores for _all_ clusters, or at least a larger sample (e.g., top 20).
- **Output:** Return the `hotness_map: Dict[int, bool]`.

**4.2. Remove `get_hotness_threshold` Function (`src/steps/step2/hotness.py`)**

- Delete the `get_hotness_threshold` function as it's no longer needed.

**4.3. Update Environment Variables (`docker-compose.yml`)**

- Remove the `HOTNESS_THRESHOLD` environment variable definition.
- Add the `TARGET_HOT_CLUSTERS` environment variable definition (e.g., `TARGET_HOT_CLUSTERS: ${TARGET_HOT_CLUSTERS:-15}`).

## 5. File Modifications

- `src/steps/step2/hotness.py`: Major changes to `calculate_hotness_factors`, removal of `get_hotness_threshold`, enhanced logging.
- `docker-compose.yml` (Recommended): Add `TARGET_HOT_CLUSTERS`, remove `HOTNESS_THRESHOLD`.

## 6. Benefits

- Directly achieves the goal of identifying a specific number of top clusters.
- More robust to variations in absolute score distributions.
- Simplifies initial configuration (no need to guess a threshold).
- Weighted scoring logic is retained for meaningful ranking.
- Improved logging aids future tuning and understanding of factor contributions.

## 7. Future Work

- Continue investigating the root causes if Influence and Relevance scores remain consistently low, as this might indicate issues in upstream data processing or entity/topic definitions.
- Refine weights (`W_...`) based on the enhanced logs and desired ranking behavior.
