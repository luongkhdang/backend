# Implementation Plan for Sophisticated Cluster Hotness Score (planthree.md)

## 1. Goal

To implement a more nuanced `is_hot` determination for clusters in `src/steps/step2.py`, replacing the simple article count threshold with a weighted "hotness score" based on cluster size, recency, entity influence, and topic relevance, as inspired by `plan3.md`.

## 2. Chosen Approach (Refined for Practicality & Relevance)

We will calculate a weighted score based on four factors:

1.  **Size Factor:** Logarithmic count, normalized.
2.  **Recency Factor:** Proportion of recent articles (e.g., last 3 days), normalized.
3.  **Entity Influence Factor:** Average influence score of linked entities, normalized.
4.  **Topic Relevance Factor:** Binary score based on matching extracted keywords against a predefined list of core topics.

**Strategy:** To incorporate Topic Relevance before cluster insertion, we will perform a simplified, targeted keyword extraction step specifically for the hotness calculation. The more detailed `interpret_cluster` function will still run _after_ insertion to populate the `metadata` column for later use.

Weights for each factor (`W_SIZE`, `W_RECENCY`, `W_INFLUENCE`, `W_RELEVANCE`) and the final `HOTNESS_THRESHOLD` will be configurable via environment variables.

## 3. Detailed Implementation Steps

**3.1. Modify Data Fetching (`src/steps/step2.py`)**

- **`get_all_embeddings` function:**
  - Update the SQL query to fetch `article_id`, `embedding`, and `pub_date` from the `articles` and `embeddings` tables.
  - Ensure `pub_date` is handled correctly (may be `None`).
  - The function should now return a list of tuples like `(article_id, embedding, pub_date)`.
- **`run` function:**
  - Adjust the unpacking after calling `get_all_embeddings` to handle the three returned values (`article_ids`, `embeddings`, `pub_dates`).
  - Load the `spacy` model (`nlp`) conditionally if Topic Relevance calculation or the later `interpret_cluster` step is enabled via environment variables (e.g., `CALCULATE_TOPIC_RELEVANCE` or `INTERPRET_CLUSTERS`).

**3.2. Add Database Client Methods (`src/database/reader_db_client.py`)**

- Keep the previously planned method: `get_entity_influence_for_articles(self, article_ids: List[int]) -> Dict[int, float]`.
- Add a new method: `get_sample_titles_for_articles(self, article_ids: List[int], sample_size: int) -> List[str]`:
  - Input: List of article database IDs, sample size.
  - Query: Fetch `title` for the given `article_ids`, potentially ordering by `pub_date DESC` or another relevant factor, limiting by `sample_size`.
  - Output: A list of non-null title strings.

**3.3. Add `get_cluster_keywords` Function (`src/steps/step2.py`)**

- Create a new, focused function `get_cluster_keywords(reader_client, article_db_ids, nlp, sample_size)` separate from the existing `interpret_cluster`.
- **Inputs:**
  - `reader_client: ReaderDBClient`.
  - `article_db_ids: List[int]` (database IDs for articles in the specific cluster).
  - `nlp`: Loaded spaCy model.
  - `sample_size: int` (from env var `CLUSTER_SAMPLE_SIZE`).
- **Internal Logic:**
  1. Call `reader_client.get_sample_titles_for_articles(article_db_ids, sample_size)` to get sample titles.
  2. Handle the case where no titles are returned.
  3. Combine the titles into a single text string.
  4. Process the text with the `nlp` model (`doc = nlp(combined_text)`).
  5. Extract the top N (e.g., 5) most frequent/relevant noun chunks (e.g., `chunk.text.lower()` for case-insensitivity). Prioritize noun chunks as they often represent topics better than single entities for this purpose.
- **Output:** Return a list of the top keyword strings (e.g., `['us trade', 'semiconductor policy', 'economic impact']`).

**predefined-list-of-core list**:
China
US / United States
Vietnam
Europe
Germany
War
Trade / Exports / Tariffs
Geopolitics / Geopolitical
Political Economy
Influence / Lobbying
Narrative / Framing
Disinformation / Misinformation
AI / Artificial Intelligence (especially regarding policy/influence)
Election / Campaign
Pentagon / Defense
State Department / Diplomacy
ITC / International Trade Commission

**3.4. Modify `calculate_hotness_factors` Function (`src/steps/step2.py`)**

- This function still runs _after_ `calculate_centroids` and _before_ `insert_clusters`.
- **Inputs:** Update signature to `calculate_hotness_factors(cluster_data, article_ids, pub_dates, reader_client, nlp=None)` (make `nlp` optional).
- **Internal Logic:**
  1. Initialize dictionaries to store raw scores for Size, Recency, Influence, and Relevance per cluster label.
  2. Get `CORE_TOPIC_KEYWORDS` from env var (e.g., comma-separated string, convert to a set of lowercase strings).
  3. Get `CLUSTER_SAMPLE_SIZE` from env var.
  4. **Iterate through `cluster_data` (each cluster label):**
     - Get `indices` and calculate `article_db_ids`.
     - **Size Factor (Raw):** Calculate `log(count + 1)`. Store.
     - **Recency Factor (Raw):** Calculate proportion of recent articles (last `RECENCY_DAYS`). Store.
     - **Entity Influence Factor (Raw):** Call `reader_client.get_entity_influence_for_articles()`, calculate average. Store.
     - **Topic Relevance Factor (Raw):**
       - Initialize raw score to 0.
       - If `nlp` is provided (meaning relevance calculation is enabled):
         - Call `get_cluster_keywords(reader_client, article_db_ids, nlp, CLUSTER_SAMPLE_SIZE)`.
         - Check if any returned keyword exists in the `CORE_TOPIC_KEYWORDS` set.
         - If a match is found, set the raw score to 1.
       - Store the raw relevance score (0 or 1).
  5. **Normalization (Min-Max Scaling):**
     - Normalize scores for Size, Recency, and Influence to 0-1 range across all clusters.
     - The Relevance score remains 0 or 1.
  6. **Calculate Weighted Score & Determine `is_hot`:**
     - Get weights (`W_SIZE`, `W_RECENCY`, `W_INFLUENCE`, `W_RELEVANCE`) and `HOTNESS_THRESHOLD` from env vars.
     - Calculate `hotness_score = (W_SIZE * norm_size) + (W_RECENCY * norm_recency) + (W_INFLUENCE * norm_influence) + (W_RELEVANCE * norm_relevance)`. Note `norm_relevance` is just the 0 or 1 score.
     - Determine `is_hot = hotness_score >= HOTNESS_THRESHOLD`.
- **Output:** Return `cluster_hotness_map: Dict[int, bool]`.

**3.5. Modify `insert_clusters` Function (`src/steps/step2.py`)**

- Update the function signature to accept `cluster_hotness_map`: `insert_clusters(reader_client, cluster_data, cluster_hotness_map)`.
- Inside the loop, get `is_hot` from the map and pass it to `reader_client.insert_cluster()`. (No change from previous plan).

**3.6. Keep `interpret_cluster` Function (`src/steps/step2.py`)**

- This function remains optional (controlled by `INTERPRET_CLUSTERS` env var) and runs _after_ cluster insertion.
- Its role is solely to populate the `clusters.metadata` column with more detailed interpretation (entities, topics) for display/analysis, not for the initial `is_hot` decision.

**3.7. Add/Update Environment Variables**

- Define defaults and allow overrides for:
  - `RECENCY_DAYS` (Default: 3)
  - `CLUSTER_SAMPLE_SIZE` (Default: 10)
  - `CORE_TOPIC_KEYWORDS` (Default: "china,us,united states,vietnam,europe,germany,war,trade,exports,tariffs,geopolitics,geopolitical,political economy,influence,lobbying,narrative,framing,disinformation,misinformation,ai,artificial intelligence,election,campaign,pentagon,defense,state department,diplomacy,itc,international trade commission")
  - `CALCULATE_TOPIC_RELEVANCE` (Default: true) - Controls whether the relevance factor is calculated.
  - `W_SIZE` (Default: 0.15)
  - `W_RECENCY` (Default: 0.30)
  - `W_INFLUENCE` (Default: 0.30)
  - `W_RELEVANCE` (Default: 0.25)
  - `HOTNESS_THRESHOLD` (Default: 0.6)

## 4. Error Handling & Edge Cases

- Handle `None` values, missing entities/scores, and potential division by zero during normalization.
- Gracefully handle cases where `spacy` or the model isn't available if relevance calculation is attempted.
- Log warnings/errors but allow the process to continue.

## 5. File Modifications

- `src/steps/step2.py`: Changes in `run`, `get_all_embeddings`, `insert_clusters`. Addition of `get_cluster_keywords` and significant updates to `calculate_hotness_factors`.
- `src/database/reader_db_client.py`: Add `get_entity_influence_for_articles` and `get_sample_titles_for_articles` methods.
- `docker-compose.yml` (Recommended): Add/update environment variables.

This plan provides a structured way to implement the more sophisticated hotness calculation while managing complexity and potential performance considerations.
