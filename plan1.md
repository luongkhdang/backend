# Plan: Enhance Cluster Metadata

**Goal:** Enhance the `metadata` JSONB column in the `clusters` table to provide a richer, more informative snapshot of each cluster's characteristics beyond just name, description, and basic metrics.

**Current Metadata:**

- `name`, `description` (if provided during creation)
- `article_count` (updated periodically)
- `metrics` (updated by `update_cluster_metrics`, structure determined by that function)
- `pacific_date` (added previously for display)

**Proposed Additional Metadata Fields:**

1.  **`top_keywords` (List[str]):** The most frequent or representative keywords derived from the titles or content of articles within the cluster. (Limit: e.g., top 10)
2.  **`key_entities` (List[Dict[str, Any]]):** The most prominent named entities (people, organizations, locations) mentioned across the articles in the cluster. Could include entity name, type, and frequency/aggregate score. (Limit: e.g., top 5)
3.  **`date_range` (Dict[str, str]):** The earliest (`min_date`) and latest (`max_date`) publication dates of articles within the cluster, indicating its temporal span. Dates stored as ISO strings.
4.  **`top_domains` (List[Dict[str, Any]]):** The most frequent source domains contributing articles to this cluster, along with their counts or percentages. (Limit: e.g., top 5)
5.  **`representative_article_id` (Optional[int]):** The ID of an article that is very close to the cluster's centroid, serving as a prime example of the cluster's content.

**Implementation Plan:**

1.  **Location:** The calculation and updating of this enhanced metadata should occur within **Step 2 (Clustering)**, specifically after the HDBSCAN algorithm has assigned articles to clusters and these assignments have been initially recorded in the database. The refactored `src/steps/step2/` structure is the target location.
2.  **New Function:** Create a new function within the Step 2 logic (e.g., in `src/steps/step2/cluster_processing.py` or a new `metadata_generation.py` utility file):
    ```python
    def generate_and_update_cluster_metadata(
        cluster_id: int,
        article_ids: List[int],
        embeddings_map: Dict[int, List[float]], # Map article_id -> embedding
        cluster_centroid: Optional[List[float]],
        reader_db_client: ReaderDBClient
    ) -> None:
        """
        Calculates enhanced metadata for a cluster and updates the database.
        Args:
            cluster_id: The ID of the cluster to process.
            article_ids: List of article IDs belonging to this cluster.
            embeddings_map: Dictionary mapping article IDs to their embeddings.
            cluster_centroid: The calculated centroid for this cluster.
            reader_db_client: Instance of ReaderDBClient for DB operations.
        """
        # Implementation steps below...
    ```
3.  **Data Fetching (within the new function):**
    - Use `reader_db_client` to fetch necessary data for the given `article_ids`:
      - Article titles (`get_sample_titles_for_articles` or a tailored query).
      - Article publication dates (`get_publication_dates_for_articles`).
      - Article domains (requires a new `reader_db_client` method or extending an existing one).
      - Linked entities via `article_entities` table (requires a new `reader_db_client` method).
4.  **Metadata Calculation (within the new function):**
    - **Keywords:** Implement a simple keyword extraction logic (e.g., process titles, remove stop words, count frequency, take top N).
    - **Entities:** Aggregate fetched entities, rank by frequency (or influence score if available), take top N.
    - **Date Range:** Find the minimum and maximum dates from the fetched publication dates. Format as ISO strings.
    - **Domains:** Count occurrences of each domain, calculate percentages if desired, take top N.
    - **Representative Article:** If a centroid exists, iterate through `article_ids`, calculate the distance between each article's embedding (from `embeddings_map`) and the `cluster_centroid`. Select the article ID with the minimum distance. (Requires a distance function, e.g., cosine distance).
5.  **Database Update (within the new function):**
    - Construct a dictionary `enhanced_metadata` containing the calculated fields (`top_keywords`, `key_entities`, `date_range`, `top_domains`, `representative_article_id`).
    - Call `reader_db_client.update_cluster_metadata(cluster_id, enhanced_metadata)`. Ensure this method correctly _merges_ the new data with any existing metadata in the JSONB column.
6.  **Integration into Step 2:**
    - Identify the loop in `src/steps/step2/cluster_processing.py` (or similar) where clusters are processed after HDBSCAN (likely after `reader_db_client.batch_update_article_clusters`).
    - Inside the loop for each valid cluster (not noise points, i.e., `cluster_id != -1`), call the `generate_and_update_cluster_metadata` function, passing the necessary arguments (cluster ID, article IDs for that cluster, embeddings, centroid, DB client).
7.  **New `ReaderDBClient` Methods (if needed):**
    - Add methods like `get_domains_for_articles(article_ids)` and `get_entities_for_articles(article_ids)` if they don't exist or cannot be easily adapted from existing ones.
8.  **Dependencies:** Keyword extraction might require basic NLP libraries (like `nltk` or `spacy` for stop words/tokenization, potentially already installed) or can be done with standard Python collections. Distance calculation requires `numpy` or `scipy.spatial.distance`.
9.  **Documentation:** Update header comments and docstrings for all modified/new files and functions.
