# Implementation Plan for Step 2: Clustering

This plan details the steps to implement the article clustering functionality (Step 2) based on embeddings, as outlined in `plan.md` and verified against the existing codebase.

**Goal:** Group articles into meaningful clusters using their embeddings, calculate representative centroids, store results in the database, and provide basic cluster interpretation. Aim for a robust, batch-oriented process suitable for commercial use, avoiding initial over-engineering.

**1. File Structure & Setup**

- Create a new file: `src/steps/step2_clustering.py`.
- This file will contain the main logic for fetching data, running clustering, calculating centroids, and updating the database.

**2. Dependencies**

- Add the following libraries to `requirements.txt`:
  - `hdbscan>=0.8.36` (Verify latest compatible version)
  - `pandas>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `numpy~=1.26.4`
- Update `Dockerfile`: Ensure the new dependencies listed above are installed via `pip install -r requirements.txt`. Keep `spacy` and the `en_core_web_lg` download if performing basic interpretation locally.

**3. Database Modifications (`src/database/reader_db_client.py`)**

- **Add Method `get_all_embeddings()`**:
  - Signature: `def get_all_embeddings(self) -> List[Tuple[int, List[float]]]:`
  - Functionality: Query the `embeddings` table to retrieve all `(article_id, embedding)` pairs where `embedding` is not NULL. Return as a list of tuples. Handle potential errors. Log the number of embeddings retrieved.
- **Add Method `insert_cluster_get_id()`**:
  - Signature: `def insert_cluster_get_id(self, centroid: List[float], article_count: int, is_hot: bool = False) -> Optional[int]:`
  - Functionality: Insert a new row into the `clusters` table with the provided `centroid`, `article_count`, and `is_hot` status. Return the newly generated primary key (`id`). Handle potential errors.
- **Add Method `batch_update_article_cluster_assignments()`**:
  - Signature: `def batch_update_article_cluster_assignments(self, assignments: List[Tuple[int, Optional[int]]]) -> Tuple[int, int]:` (Returns success/fail count)
  - Functionality: Efficiently update the `cluster_id` column in the `articles` table for multiple articles. Input is a list of `(article_id, db_cluster_id)` tuples. `db_cluster_id` should be `NULL` for noise points (-1 label from HDBSCAN). Use `psycopg2.extras.execute_values` with an `UPDATE ... FROM (VALUES ...)` structure for performance. Handle potential errors and transaction management. Log success/failure counts.
- **Add Method `upsert_cluster_metadata()`** (If storing interpretation results):
  - Signature: `def upsert_cluster_metadata(self, cluster_id: int, metadata: Dict[str, Any]) -> bool:`
  - Functionality: Update the `metadata` JSONB column for a given `cluster_id`. Use `UPDATE clusters SET metadata = metadata || %s WHERE id = %s;` to merge JSONB data.
- **Modify `initialize_tables()`**:
  - Add `metadata JSONB` column to the `clusters` table definition.
  - Add HNSW index creation after table definitions:
    ```sql
    CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS idx_clusters_hnsw ON clusters USING hnsw (centroid vector_cosine_ops);
    ```
  - Ensure other relevant indexes exist (e.g., on `articles.cluster_id`).

**4. Clustering Logic (`src/steps/step2_clustering.py`)**

- Import necessary libraries (`hdbscan`, `numpy`, `pandas`, `sklearn.preprocessing`, `logging`, `os`, `time`, `typing`).
- Import `ReaderDBClient` from `src.database.reader_db_client`.
- Define a main function `run_step2()`.
- **Inside `run_step2()`**:
  - Initialize `ReaderDBClient`.
  - **Fetch Data**: Call `reader_db_client.get_all_embeddings()`. Log count. Handle empty results.
  - **Prepare Data**:
    - Create a mapping (e.g., list or dict) from the original `article_id` to the index in the embeddings list.
    - Convert the list of embedding vectors into a NumPy array `X`.
  - **Run HDBSCAN**:
    - Instantiate `clusterer = hdbscan.HDBSCAN(metric='cosine', min_cluster_size=int(os.getenv('MIN_CLUSTER_SIZE', '10')), prediction_data=True, core_dist_n_jobs=-1)` (Use `core_dist_n_jobs=-1` for potential parallelism).
    - Fit the model: `labels = clusterer.fit_predict(X)`. Log the number of clusters found and noise points.
  - **Calculate Centroids**:
    - Create a dictionary `cluster_data = {label: {'indices': []} for label in np.unique(labels) if label >= 0}`.
    - Iterate through `labels` and populate `cluster_data[label]['indices'].append(index)`.
    - Iterate through `cluster_data`:
      - Select embeddings using `indices`: `cluster_embeddings = X[data['indices']]`.
      - Calculate mean vector: `centroid = np.mean(cluster_embeddings, axis=0)`.
      - Normalize centroid: `normalized_centroid = sklearn.preprocessing.normalize(centroid.reshape(1, -1), norm='l2')[0].tolist()`.
      - Store results: `cluster_data[label]['centroid'] = normalized_centroid`
      - Store count: `cluster_data[label]['count'] = len(data['indices'])`
  - **Update Database (Clusters)**:
    - Initialize `hdbscan_to_db_id_map = {}`.
    - Iterate through `cluster_data`:
      - Call `db_cluster_id = reader_db_client.insert_cluster_get_id(centroid=data['centroid'], article_count=data['count'])`.
      - If `db_cluster_id` is not None, store the mapping: `hdbscan_to_db_id_map[label] = db_cluster_id`. Handle insertion errors.
  - **Update Database (Article Assignments)**:
    - Retrieve the original `article_id` list corresponding to the order of `labels`.
    - Prepare the list of assignments: `assignments = [(article_id, hdbscan_to_db_id_map.get(label)) for article_id, label in zip(original_article_ids, labels)]` (Note: `.get(label)` implicitly returns `None` if label is -1 or not found in the map).
    - Call `reader_db_client.batch_update_article_cluster_assignments(assignments)`. Log results.
  - **(Optional) Basic Interpretation**:
    - If spaCy is available:
      - Load spaCy model (`spacy.load("en_core_web_lg")`).
      - For each `db_cluster_id` generated:
        - Fetch a sample (e.g., 10-20) of `article_id`s belonging to this cluster.
        - Fetch corresponding article titles/content using `reader_db_client`.
        - Combine text samples.
        - Process combined text with spaCy to extract keywords (e.g., top N noun chunks or named entities).
        - Log the keywords or store them using `reader_db_client.upsert_cluster_metadata(db_cluster_id, {'keywords': extracted_keywords})`.
  - **Logging**: Add comprehensive logging for each phase (data fetching, clustering, centroid calculation, DB updates).
  - **Cleanup**: Ensure `reader_db_client.close()` is called, possibly in a `finally` block.
  - Return a status dictionary summarizing the operation (e.g., embeddings processed, clusters found, articles updated).

**5. Orchestration (`src/main.py`)**

- Import `run_step2` from `src.steps.step2_clustering`.
- Add a conditional execution block after Step 1 completes:
  ```python
  if os.getenv("RUN_CLUSTERING_STEP", "false").lower() == "true":
      logger.info("========= STARTING STEP 2: CLUSTERING =========")
      step2_status = run_step2()
      logger.info("Step 2 Summary:")
      logger.info(json.dumps(step2_status, indent=2))
      logger.info("========= STEP 2 COMPLETE =========")
  else:
      logger.info("Skipping Step 2: Clustering (RUN_CLUSTERING_STEP not true)")
  ```

**6. Configuration (`docker-compose.yml`)**

- Add environment variables to the `article-transfer` service:
  - `RUN_CLUSTERING_STEP: "false"` (or `"true"` to run by default)
  - `MIN_CLUSTER_SIZE: "10"` (Adjust default as needed)

**7. Error Handling & Robustness**

- Implement try-except blocks around major operations (DB calls, clustering).
- Add clear logging messages for errors and progress.
- Ensure database connections are properly closed.
- Consider potential memory constraints when loading all embeddings, although start with the simple approach first. If memory becomes an issue, investigate batch processing for clustering.

This plan provides a concrete path to implement Step 2, addressing the requirements from `plan.md` and integrating with the existing codebase structure.
