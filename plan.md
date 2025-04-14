It will use reader_db_client to fetch data and update results.
It will leverage local ML/NLP libraries (@localnlp_client.py).

Data Retrieval (reader_db_client):
Fetch Embeddings: Implement a method in reader_db_client (e.g., get_all_embeddings()) to retrieve all (article_id, embedding) pairs from the embeddings table.
Data Structure: Load these into memory, perhaps using Pandas DataFrames, a simple lists/dictionaries mapping article_id to its NumPy embedding vector. This will be the input for the clustering algorithm. For very large datasets, consider memory constraints and potential batching if needed, but start simple.

Core Clustering Algorithm (Local):
Algorithm: Use HDBSCAN (pip install hdbscan).
Why: As per your context, it's preferred because it doesn't require pre-specifying the number of clusters (k), handles noise points (articles not belonging to any cluster) well by assigning them label -1, and can find clusters of varying densities/shapes common in text data. It's generally more robust than KMeans for this type of task.
Implementation:
Import hdbscan.
Prepare the embeddings as a NumPy array (X).
Instantiate the clusterer:
import hdbscan

# metric='cosine' is suitable for normalized text embeddings

# Adjust min_cluster_size based on expected smallest group size (e.g., 5-15)

# min_samples controls noise points (leave default or slightly smaller than min_cluster_size)

clusterer = hdbscan.HDBSCAN(metric='cosine', min_cluster_size=10, min_samples=None, prediction_data=True)
Fit the model: cluster_labels = clusterer.fit_predict(X)
Keep track of the mapping between the original article_id list and the cluster_labels array indices.

Centroid Calculation
Why: Provides a representative vector for each cluster, useful for storage, interpretation, and finding related articles.
how:
Iterate through unique cluster labels found (labels >= 0).
For each label, select the embeddings belonging to that cluster.
Calculate the mean vector of these embeddings.
Normalize the mean vector (important for cosine distance). Store this as the cluster centroid.

Database Updates (reader_db_client)
clusters Table:
Implement ReaderDBClient.upsert_cluster(centroid, article_count) (or similar). Use INSERT ... ON CONFLICT DO UPDATE if re-clustering might update existing centroids/counts.
In step2.py, iterate through calculated centroids. For each, call upsert_cluster to store the centroid and the number of articles in that cluster. Store the returned cluster_id mapped to the original HDBSCAN label.

articles Table:
Implement ReaderDBClient.batch_update_article_clusters(cluster_assignments: List[Tuple[int, int]]). This method should efficiently update the cluster_id for many articles at once (e.g., using UPDATE ... FROM (VALUES...) or temporary tables).
In step2_clustering.py, create a list of (article_id, db_cluster_id) tuples based on the HDBSCAN results and the newly inserted cluster IDs. Handle noise points (label -1) by assigning cluster_id=NULL. Call batch_update_article_clusters.

pgvector Indexing (Crucial for Performance):

Action: Ensure vector indexes are created in ReaderDBClient.initialize_tables.
Code (initialize_tables): Add SQL commands after table creation:
-- Use vector_cosine_ops if embeddings are normalized and using cosine distance
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_clusters_hnsw ON clusters USING hnsw (centroid vector_cosine_ops);
Why: This dramatically speeds up potential nearest neighbor lookups used internally by some clustering algorithms or for later analysis (like finding articles near a centroid).

Cluster Interpretation (Keep Simple Initially):

Goal: Get a basic understanding of cluster themes without over-engineering.
How (step2.py or separate script):
For each generated cluster ID (or a subset of interest):
Retrieve a sample of article titles/summaries associated with that cluster ID from the database.
Combine the text of these samples.
Keyword Extraction (spaCy): Use your existing spaCy setup (en_core_web_lg) to extract the most frequent or relevant noun chunks or named entities from the combined text. These keywords often provide a good initial theme.
Store these keywords/tags, perhaps in a new JSONB metadata column in the clusters table (requires schema update) or log them for manual review.

Orchestration (src/main.py):

Add step2.py as a new step in the main pipeline orchestration.
Decide on frequency: Clustering usually runs less often than ingestion/embedding (e.g., daily, weekly). Make this configurable.
Avoiding Over-Engineering:

Stick to HDBSCAN: Avoid the complexity of finding k for KMeans initially.
Local First: Perform core clustering and basic interpretation (keyword extraction) locally using free libraries (hdbscan, spacy, numpy).
Database for Storage: Use the database primarily to store the results (cluster assignments, centroids) rather than implementing complex clustering logic in SQL. Rely on pgvector indexing for performance.
Incremental Interpretation: Start with simple keyword extraction for themes. Only add more complex interpretation (zero-shot, LLM calls) if the basic approach proves insufficient for the commercial goal.
Batch Processing: Design the clustering step as a batch process that can run on a schedule. Avoid real-time clustering unless absolutely necessary.

Confirm/Implement Batch Updates: Ensure the batch_update_article_clusters method exists or is added to ReaderDBClient for efficiency.
Memory Monitoring: Keep the note about memory usage for large datasets in mind during implementation.
Error Handling: Implement robust error handling throughout the clustering script and database interactions.

suggested latest stable version of tech stack:
Pandas: 2.2.3
NumPy: 2.2.4
HDBSCAN: 0.8.40
pgvector: 0.8.0

Tasks for Step 2 (Clustering Implementation):
Based on your validated plan and the provided context:

Setup Script: Create the main script for this step (e.g., src/steps/step2_clustering.py).
Data Retrieval:
Implement fetching of all relevant (article_id, embedding) pairs from the reader.embeddings table using the reader_db_client.
Load embeddings into memory (e.g., using Pandas or dictionaries), mapping article_id to its NumPy embedding vector. Keep potential memory constraints for large datasets in mind.
Prepare Data: Convert the embeddings data structure into a NumPy array suitable for input to HDBSCAN. Maintain a mapping between the array indices and the original article_ids.
Core Clustering:
Import and instantiate the hdbscan.HDBSCAN clusterer.
Configure parameters, especially metric='cosine' and tune min_cluster_size based on expected cluster sizes. Consider min_samples for noise handling. Set prediction_data=True if needed later.
Fit the model to the embeddings array (clusterer.fit_predict(X)).
Centroid Calculation:
Iterate through the unique cluster labels generated by HDBSCAN (excluding the noise label -1).
For each cluster, select the corresponding embedding vectors.
Calculate the mean vector (centroid).
Normalize the centroid vector (L2 normalization).
Database Updates (using reader_db_client):
Clusters Table:
Implement or confirm an upsert_cluster method in ReaderDBClient that takes the normalized centroid and article count. Use INSERT ... ON CONFLICT DO UPDATE.
Call this method for each calculated centroid, storing the returned database cluster_id. Map this DB cluster_id back to the HDBSCAN label.
Articles Table:
Implement or confirm a batch_update_article_clusters method in ReaderDBClient for efficient updates.
Prepare a list of (article_id, db_cluster_id) tuples based on the clustering results and the mapped DB cluster IDs.
Assign NULL as the db_cluster_id for articles labeled as noise (-1).
Call the batch update method.
Indexing Verification:
Ensure the reader_db_client.initialize_tables method includes the CREATE INDEX IF NOT EXISTS ... USING hnsw (... vector_cosine_ops) commands for both the embeddings(embedding) and clusters(centroid) columns.
Basic Cluster Interpretation (Optional but Recommended):
Implement logic to retrieve a sample of article titles/content for each generated cluster ID.
Use spaCy (e.g., en_core_web_lg) to extract keywords (noun chunks, named entities) from the combined text of the samples.
Decide how to store/use these keywords (e.g., log them, add a metadata column to clusters).
Orchestration:
Modify src/main.py to import and execute the step2_clustering.py script after Step 1.
Make the frequency of running Step 2 configurable (e.g., via environment variable or command-line argument).
Configuration & Environment:
Ensure necessary libraries (hdbscan, numpy, pandas, spacy) are added to requirements.txt and the Docker environment (DOCKER.txt).
