# Step 2: Article Clustering Module

This directory contains the modularized implementation of Step 2 of the data refinery pipeline: clustering articles based on their embeddings using HDBSCAN.

## Module Structure

- `__init__.py`: Entry point that exports the main `run()` function
- `core.py`: Core implementation that orchestrates the entire clustering process
- `data_fetching.py`: Functions for retrieving article embeddings from the database
- `clustering.py`: Implements the clustering algorithm and centroid calculation
- `hotness.py`: Determines which clusters should be marked as "hot"
- `database.py`: Handles database operations for storing clusters and updating articles
- `interpretation.py`: Provides cluster interpretation and keyword extraction
- `utils.py`: Utility functions used across the module

## Execution Flow

1. **Data Retrieval**: Fetch article embeddings and metadata from the database
2. **Data Preparation**: Format embeddings for clustering
3. **Clustering**: Apply HDBSCAN to group similar articles
4. **Centroid Calculation**: Calculate representative vectors for each cluster
5. **Hotness Determination**: Mark clusters as "hot" based on multiple factors
6. **Database Update**: Store cluster data and update article assignments
7. **Interpretation**: Extract keywords and identify topics for each cluster

## Dependencies

- `numpy`: For numerical operations on embeddings
- `hdbscan`: For the clustering algorithm
- `scikit-learn`: For preprocessing and distance calculations
- `spacy` (optional): For NLP-based cluster interpretation

## Environment Variables

- `MIN_CLUSTER_SIZE`: Minimum size of clusters (default: 10)
- `HOT_CLUSTER_THRESHOLD`: Minimum size for a cluster to be hot (default: 20)
- `INTERPRET_CLUSTERS`: Whether to perform cluster interpretation (default: false)
- `MAX_CLUSTERS_TO_INTERPRET`: Maximum number of clusters to interpret (default: 10)
- `CLUSTER_SAMPLE_SIZE`: Number of articles to sample per cluster (default: 10)
- `HOTNESS_SCORE_THRESHOLD`: Threshold for marking clusters as hot (default: 0.7)
- `HOTNESS_RECENCY_WEIGHT`: Weight for recency in hotness calculation (default: 0.5)
- `HOTNESS_INFLUENCE_WEIGHT`: Weight for influence in hotness calculation (default: 0.3)
- `HOTNESS_SIZE_WEIGHT`: Weight for size in hotness calculation (default: 0.2)

## Usage

```python
from src.steps.step2 import run

# Execute clustering process
status = run()
print(f"Clustering status: {'Success' if status['success'] else 'Failed'}")
print(f"Found {status['clusters_found']} clusters")
```
