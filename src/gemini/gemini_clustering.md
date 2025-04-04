Embedding Integration (Gemini)
Role: Gemini embeddings turn article content into 768-dimensional vectors, stored in embeddings.embedding. They power:
Clustering: Group 2,000 articles into 50-100 topics (clusters), splitting hot (30%) from background (70%).

Correlations: Find similar articles/essays (e.g., cosine similarity on VECTOR)—e.g., “China’s dip” matches “Vietnam’s rise.”

Search: User filters (e.g., “Show me X Corp”) query embeddings for related content.

Storage: pgvector handles VECTOR type efficiently—e.g., SELECT \* FROM embeddings ORDER BY embedding <-> '[vector]' LIMIT 10 for nearest neighbors.

Scale: 2,000 articles/day = 2,000 embeddings (~1.5MB/day at 768 floats), manageable on a 1TB drive for years.
