Stage D: Entity Extraction and Scoring

- **Goal:** Identify relevant entities, link them to articles, and calculate an influence score,balancing accuracy with API quota cost.
- **Output:** Populate `reader.entities` (upserting, updating `mentions`, `influence_score`, `last_seen`) and `reader.article_entities`
