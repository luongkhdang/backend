# Database Schema Summary (@schema.md)

**Note:** This summary is generated from the schema definition in `src/database/modules/schema.py`. It reflects the intended structure but does not include live database statistics like row counts.

## Extensions

- **`vector`**: Enabled for vector similarity search (used in `embeddings` and potentially `clusters`).

## Tables

(Managed primarily by `src/database/modules/schema.py`)

### `articles`

(Managed by `src/database/modules/articles.py`)

| Column               | Type        | Constraints                             | Default | Description                                        |
| -------------------- | ----------- | --------------------------------------- | ------- | -------------------------------------------------- |
| `id`                 | `SERIAL`    | `PRIMARY KEY`                           |         | Unique identifier for the article                  |
| `scraper_id`         | `INTEGER`   | `UNIQUE`                                |         | ID from the original scraper/source                |
| `title`              | `TEXT`      |                                         |         | Title of the article                               |
| `content`            | `TEXT`      |                                         |         | Full text content of the article                   |
| `pub_date`           | `TIMESTAMP` |                                         |         | Publication date of the article                    |
| `domain`             | `TEXT`      |                                         |         | Domain name of the article source (e.g., nyt.com)  |
| `processed_at`       | `TIMESTAMP` |                                         |         | Timestamp when the article was processed           |
| `extracted_entities` | `BOOLEAN`   |                                         | `FALSE` | Flag indicating if entities have been extracted    |
| `is_hot`             | `BOOLEAN`   |                                         | `FALSE` | Flag indicating if the article is considered 'hot' |
| `cluster_id`         | `INTEGER`   | (Implicit FK to `clusters.id` SET NULL) |         | ID of the cluster the article belongs to           |
| `frame_phrases`      | `TEXT[]`    |                                         | `NULL`  | Array of narrative frame phrases, can be NULL      |

### `entities`

(Managed by `src/database/modules/entities.py`)

| Column                    | Type        | Constraints           | Default             | Description                                        |
| ------------------------- | ----------- | --------------------- | ------------------- | -------------------------------------------------- |
| `id`                      | `SERIAL`    | `PRIMARY KEY`         |                     | Unique identifier for the entity                   |
| `name`                    | `TEXT`      | `UNIQUE`              |                     | Name of the entity                                 |
| `entity_type`             | `TEXT`      |                       |                     | Type of the entity (e.g., PERSON, ORG)             |
| `influence_score`         | `FLOAT`     |                       | `0.0`               | Calculated influence score                         |
| `mentions`                | `INTEGER`   |                       | `0`                 | Global mention count across all articles           |
| `first_seen`              | `TIMESTAMP` |                       |                     | Timestamp when the entity was first seen           |
| `last_seen`               | `TIMESTAMP` |                       |                     | Timestamp when the entity was last seen            |
| `created_at`              | `TIMESTAMP` |                       | `CURRENT_TIMESTAMP` | Timestamp when the entity record was created       |
| `updated_at`              | `TIMESTAMP` |                       | `CURRENT_TIMESTAMP` | Timestamp when the entity record was last updated  |
| `influence_calculated_at` | `TIMESTAMP` | (Added if not exists) |                     | Timestamp when influence score was last calculated |

### `article_entities` (Junction Table)

(Managed by `src/database/modules/entities.py`)

| Column                   | Type        | Constraints                                           | Default             | Description                                            |
| ------------------------ | ----------- | ----------------------------------------------------- | ------------------- | ------------------------------------------------------ |
| `article_id`             | `INTEGER`   | `PRIMARY KEY`, `FK to articles(id) ON DELETE CASCADE` |                     | ID of the article                                      |
| `entity_id`              | `INTEGER`   | `PRIMARY KEY`, `FK to entities(id) ON DELETE CASCADE` |                     | ID of the entity                                       |
| `mention_count`          | `INTEGER`   |                                                       | `1`                 | Number of times the entity is mentioned in the article |
| `created_at`             | `TIMESTAMP` |                                                       | `CURRENT_TIMESTAMP` | Timestamp when the link was created                    |
| `is_influential_context` | `BOOLEAN`   | (Added if not exists)                                 | `FALSE`             | Flag indicating if the mention context was influential |

### `embeddings`

(Managed by `src/database/modules/embeddings.py`)

| Column       | Type          | Constraints                                      | Default             | Description                                       |
| ------------ | ------------- | ------------------------------------------------ | ------------------- | ------------------------------------------------- |
| `id`         | `SERIAL`      | `PRIMARY KEY`                                    |                     | Unique identifier for the embedding record        |
| `article_id` | `INTEGER`     | `UNIQUE`, `FK to articles(id) ON DELETE CASCADE` |                     | ID of the article this embedding belongs to       |
| `embedding`  | `VECTOR(768)` |                                                  |                     | The 768-dimension embedding vector                |
| `created_at` | `TIMESTAMP`   |                                                  | `CURRENT_TIMESTAMP` | Timestamp when the embedding was generated/stored |

### `clusters`

(Managed by `src/database/modules/clusters.py`)

| Column          | Type          | Constraints   | Default        | Description                                        |
| --------------- | ------------- | ------------- | -------------- | -------------------------------------------------- |
| `id`            | `SERIAL`      | `PRIMARY KEY` |                | Unique identifier for the cluster                  |
| `centroid`      | `VECTOR(768)` |               |                | Centroid vector representing the cluster           |
| `is_hot`        | `BOOLEAN`     |               | `FALSE`        | Flag indicating if the cluster is considered 'hot' |
| `article_count` | `INTEGER`     |               |                | Cached count of articles in the cluster            |
| `created_at`    | `DATE`        |               | `CURRENT_DATE` | Date when the cluster was created                  |
| `metadata`      | `JSONB`       |               |                | JSON blob for additional cluster metadata          |
| `hotness_score` | `FLOAT`       |               | `NULL`         | Calculated hotness score for the cluster           |

### `essays`

(Managed by `src/database/modules/essays.py`)

| Column        | Type        | Constraints                             | Default             | Description                                    |
| ------------- | ----------- | --------------------------------------- | ------------------- | ---------------------------------------------- |
| `id`          | `SERIAL`    | `PRIMARY KEY`                           |                     | Unique identifier for the essay                |
| `type`        | `TEXT`      |                                         |                     | Type of essay (e.g., summary, analysis)        |
| `article_id`  | `INTEGER`   |                                         |                     | Related article ID (optional, not FK enforced) |
| `title`       | `TEXT`      |                                         |                     | Title of the essay                             |
| `content`     | `TEXT`      |                                         |                     | Content of the essay                           |
| `layer_depth` | `INTEGER`   |                                         |                     | Depth or layer if part of a hierarchy          |
| `cluster_id`  | `INTEGER`   | `FK to clusters(id) ON DELETE SET NULL` |                     | Associated cluster ID                          |
| `created_at`  | `TIMESTAMP` |                                         | `CURRENT_TIMESTAMP` | Timestamp when the essay was created           |
| `tags`        | `TEXT[]`    |                                         |                     | Array of text tags associated with the essay   |

### `essay_entities` (Junction Table)

(Managed by `src/database/modules/essays.py`)

| Column      | Type      | Constraints                                           | Default | Description             |
| ----------- | --------- | ----------------------------------------------------- | ------- | ----------------------- |
| `essay_id`  | `INTEGER` | `PRIMARY KEY`, `FK to essays(id) ON DELETE CASCADE`   |         | ID of the essay         |
| `entity_id` | `INTEGER` | `PRIMARY KEY`, `FK to entities(id) ON DELETE CASCADE` |         | ID of the linked entity |

### `domain_statistics`

(Managed by `src/database/modules/domains.py`)

| Column                    | Type        | Constraints   | Default             | Description                                                            |
| ------------------------- | ----------- | ------------- | ------------------- | ---------------------------------------------------------------------- |
| `domain`                  | `TEXT`      | `PRIMARY KEY` |                     | The domain name (e.g., nyt.com)                                        |
| `total_entries`           | `INTEGER`   |               | `0`                 | Total articles processed from this domain                              |
| `hot_entries`             | `INTEGER`   |               | `0`                 | Number of 'hot' articles from this domain                              |
| `average_cluster_hotness` | `FLOAT`     |               | `0.0`               | Average hotness score of clusters containing articles from this domain |
| `goodness_score`          | `FLOAT`     |               | `0.0`               | Calculated overall 'goodness' score for the domain                     |
| `calculated_at`           | `TIMESTAMP` |               | `CURRENT_TIMESTAMP` | Timestamp when the statistics were last calculated                     |

### `entity_influence_factors`

(Managed by `src/database/modules/influence.py`)

| Column                  | Type        | Constraints                                           | Default             | Description                                    |
| ----------------------- | ----------- | ----------------------------------------------------- | ------------------- | ---------------------------------------------- |
| `entity_id`             | `INTEGER`   | `PRIMARY KEY`, `FK to entities(id) ON DELETE CASCADE` |                     | ID of the entity                               |
| `calculation_timestamp` | `TIMESTAMP` | `PRIMARY KEY`                                         | `CURRENT_TIMESTAMP` | Timestamp of this specific calculation         |
| `base_mention_score`    | `FLOAT`     |                                                       |                     | Component score based on mentions/reach        |
| `source_quality_score`  | `FLOAT`     |                                                       |                     | Component score based on source quality        |
| `content_context_score` | `FLOAT`     |                                                       |                     | Component score based on influential context   |
| `temporal_score`        | `FLOAT`     |                                                       |                     | Component score based on recency/trend         |
| `raw_data`              | `JSONB`     |                                                       |                     | Raw data used for this calculation (for audit) |

### `entity_type_weights`

(Managed by `src/database/modules/influence.py`)

| Column        | Type        | Constraints   | Default             | Description                                |
| ------------- | ----------- | ------------- | ------------------- | ------------------------------------------ |
| `entity_type` | `TEXT`      | `PRIMARY KEY` |                     | The entity type name                       |
| `weight`      | `FLOAT`     | `NOT NULL`    | `1.0`               | Weight multiplier for this entity type     |
| `updated_at`  | `TIMESTAMP` |               | `CURRENT_TIMESTAMP` | Timestamp when the weight was last updated |

_(Initial weights inserted for: PERSON, ORGANIZATION, GOVERNMENT_AGENCY, LOCATION, GEOPOLITICAL_ENTITY, CONCEPT, LAW_OR_POLICY, EVENT, OTHER)_

### `entity_snippets`

(Managed by `src/database/modules/entity_snippets.py`)

| Column           | Type        | Constraints                            | Default             | Description                                   |
| ---------------- | ----------- | -------------------------------------- | ------------------- | --------------------------------------------- |
| `id`             | `SERIAL`    | `PRIMARY KEY`                          |                     | Unique identifier for the snippet record      |
| `entity_id`      | `INTEGER`   | `FK to entities(id) ON DELETE CASCADE` |                     | ID of the associated entity                   |
| `article_id`     | `INTEGER`   | `FK to articles(id) ON DELETE CASCADE` |                     | ID of the source article                      |
| `snippet`        | `TEXT`      | `NOT NULL`                             |                     | The text snippet itself                       |
| `is_influential` | `BOOLEAN`   |                                        | `TRUE`              | Flag indicating if the snippet is influential |
| `created_at`     | `TIMESTAMP` |                                        | `CURRENT_TIMESTAMP` | Timestamp when the snippet was stored         |

### `clusters_fundamental`

(Managed by `src/database/modules/clusters.py`)

| Column     | Type          | Constraints   | Default | Description                                   |
| ---------- | ------------- | ------------- | ------- | --------------------------------------------- |
| `id`       | `SERIAL`      | `PRIMARY KEY` |         | Unique identifier for the fundamental cluster |
| `centroid` | `VECTOR(768)` |               |         | Centroid vector representing the cluster      |
| `metadata` | `JSONB`       |               |         | JSON blob for cluster metadata                |

## Indexes

- **`articles`**:
  - `idx_articles_scraper_id` on (`scraper_id`)
  - `idx_articles_cluster_id` on (`cluster_id`)
  - `idx_articles_domain` on (`domain`)
  - `idx_articles_pub_date` on (`pub_date`)
- **`entities`**:
  - `idx_entities_name` on (`name`)
  - `idx_entities_type` on (`entity_type`)
- **`article_entities`**:
  - `idx_article_entities_article_id` on (`article_id`)
  - `idx_article_entities_entity_id` on (`entity_id`)
- **`embeddings`**:
  - `idx_embeddings_article_id` on (`article_id`)
- **`domain_statistics`**:
  - `idx_domain_statistics_score` on (`goodness_score` DESC)
- **`entity_snippets`**:
  - `idx_entity_snippets_entity_id` on (`entity_id`)
  - `idx_entity_snippets_article_id` on (`article_id`)

_(Note: Primary Key indexes are created automatically)_
