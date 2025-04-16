# Step 3 Future Implementation Plan (part2.md)

This document outlines the future enhancements to be implemented after the basic version described in `part1.md` is completed and functioning properly. The focus is on implementing the more complex features that were simplified in the initial version.

## 1. Global Influence Score Calculation Algorithm

The initial implementation in `part1.md` only updates entity mention counts. This section details the more sophisticated influence score calculation algorithm.

### 1.1 Influence Score Inputs

The following factors will be used to calculate each entity's global influence score:

1. **Base Mention Metrics**:

   - Total mention count across all articles (`mentions`)
   - Number of distinct articles mentioning the entity (`article_count`)
   - Proportion of mentions where `is_influential_context=true` (`influence_ratio`)

2. **Source Quality Factors**:

   - Average `domain_goodness_score` of articles mentioning the entity
   - Processing tier distribution (% from Tier 0, 1, 2) as a quality signal

3. **Content Context Factors**:

   - Average `cluster_hotness_score` of articles mentioning the entity
   - Entity type weights (e.g., PERSON vs ORGANIZATION vs CONCEPT)

4. **Temporal Factors**:
   - Recency of mentions (weighted towards more recent)
   - Mention frequency over time (trending up or down)

### 1.2 Database Schema Updates

Additional tables/columns needed:

```sql
-- Add column to store influence calculation timestamp
ALTER TABLE entities ADD COLUMN IF NOT EXISTS influence_calculated_at TIMESTAMP;

-- Add new table to store influence calculation factors for debugging/transparency
CREATE TABLE IF NOT EXISTS entity_influence_factors (
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    base_mention_score FLOAT,
    source_quality_score FLOAT,
    content_context_score FLOAT,
    temporal_score FLOAT,
    raw_data JSONB,  -- Store raw inputs for auditability
    PRIMARY KEY (entity_id, calculation_timestamp)
);

-- Add table for entity type weights that can be adjusted
CREATE TABLE IF NOT EXISTS entity_type_weights (
    entity_type TEXT PRIMARY KEY,
    weight FLOAT NOT NULL DEFAULT 1.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Initialize with default weights
INSERT INTO entity_type_weights (entity_type, weight) VALUES
('PERSON', 1.0),
('ORGANIZATION', 1.1),
('GOVERNMENT_AGENCY', 1.2),
('LOCATION', 0.8),
('GEOPOLITICAL_ENTITY', 1.3),
('CONCEPT', 1.0),
('LAW_OR_POLICY', 1.1),
('EVENT', 0.9),
('OTHER', 0.7)
ON CONFLICT (entity_type) DO NOTHING;
```

### 1.3 Calculation Algorithm

The algorithm will be implemented in a new module `src/database/modules/influence.py`:

```python
def calculate_entity_influence_score(conn, entity_id: int, recency_days: int = 30) -> float:
    """
    Calculate comprehensive influence score for an entity based on multiple factors.

    Args:
        conn: Database connection
        entity_id: Entity ID to calculate score for
        recency_days: Number of days to prioritize for recency calculation

    Returns:
        Calculated influence score
    """
    try:
        # Step 1: Collect raw data
        raw_data = _collect_entity_influence_data(conn, entity_id, recency_days)

        # Step 2: Calculate component scores (each normalized to 0-1 range)
        base_score = _calculate_base_mention_score(raw_data)
        quality_score = _calculate_source_quality_score(raw_data)
        context_score = _calculate_content_context_score(raw_data)
        temporal_score = _calculate_temporal_score(raw_data)

        # Step 3: Apply weights to component scores (adjustable)
        WEIGHTS = {
            'base': 0.35,    # Base mention metrics
            'quality': 0.20, # Source quality
            'context': 0.25, # Content context
            'temporal': 0.20 # Recency and trend
        }

        final_score = (
            WEIGHTS['base'] * base_score +
            WEIGHTS['quality'] * quality_score +
            WEIGHTS['context'] * context_score +
            WEIGHTS['temporal'] * temporal_score
        )

        # Step 4: Apply entity type modifier (from entity_type_weights table)
        entity_type = _get_entity_type(conn, entity_id)
        type_weight = _get_entity_type_weight(conn, entity_type)
        final_score *= type_weight

        # Step 5: Store calculation factors for transparency
        _store_influence_factors(conn, entity_id, {
            'base_score': base_score,
            'quality_score': quality_score,
            'context_score': context_score,
            'temporal_score': temporal_score,
            'final_score': final_score,
            'raw_data': raw_data
        })

        # Step 6: Update the entity's influence score
        _update_entity_influence_score(conn, entity_id, final_score)

        return final_score

    except Exception as e:
        logger.error(f"Error calculating influence score for entity {entity_id}: {e}")
        return 0.0
```

The algorithm will be broken down into component functions, each handling different aspects:

```python
def _collect_entity_influence_data(conn, entity_id: int, recency_days: int) -> Dict[str, Any]:
    """Collect all data needed for influence calculation."""
    # SQL queries to gather:
    # 1. Total mentions and article count
    # 2. Influential context ratio
    # 3. Domain goodness scores of mentioning articles
    # 4. Cluster hotness scores of mentioning articles
    # 5. Processing tier distribution
    # 6. Temporal data (recent vs older mentions)
    # Return as structured dictionary

def _calculate_base_mention_score(data: Dict[str, Any]) -> float:
    """Calculate normalized score based on mention metrics."""
    # Use log scale for mentions to prevent domination by very frequent entities
    # Combine with influential context ratio

def _calculate_source_quality_score(data: Dict[str, Any]) -> float:
    """Calculate score based on quality of sources mentioning the entity."""
    # Weight by domain goodness score
    # Consider processing tier as quality signal

def _calculate_content_context_score(data: Dict[str, Any]) -> float:
    """Calculate score based on content context factors."""
    # Weight by cluster hotness
    # Consider entity relationships (future)

def _calculate_temporal_score(data: Dict[str, Any]) -> float:
    """Calculate score based on temporal factors."""
    # Recent mentions get higher weight
    # Consider trend (increasing/decreasing mentions)
```

### 1.4 Implementation in ReaderDBClient

```python
def calculate_entities_influence_scores(self, entity_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Calculate influence scores for specified entities or all entities.

    Args:
        entity_ids: Optional list of entity IDs to calculate for. If None, calculates for all.

    Returns:
        Dict with success count, error count, and calculation summary
    """
    try:
        with self.get_connection() as conn:
            if entity_ids is None:
                # Get entities updated recently
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT e.id
                    FROM entities e
                    JOIN article_entities ae ON e.id = ae.entity_id
                    WHERE ae.created_at > (CURRENT_DATE - INTERVAL '7 DAY')
                    OR e.influence_calculated_at IS NULL
                    OR e.influence_calculated_at < (CURRENT_DATE - INTERVAL '7 DAY')
                """)
                entity_ids = [row[0] for row in cursor.fetchall()]
                cursor.close()

            success_count = 0
            error_count = 0

            for entity_id in entity_ids:
                try:
                    score = calculate_entity_influence_score(conn, entity_id)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error calculating influence for entity {entity_id}: {e}")
                    error_count += 1

            return {
                "success_count": success_count,
                "error_count": error_count,
                "total_entities": len(entity_ids)
            }

    except Exception as e:
        logger.error(f"Error in batch influence calculation: {e}")
        return {
            "success_count": 0,
            "error_count": 0,
            "total_entities": 0,
            "error": str(e)
        }
```

## 2. Domain Goodness Score Calculation

The full implementation of the domain goodness calculation logic from `plan1.md` step 1.

### 2.1 Schema Updates

Create a dedicated table for domain statistics:

```sql
CREATE TABLE IF NOT EXISTS domain_statistics (
    domain TEXT PRIMARY KEY,
    total_entries INTEGER DEFAULT 0,
    hot_entries INTEGER DEFAULT 0,
    average_cluster_hotness FLOAT DEFAULT 0.0,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2.2 Implementation

Create a new module `src/steps/domain_goodness.py`:

```python
def calculate_domain_goodness_scores(db_client, min_entries: int = 5, hot_threshold_percent: float = 15.0,
                                    bonus_value: float = 0.1, domain_volume_weight: float = 0.6,
                                    domain_hotness_weight: float = 0.4) -> Dict[str, Any]:
    """
    Calculate goodness scores for all domains based on volume and hotness.

    Args:
        db_client: ReaderDBClient instance
        min_entries: Minimum entries required for scoring (unless 100% hot)
        hot_threshold_percent: Percentage threshold for consistency bonus
        bonus_value: Value of consistency bonus
        domain_volume_weight: Weight for normalized log volume component
        domain_hotness_weight: Weight for hotness percentage component

    Returns:
        Dict with calculation statistics
    """
    # Implementation similar to SQL logic in plan1.md
    # 1. Calculate raw stats per domain (total_entries, hot_count)
    # 2. Calculate hot percentage
    # 3. Apply minimum entry threshold
    # 4. Calculate log volume and normalize
    # 5. Calculate base goodness score
    # 6. Add consistency bonus
    # 7. Store results in calculated_domain_goodness table
```

This function would be scheduled to run weekly or whenever needed.

## 3. Supporting Snippets Storage and Retrieval

Implementation for storing and retrieving the supporting snippets from entity extraction.

### 3.1 Schema Updates

```sql
CREATE TABLE IF NOT EXISTS entity_snippets (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    snippet TEXT NOT NULL,
    is_influential BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_entity_snippets_entity_id ON entity_snippets(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_snippets_article_id ON entity_snippets(article_id);
```

### 3.2 Implementation

Update the `_store_results` function in `step3.py` to handle snippets:

```python
def _store_entity_snippets(db_client, entity_id: int, article_id: int, entity_data: Dict[str, Any]) -> int:
    """
    Store supporting snippets for an entity in an article.

    Args:
        db_client: ReaderDBClient instance
        entity_id: Entity ID
        article_id: Article ID
        entity_data: Entity data dictionary from extraction

    Returns:
        Number of snippets stored
    """
    if (not entity_data.get('is_influential_context') or
        not entity_data.get('supporting_snippets')):
        return 0

    snippets_stored = 0

    for snippet in entity_data['supporting_snippets']:
        if db_client.store_entity_snippet(
            entity_id=entity_id,
            article_id=article_id,
            snippet=snippet,
            is_influential=True
        ):
            snippets_stored += 1

    return snippets_stored
```

Add corresponding function to `ReaderDBClient`:

```python
def store_entity_snippet(self, entity_id: int, article_id: int,
                         snippet: str, is_influential: bool = True) -> bool:
    """Store a supporting snippet for an entity."""
    try:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO entity_snippets
                    (entity_id, article_id, snippet, is_influential)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (entity_id, article_id, snippet, is_influential))

            success = cursor.fetchone() is not None
            conn.commit()
            cursor.close()
            return success
    except Exception as e:
        logger.error(f"Error storing entity snippet: {e}")
        return False

def get_entity_snippets(self, entity_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Get supporting snippets for an entity."""
    try:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT es.id, es.entity_id, es.article_id, es.snippet,
                       es.is_influential, es.created_at, a.title as article_title
                FROM entity_snippets es
                JOIN articles a ON es.article_id = a.id
                WHERE es.entity_id = %s
                ORDER BY es.created_at DESC
                LIMIT %s
            """, (entity_id, limit))

            columns = [desc[0] for desc in cursor.description]
            snippets = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            return snippets
    except Exception as e:
        logger.error(f"Error retrieving entity snippets: {e}")
        return []
```

## 4. Integration with Main Pipeline

Update `src/main.py` to include the new functionality:

```python
# Add to imports
from src.steps.step3 import run as run_step3
from src.steps.domain_goodness import calculate_domain_goodness_scores

# After executing step 2
if os.getenv("RUN_STEP3", "false").lower() == "true":
    logger.info("========= STARTING STEP 3: ENTITY EXTRACTION =========")
    try:
        step3_status = run_step3()
        logger.info("Step 3 Summary:")
        logger.debug(json.dumps(step3_status, indent=2))

        if step3_status.get("success", False):
            logger.info(
                f"Entity extraction successful: {step3_status.get('processed', 0)} articles processed")

            # Calculate influence scores for entities extracted in this run
            if os.getenv("CALCULATE_INFLUENCE_SCORES", "true").lower() == "true":
                logger.info("Calculating influence scores for extracted entities")
                influence_status = db_client.calculate_entities_influence_scores()
                logger.info(f"Influence score calculation: {influence_status}")
        else:
            logger.warning(
                f"Entity extraction completed with issues: {step3_status.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Step 3 failed with error: {e}", exc_info=True)
    finally:
        logger.info("========= STEP 3 COMPLETE =========")
else:
    logger.info("Skipping Step 3: Entity Extraction (RUN_STEP3 not true)")

# Weekly domain goodness calculation (can be triggered separately or scheduled)
if os.getenv("CALCULATE_DOMAIN_GOODNESS", "false").lower() == "true":
    logger.info("========= CALCULATING DOMAIN GOODNESS SCORES =========")
    try:
        domain_status = calculate_domain_goodness_scores(db_client)
        logger.info(f"Domain goodness calculation: {domain_status}")
    except Exception as e:
        logger.error(f"Domain goodness calculation failed: {e}", exc_info=True)
    finally:
        logger.info("========= DOMAIN GOODNESS CALCULATION COMPLETE =========")
```

## 5. Performance Optimizations

Additional optimizations for handling large volumes of entities and articles:

1. **Batch Processing**:

   - Process entities in batches to avoid memory issues
   - Use connection pooling for database operations

2. **Caching**:

   - Cache domain goodness scores and cluster hotness scores
   - Cache entity type weights

3. **Async Processing**:
   - Move influence score calculation to a background task
   - Implement a job queue for processing entities

## 6. Timeline and Priorities

1. **Phase 1 (First)**:

   - Implement entity snippets storage and retrieval
   - Implement basic domain goodness calculation

2. **Phase 2 (Second)**:

   - Implement global influence score calculation
   - Add caching and optimization

3. **Phase 3 (Last)**:
   - Add advanced analytics and reporting
   - Implement trend detection and visualization
