!IMPORTANT: SINCE WE ARE IN DEVELOPMENT MODE, LET'S ONLY PROCESS 1 DAY OLD ARTICLES (TODAY AND YESTERDAY).

THIS IS THE IMPLEMENTATION PLAN FOR STEP 3.

**Objective:** To process 2000 news articles daily, extracting relevant entities, determining their contextual influence within each article, and calculating a global influence score for each unique entity, leveraging tiered AI processing based on combined topic hotness and source quality.

**Inputs:**

- ~2000 daily articles from `articles` table (`id`, `domain`, `cluster_id`, `content`).
- `clusters` table (`id`, `hotness_score`).
- Pre-calculated `domain_goodness_score` for each domain.
- (Implicitly) Existing data in `entities` and `article_entities` for updates.

**Outputs:**

- New/updated records in `article_entities` linking articles to entities with per-article `mention_count`.
- New/updated records in `entities` table with `entity_type`, global `mentions` count, and calculated `influence_score`.

**Detailed Step-by-Step Plan:**

1.  **Domain Goodness Score Calculation (Run Periodically - e.g., Daily/Weekly):**

    - **Purpose:** To assess the general quality and relevance of news sources based on volume and track record of producing "hot" content.
    - **Method:**
      - Calculate `total_entries` and `hot_percentage` per domain (deriving `is_hot` potentially from `cluster.hotness_score > threshold`).
      - Apply minimum entry threshold: Score domains only if `total_entries > 5` **OR** `hot_percentage = 100.0`. Assign 0 otherwise.
      - Calculate `NormLogVolume` based on `LOG(total_entries)`.
      - Calculate `BaseGoodness = (0.6 * NormLogVolume) + (0.4 * hot_percentage / 100.0)`.
      - Add Consistency Bonus: `FinalGoodness = BaseGoodness + (0.1 if hot_percentage > 15.0 else 0.0)`.
      - Store `domain_goodness_score` per domain for lookup.

2.  **Daily Article Prioritization & Tiering (Run Daily):**

    - **Purpose:** To rank the 2000 daily articles and assign a processing tier to allocate AI resources effectively.
    - **Method:**
      - For each of the 2000 `articles`: fetch its `cluster_hotness_score` (via `cluster_id`, default 0) and its pre-calculated `domain_goodness_score`.
      - Normalize both scores to a 0-1 range.
      - Calculate `Combined_Priority_Score = (0.65 * norm_hotness) + (0.35 * norm_domain_goodness)`.
      - Rank articles by `Combined_Priority_Score` (descending).
      - Assign `processing_tier` (0, 1, or 2) based on rank: Tier 0 = Top ~150, Tier 1 = Next ~350, Tier 2 = Remainder ~1500. Associate this tier with the article for this run.

3.  **Gemini API Entity/Context Extraction (Run Daily - Tier-Based):**

    - **Purpose:** To use the appropriate Gemini model to extract entities and assess their context within each article's content.
    - **Method:**
      - For each article, select the Gemini model based on `processing_tier`:
        - Tier 0: `'models/gemini-2.0-flash-thinking-exp-01-21'` (Fallback: `'models/gemini-2.0-flash'`)
        - Tier 1: `'models/gemini-2.0-flash-exp'` (Fallback: `'models/gemini-2.0-flash'`)
        - Tier 2: `'models/gemini-2.0-flash'` (Fallback: `'models/gemini-2.0-flash-lite'`)
      - Use the finalized Gemini prompt (requesting structured JSON with `entity_name`, `entity_type`, `mention_count` [per-article], `is_influential_context`, `supporting_snippets`).
      - Submit API request with article `content`, handle retries/fallbacks.
      - Parse the structured JSON response. Store the result associated with the `article_id`, noting the source tier (T0/T1/T2).

4.  **Store Per-Article Entity Results :**

    - **Purpose:** To persist the link between articles and the entities mentioned within them, along with the per-article mention count.
    - **Method:**
      - For each entity identified in an article (from Step 3 or 4):
        - Find or create the `entity_id` in the `entities` table:
          ```sql
          INSERT INTO entities (name, entity_type)
          VALUES ($1, $2)
          ON CONFLICT (name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP -- Optionally update type if needed
          RETURNING id;
          ```
          _(Get the returned `entity_id`)_.
        - Insert the relationship into the junction table:
          ```sql
          INSERT INTO article_entities (article_id, entity_id, mention_count)
          VALUES ($1, $2, $3)
          ON CONFLICT (article_id, entity_id) DO NOTHING;
          ```
          _(Where $1=article_id, $2=entity_id, $3=per-article mention_count)_.

5.  **Calculate Global Influence Score & Update Entities Table (Run Daily/Batch):**
    - **Purpose:** To calculate and update the global `influence_score` and total `mentions` for each unique entity based on its appearances across articles over time.
    - **Method (Conceptual - Algorithm Needs Development):**
      - For each unique `entity_id` processed or updated today (or in a batch):
        - **Gather Inputs:** Aggregate relevant data points associated with this entity across _multiple_ `article_entities` records (potentially filtering by recency):
          - Per-article `mention_count` (sum for global `mentions`).
          - `is_influential_context` flags (proportion/count of influential mentions).
          - `domain_goodness_score` of the source articles.
          - `cluster_hotness_score` of the source articles/clusters.
          - Recency (`first_seen` via `entities.created_at`, `last_seen` via `MAX(article_entities.created_at)` or `entities.updated_at`).
          - Extraction quality (counts/proportions from Tier 0/1/2 vs Local Fallback).
          - `entity_type`.
        - **Calculate Score:** Apply a scoring algorithm (e.g., weighted sum of normalized input factors) to compute the `influence_score`. This algorithm needs to be designed based on desired emphasis (e.g., prioritize recent influential mentions in hot topics from good domains). _Keep the initial algorithm relatively simple._
        - **Update `entities` Table:**
          ```sql
          UPDATE entities
          SET
              influence_score = $1, -- Calculated score
              mentions = $2,       -- Updated global mention count
              entity_type = $3,    -- Can update type if needed
              updated_at = CURRENT_TIMESTAMP
          WHERE id = $4;
          ```

---

This comprehensive plan outlines the flow for Stage D, from prioritizing articles using combined signals, leveraging tiered AI for extraction, handling fallbacks, storing results per article, and finally updating the global entity metrics including the crucial `influence_score`.
