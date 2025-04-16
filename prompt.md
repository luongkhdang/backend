### SQL Queries for Prioritization & Tiering

These queries implement the logic defined in Step 1 (Domain Goodness Calculation - run periodically) and Steps 2 & 3 (Daily Prioritization & Tier Assignment).

**Query 1: Calculate Domain Goodness Score**

This query calculates the score for each domain based on volume and hotness, incorporating the specific threshold and bonus logic. The results should be stored in a table (e.g., `calculated_domain_goodness`) for efficient daily lookup.

```sql
-- Parameters (adjust as needed)
DEFINE MIN_ENTRIES = 5;
DEFINE HOT_THRESHOLD_PERCENT = 15.0;
DEFINE BONUS_VALUE = 0.1;
DEFINE DOMAIN_VOLUME_WEIGHT = 0.6;
DEFINE DOMAIN_HOTNESS_WEIGHT = 0.4;
DEFINE MIN_CLUSTER_HOTNESS_FOR_IS_HOT = 0.3; -- Example threshold for an article to count as 'hot'

-- Calculate and store domain goodness scores
-- Replace 'calculated_domain_goodness' with your actual table name
CREATE TABLE calculated_domain_goodness AS
WITH DomainRawStats AS (
    -- Calculate total entries and hot count per domain
    SELECT
        a.domain,
        COUNT(a.id) AS total_entries,
        SUM(CASE WHEN COALESCE(c.hotness_score, 0.0) >= :MIN_CLUSTER_HOTNESS_FOR_IS_HOT THEN 1 ELSE 0 END) AS hot_count
    FROM articles a
    LEFT JOIN clusters c ON a.cluster_id = c.id
    WHERE a.domain IS NOT NULL -- Ensure domain is not null
    GROUP BY a.domain
),
DomainPercents AS (
    -- Calculate hot percentage
    SELECT
        domain,
        total_entries,
        hot_count,
        CASE
            WHEN total_entries > 0 THEN (hot_count * 100.0) / total_entries
            ELSE 0.0
        END AS hot_percentage
    FROM DomainRawStats
),
ValidDomains AS (
    -- Apply minimum entry threshold (unless 100% hot)
    SELECT
        domain,
        total_entries,
        hot_percentage
    FROM DomainPercents
    WHERE total_entries >= :MIN_ENTRIES OR hot_percentage = 100.0
),
LogVolumeStats AS (
    -- Calculate log volume (base 10 for potentially better distribution)
    SELECT
        domain,
        total_entries,
        hot_percentage,
        LOG(10, GREATEST(total_entries, 1)) AS LogVolume
    FROM ValidDomains
),
MaxLogVolume AS (
    -- Find max log volume for normalization
    SELECT MAX(LogVolume) AS MaxLogVol FROM LogVolumeStats WHERE LogVolume IS NOT NULL AND LogVolume > 0 -- Avoid issues with max=0
),
DomainGoodnessBase AS (
    -- Calculate normalized log volume and base score
    SELECT
        lvs.domain,
        lvs.total_entries,
        lvs.hot_percentage,
        -- Normalize log volume (handle case where MaxLogVol might be 0 or null)
        CASE
            WHEN COALESCE(mlv.MaxLogVol, 0) > 0 THEN (lvs.LogVolume / mlv.MaxLogVol)
            ELSE 0.0
        END AS NormLogVolume,
        -- Base score calculation
        (:DOMAIN_VOLUME_WEIGHT * (CASE WHEN COALESCE(mlv.MaxLogVol, 0) > 0 THEN (lvs.LogVolume / mlv.MaxLogVol) ELSE 0.0 END)) +
        (:DOMAIN_HOTNESS_WEIGHT * lvs.hot_percentage / 100.0) AS BaseGoodness
    FROM LogVolumeStats lvs
    CROSS JOIN MaxLogVolume mlv
)
-- Final Domain Goodness Score
SELECT
    dgb.domain,
    -- Add consistency bonus
    ROUND(
        dgb.BaseGoodness +
        CASE
            WHEN dgb.hot_percentage >= :HOT_THRESHOLD_PERCENT THEN :BONUS_VALUE
            ELSE 0.0
        END,
    1) AS domain_goodness_score,
    CURRENT_TIMESTAMP AS calculated_at -- Track when it was calculated
FROM DomainGoodnessBase dgb

UNION ALL

-- Add domains that didn't meet the threshold with a score of 0
SELECT
    dp.domain,
    0.0 AS domain_goodness_score,
    CURRENT_TIMESTAMP AS calculated_at
FROM DomainPercents dp
LEFT JOIN ValidDomains vd ON dp.domain = vd.domain
WHERE vd.domain IS NULL;

-- Add appropriate indexes on calculated_domain_goodness(domain)
```

**Query 2: Prioritize Daily Articles and Assign Tiers**

This query selects the 2000 articles for the day, calculates their combined priority score, ranks them, and assigns the processing tier. The output is intended to feed the next step (API processing).

```sql
-- Parameters (adjust as needed)
DEFINE CLUSTER_HOTNESS_WEIGHT = 0.65;
DEFINE DOMAIN_GOODNESS_WEIGHT = 0.35;
DEFINE TIER0_COUNT = 150;
DEFINE TIER1_COUNT = 350;

-- Select daily articles, calculate priority, rank, and assign tiers
WITH DailyArticles AS (
    -- Select the 2000 articles for today's processing run
    SELECT id AS article_id, domain, cluster_id, content
    FROM articles
    WHERE is_processed = FALSE -- Example: select unprocessed articles
    ORDER BY created_at DESC -- Or some other logic for selecting daily batch
    LIMIT 2000
),
ArticleScores AS (
    -- Join daily articles with cluster hotness and pre-calculated domain goodness
    SELECT
        da.article_id,
        da.content,
        da.domain, -- Keep domain if needed later
        COALESCE(c.hotness_score, 0.0) AS cluster_hotness_score,
        COALESCE(cdg.domain_goodness_score, 0.0) AS domain_goodness_score
    FROM DailyArticles da
    LEFT JOIN clusters c ON da.cluster_id = c.id
    LEFT JOIN calculated_domain_goodness cdg ON da.domain = cdg.domain -- Join with pre-calculated scores
),
NormalizationParams AS (
    -- Find max scores IN THIS BATCH for normalization relative to today's data
    SELECT
        GREATEST(MAX(cluster_hotness_score), 0.0001) AS max_hotness, -- Use GREATEST to avoid division by zero
        GREATEST(MAX(domain_goodness_score), 0.0001) AS max_goodness
    FROM ArticleScores
),
CombinedScores AS (
    -- Calculate normalized scores and the final combined priority score
    SELECT
        a.article_id,
        a.content,
        a.cluster_hotness_score,
        a.domain_goodness_score,
        -- Normalize scores
        (a.cluster_hotness_score / np.max_hotness) AS norm_hotness,
        (a.domain_goodness_score / np.max_goodness) AS norm_domain_goodness,
        -- Calculate combined score
        (:CLUSTER_HOTNESS_WEIGHT * (a.cluster_hotness_score / np.max_hotness)) +
        (:DOMAIN_GOODNESS_WEIGHT * (a.domain_goodness_score / np.max_goodness))
        AS combined_priority_score
    FROM ArticleScores a
    CROSS JOIN NormalizationParams np
),
RankedArticles AS (
    -- Rank articles based on the combined score
    SELECT
        article_id,
        content,
        combined_priority_score,
        ROW_NUMBER() OVER (ORDER BY combined_priority_score DESC, article_id) as priority_rank
    FROM CombinedScores
)
-- Final Output for Processing: Articles with assigned Tiers
SELECT
    ra.article_id,
    ra.content, -- Pass content to the next stage
    ra.priority_rank,
    cs.cluster_hotness_score, -- Include scores for potential later use
    cs.domain_goodness_score,
    ra.combined_priority_score,
    CASE
        WHEN ra.priority_rank <= :TIER0_COUNT THEN 0 -- Tier 0
        WHEN ra.priority_rank <= (:TIER0_COUNT + :TIER1_COUNT) THEN 1 -- Tier 1
        ELSE 2 -- Tier 2
    END AS processing_tier
FROM RankedArticles ra
JOIN CombinedScores cs ON ra.article_id = cs.article_id -- Join back to get original scores if needed
ORDER BY ra.priority_rank;
```

---

### Final Gemini API Prompt for Entity/Context Extraction

This prompt is refined based on best practices and the specific requirements for Stage D.

````text
## Role:
You are an expert AI analyst specializing in identifying entities and their influence within geopolitical and economic news narratives. Your task contributes to a system designed to help users trace power dynamics in the news.

## Task:
Carefully analyze the provided news article text. Extract key entities and assess their contextual significance based on the following criteria:

1.  **Entities:** Identify significant named entities. Categorize using these specific types:
    * PERSON
    * ORGANIZATION (Companies, NGOs, Political Parties, etc.)
    * GOVERNMENT_AGENCY (Specific departments, ministries, central banks, etc.)
    * LOCATION (City, State, Country, recognised geographical areas)
    * GEOPOLITICAL_ENTITY (Formal or informal groups of nations like G7, ASEAN, BRICS; disputed regions like 'South China Sea')
    * CONCEPT (Abstract but central ideas like 'Belt and Road Initiative', 'economic decoupling', 'populism', 'free trade agreement')
    * LAW_OR_POLICY (Specific named laws, treaties, or official policies)
    * EVENT (Named conferences, conflicts, elections, etc.)
    * OTHER (If significant but doesn't fit above)
2.  **Mention Count (Per Article):** Count the number of times each unique entity is mentioned within *this specific article*.
3.  **Influence Context Flag:** Determine if the entity is portrayed as having agency, influence, or playing a significant role in the events described (`is_influential_context`: true), or if it's mentioned passively or merely as context (`is_influential_context`: false).
4.  **Supporting Snippets:** If `is_influential_context` is true, extract 1-3 brief, direct quotes (max ~25 words each) from the article that best exemplify this influence or agency.

## Constraints & Guidelines:
* Focus solely on the provided text. Do not infer information not present.
* Use the exact entity type categories listed above.
* Group clear variations of the same entity name (e.g., "Federal Reserve", "Fed") under a canonical name if context confirms they are identical, and sum their mentions. If ambiguous, list separately.
* The `is_influential_context` flag should reflect the entity's role *in the narrative* of this article.
* Supporting snippets must be verbatim quotes.
* Be comprehensive but avoid extracting trivial mentions. Focus on entities relevant to the core topic.

## Output Format:
Respond **only** with a single JSON object enclosed in ```json ```. The object must follow this exact structure:

```json
{
  "extracted_entities": [
    {
      "entity_name": "string",  // Canonical name
      "entity_type": "string",  // One of the specified types
      "mention_count_article": integer, // Count within this article
      "is_influential_context": boolean,
      "supporting_snippets": [    // List of strings, empty if not influential
        "string"
      ]
    }
    // ... more entity objects if found
  ]
}
````

## Article Text:

```text
{ARTICLE_CONTENT_HERE}
```

## Analysis Output:

```json

```

**How to Use:** Replace `{ARTICLE_CONTENT_HERE}` with the article's full content. Send this prompt to the Gemini model assigned based on the article's `processing_tier`. Parse the JSON output to feed into Steps 5 and 6 of the Stage D plan (storing results and calculating global influence scores).
