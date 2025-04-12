Okay, here is a comprehensive document detailing the project based on all the information gathered from the "Fracture" platform report (`asd.txt`), the Gemini API technical evaluation (`Gemini News Analysis Research Project_.pdf`), and the database/data flow specifications (`database.txt`).

---

## Comprehensive Project Documentation: Fracture Civic Intelligence Platform

**Version:** 1.0 (April 11, 2025)
**Compiled From:** `Gemini News Analysis Research Project_.pdf`, `asd.txt`, `database.txt`

### 1. Project Overview & Goals

**Project Name:** Fracture – A Civic Intelligence Platform [cite: 250]

**Context:** The platform operates in an environment characterized by epistemic fragmentation, political theater, institutional mistrust, narrative warfare, and AI-driven misinformation[cite: 250]. It aims to exist "in the cracks between competing truths"[cite: 250].

**Core Purpose (Rearticulated):** Fracture is not merely a news feed; it's an educational simulator and civic tool designed to train users in critical information literacy amidst a complex and often misleading information landscape[cite: 252, 256, 260]. Its primary goals are to enable users to:

- **Trace Power:** Map influence across entities, time, and language[cite: 126, 257].
- **Recognize Framing:** Decode narrative tactics, ideological biases, and identify the presence or absence of specific frames[cite: 126, 135, 258].
- **Interpret Ambiguity:** Develop the capacity to form meaning under conditions of uncertainty and contradiction[cite: 126, 258].
- **See Sparks:** Identify and focus on the significant twists, contradictions, omissions, or reveals within information flow, rather than just surface-level headlines[cite: 129, 133, 145, 255, 259].
- **Hold Competing Truths:** Equip users to navigate multiperspectival realities with clarity, avoiding cynicism[cite: 129, 255, 259].

**Target Focus Areas:** Initial focus includes China-US trade, Vietnam-US trade, US political economy, global influences, and key figures like Jack Ma and Pham Nhat Vuong[cite: 254].

**Overall Vision:** To reshape how legitimacy is perceived by revealing the construction of belief, resisting co-option, and mapping society's narrative evolution, ultimately serving as an archive, dataset, and training ground for post-partisan literacy[cite: 291, 292].

---

### 2. Core Concepts & Platform Philosophy

Fracture is built upon several key design philosophies and user experience concepts:

- **Dual Interface:** Offers two primary modes of interaction[cite: 253]:
  - **Scroll Down (Recognize):** A quick scan (20-30 paragraphs daily) for identifying events, sources, and present/absent frames[cite: 135, 261, 262]. Designed for rapid recognition (~1 minute per paragraph)[cite: 280]. Summaries are factual but contain a "spark" – a twist, question, or omission – to prompt deeper investigation[cite: 133, 145, 264, 274, 281].
  - **Scroll Right (Interpret - "The Rabbit Hole"):** A guided, layered journey into meaning-making, exploring context, power dynamics, correlations, underlying currents, and counter-narratives[cite: 139, 265]. This involves progressing through distinct analytical layers [cite: 139, 266-270, 282]. Designed for variable engagement (5-15+ minutes)[cite: 282].
- **Drama as Clarity:** Uses "sparks" (contradictions, reveals, unanswered questions) within content not for clickbait, but to draw users into the productive friction between narratives, fostering insight[cite: 145, 271, 293].
- **Multiperspectivalism:** Rejects neutrality in favor of structured disagreement[cite: 146, 272, 293]. Presents information through multiple, often competing, lenses (e.g., tagged with different political theories or institutional frames)[cite: 146, 272, 277]. Includes cues for certainty levels and counterfactual prompts[cite: 272].
- **Theory as Canvas:** Explicitly uses concepts from political and social theory (e.g., Realism, Liberalism, sovereignty, elite capture) to frame events, turning the platform into a real-world simulator for applying these concepts[cite: 147, 270, 273, 277, 294].
- **Data-Driven:** Grounds summaries and analyses in verifiable data points, sourced from news articles, official statistics (e.g., WTO, US ITC), research papers, and the platform's own generated insights (e.g., cluster analysis, entity influence scores)[cite: 148, 254, 263, 266, 267, 270, 274, 276, 294]. Data points are often presented clearly in brackets[cite: 278, 104].

---

### 3. Technical Architecture & Data Flow

The system comprises two interconnected projects: **Scraper** and **Reader**.

**3.1 High-Level Components:**

- **Scraper Project:** Responsible for the daily collection of thousands of raw news articles from various online sources[cite: 4, 35, 36]. Focuses on robust data ingestion and basic metadata storage[cite: 6].
- **Reader Project:** Responsible for processing, analyzing, and synthesizing the raw data collected by the Scraper[cite: 7, 35, 36]. It leverages AI (specifically Gemini API models) and a structured database to extract insights, cluster information, and generate the layered content for the Fracture interface[cite: 7, 18, 26].

**3.2 Technology Stack:**

- **Primary AI Models:** Google Gemini API (Free Tier initially)[cite: 6, 130].
  - **Embeddings:** `gemini-embedding-exp-03-07` (or successor) noted for state-of-the-art performance, high dimensionality (e.g., 3072D, though schema uses 768D - needs alignment), and specific task types (`CLUSTERING`, `SEMANTIC_SIMILARITY`)[cite: 9, 14, 15, 16, 42, 43, 65, 67, 95, 96]. Matryoshka Representation Learning (MRL) allows for dimensionality reduction if needed[cite: 19].
  - **Generation/Reasoning:** Gemini 1.5 Flash / 2.0 Flash considered for large context windows (1M tokens), speed, and multimodal potential (though current focus is text)[cite: 10, 11, 60, 158, 283]. Potentially uses X-AI 1M token model mentioned in `asd.txt`[cite: 256, 283].
- **Database:** PostgreSQL[cite: 55, 256].
  - **Vector Support:** `pgvector` extension required for storing and efficiently querying high-dimensional `VECTOR` data types used for embeddings and cluster centroids[cite: 20, 23, 68].
- **Infrastructure:** Likely cloud-based (e.g., GCP mentioned in PDF) for scalability, using services like Cloud Functions or Cloud Run[cite: 35].
- **Frontend:** Infinite-scrolling interface (vertical for Recognize, horizontal for Interpret)[cite: 135, 139, 256, 284].

**3.3 Detailed Data Flow:**

1.  **Ingestion (Scraper):**
    - Scraper fetches articles, storing raw HTML/text content, URL, source domain, publication date, title, and scraping metadata (`proceeding_status`, timestamps) into `scraper.articles`[cite: 4, 5, 37, 38]. URLs are unique constraints[cite: 5, 37].
2.  **Processing Trigger (Reader):**
    - Reader identifies new, successfully scraped articles in `scraper.articles` (e.g., `proceeding_status = 'Success'`).
3.  **Article Refinement (Reader):**
    - Reader retrieves essential data (`url`, `title`, `content`, `pub_date`, `domain`) for a scraped article.
    - Creates a new record in `reader.articles`, linking it via `scraper_id`[cite: 8, 40, 56, 58]. Content may be cleaned or standardized here.
4.  **Embedding Generation (Reader):**
    - The cleaned `content` (and potentially `title`) of the article in `reader.articles` is sent to the Gemini Embedding API[cite: 12, 18, 24, 65].
    - The task type is specified (e.g., `CLUSTERING`)[cite: 15, 42].
    - The resulting vector embedding is stored in `reader.embeddings`, linked via `article_id`[cite: 12, 48, 66].
5.  **Entity Extraction & Scoring (Reader):**
    - AI analyzes the article content (`reader.articles`) to identify relevant entities (`Person`, `Organization`, etc.)[cite: 10, 17]. (See Prompt 1 below).
    - New entities are added to `reader.entities`; existing entities have `mentions` incremented and `last_seen` updated[cite: 10, 43, 44, 60].
    - Links between the article and its entities are created in `reader.article_entities`, storing the `mention_count` for that specific article[cite: 11, 46, 62, 63].
    - Contextual phrases around mentions are extracted (See Prompt 2 below) to potentially inform the `influence_score` calculation[cite: 10, 27, 44, 60]. The score itself might be calculated algorithmically based on mentions, context factors, and potentially entity type or other metrics.
6.  **Clustering (Reader):**
    - Periodically (e.g., daily), clustering algorithms (like KMeans) run on the embeddings stored in `reader.embeddings`[cite: 13, 18, 25, 38, 71].
    - New clusters are created in `reader.clusters`, storing the `centroid` vector and `article_count`[cite: 13, 49, 50, 70].
    - Top clusters (e.g., top 10 or representing top 20-30% of articles) are flagged `is_hot = TRUE`[cite: 9, 13, 50, 57, 70, 71].
    - Each article in `reader.articles` is updated with its assigned `cluster_id`[cite: 9, 42, 56, 72]. The article's `is_hot` flag may be set based on its cluster's status[cite: 9, 42, 56, 57].
7.  **Cluster Interpretation (Reader - Optional AI Step):**
    - AI can optionally analyze articles within a hot cluster to suggest themes/names and identify dominant framing (See Prompt 4 below). This can inform manual review or automated tagging.
8.  **Content Generation - Essays (Reader):**
    - AI uses the comprehensive data (processed article, entities, cluster info, embeddings, external data) to generate content for the Fracture interface (See Prompt 5 below)[cite: 14, 19, 26, 79, 82].
    - **News Feed Paragraphs:** 20-30 concise summaries with "sparks" generated daily, stored in `reader.essays` (`type='news_feed'`)[cite: 14, 19, 51, 73, 74, 86, 91].
    - **Rabbit Hole Layers:** 5 distinct layers (Recap, Theories, Correlations, Angles, Opinions) generated for relevant articles, stored as separate entries in `reader.essays` (`type='rabbit_hole'`, `layer_depth=1-5`) [cite: 14, 19, 51-53, 73, 74, 94, 106]. Up to ~740 layers generated daily[cite: 19, 74].
9.  **Linking Essays (Reader):**
    - Generated essays are linked to their source `article_id` in `reader.essays`[cite: 52, 73].
    - Relevant topic and framing `tags` are added to `reader.essays.tags`[cite: 52, 73, 75].
    - Key entities mentioned within each essay are linked via the `reader.essay_entities` junction table[cite: 15, 53, 54, 76, 77, 78, 120].

---

### 4. Database Schema (PostgreSQL with pgvector)

**4.1 Scraper Database Schema:**

```sql
-- Stores raw scraped article data and metadata
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    proceeding_status TEXT NOT NULL DEFAULT 'Pending', -- Tracks scraping status (e.g., Pending, Success, Error)
    url TEXT UNIQUE NOT NULL, -- Source URL, unique identifier for raw article
    domain TEXT, -- Source website domain (e.g., nytimes.com)
    title TEXT, -- Original article title
    content TEXT, -- Raw scraped content (HTML or extracted text)
    pub_date TIMESTAMP, -- Original publication date, if available
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the record was created in this table
    scraped_at TIMESTAMP, -- When the scraping attempt occurred
    error_message TEXT -- Stores error details if scraping failed
); [cite: 5, 37, 38]
```

**4.2 Reader Database Schema:**

```sql
-- 1. Processed Articles Table
-- Stores cleaned-up articles linked to the scraper, ready for analysis
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    scraper_id INTEGER UNIQUE, -- Foreign key linking to scraper.articles.id [cite: 8, 40, 56, 58]
    title TEXT NOT NULL, -- Processed/cleaned title
    content TEXT NOT NULL, -- Processed/cleaned main content
    pub_date TIMESTAMP, -- Original publication date
    domain TEXT, -- Source domain
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the article was processed by the Reader
    is_hot BOOLEAN DEFAULT FALSE, -- Flag for articles in top clusters (e.g., top 20-30%) [cite: 9, 42, 56, 57]
    cluster_id INTEGER -- Foreign key linking to the clusters table [cite: 9, 42, 56, 72]
); [cite: 8, 9, 40-42, 56]

-- 2. Entities Table
-- Stores unique power players (people, orgs, etc.) and their influence metrics
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL, -- Name of the entity (e.g., "Jack Ma")
    type TEXT NOT NULL, -- Type (e.g., "Person", "Organization", "Government") [cite: 44, 59]
    influence_score FLOAT DEFAULT 0.0, -- Calculated score reflecting importance/power [cite: 10, 44, 60]
    mentions INTEGER DEFAULT 0, -- Total number of mentions across all articles [cite: 10, 44, 59]
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Timestamp when entity was first identified
    last_seen TIMESTAMP, -- Timestamp when entity was last mentioned/seen [cite: 45, 61]
    UNIQUE (name, type) -- Ensure unique entities based on name and type [cite: 45, 59]
); [cite: 10, 43-45, 59, 60]

-- 3. Article-Entities Junction Table
-- Maps the many-to-many relationship between articles and entities
CREATE TABLE IF NOT EXISTS article_entities (
    article_id INTEGER REFERENCES articles(id), -- Foreign key to reader.articles
    entity_id INTEGER REFERENCES entities(id), -- Foreign key to reader.entities
    mention_count INTEGER DEFAULT 1, -- How many times this entity was mentioned in this specific article [cite: 11, 46, 62, 63]
    PRIMARY KEY (article_id, entity_id) -- Composite primary key [cite: 46, 47, 62]
); [cite: 11, 46, 47, 62]

-- 4. Embeddings Table
-- Stores AI-generated vector embeddings for articles
-- Requires pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    article_id INTEGER UNIQUE REFERENCES articles(id), -- Link to the specific article [cite: 12, 48, 66]
    embedding VECTOR(768), -- Stores the embedding vector (dimension 768 shown, align with chosen model) [cite: 12, 23, 48, 66, 67]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP -- When the embedding was generated
); [cite: 12, 48, 49, 65, 66]

-- 5. Clusters Table
-- Stores information about groups of semantically similar articles
CREATE TABLE IF NOT EXISTS clusters (
    id SERIAL PRIMARY KEY,
    centroid VECTOR(768), -- Average embedding vector of articles in the cluster [cite: 13, 49, 70]
    article_count INTEGER DEFAULT 0, -- Number of articles currently in this cluster [cite: 13, 50, 70]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the cluster was first formed/updated
    is_hot BOOLEAN DEFAULT FALSE -- Flag for top clusters by size or significance [cite: 13, 50, 70, 71]
); [cite: 13, 49, 50, 69, 70]

-- 6. Essays Table
-- Stores the generated content for the Fracture UI (News Feed paragraphs and Rabbit Hole layers)
CREATE TABLE IF NOT EXISTS essays (
    id SERIAL PRIMARY KEY,
    type TEXT NOT NULL, -- Type of content: 'news_feed' or 'rabbit_hole' [cite: 14, 51, 73, 74]
    article_id INTEGER REFERENCES articles(id), -- Link to the source article (can be NULL for analytical essays?) [cite: 14, 52, 73]
    content TEXT NOT NULL, -- The actual generated text content
    layer_depth INTEGER, -- For 'rabbit_hole': 1 (Recap) to 5 (Opinions) [cite: 14, 52, 73, 74]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- When the essay/layer was generated
    tags TEXT[] -- Array for storing topic tags, framing tags, etc. [cite: 14, 52, 73, 75]
); [cite: 14, 51-53, 72, 73]

-- 7. Essay-Entities Junction Table
-- Maps the many-to-many relationship between generated essays and entities mentioned within them
CREATE TABLE IF NOT EXISTS essay_entities (
    essay_id INTEGER REFERENCES essays(id), -- Foreign key to reader.essays
    entity_id INTEGER REFERENCES entities(id), -- Foreign key to reader.entities
    PRIMARY KEY (essay_id, entity_id) -- Composite primary key [cite: 15, 54, 77]
); [cite: 15, 53, 54, 76, 77]

```

---

### 5. AI Analysis Pipeline & Prompts

The Reader project employs a sequence of AI tasks to process data and generate insights.

**Sequence of AI Tasks:**

1.  **Embedding Generation:** Create vector representation of article content.
2.  **Entity Extraction:** Identify key actors (people, orgs, etc.).
3.  **Entity Influence Context Extraction:** Gather contextual clues for scoring.
4.  **(Optional) Cluster Interpretation:** Analyze content clusters for themes/framing.
5.  **Content Generation (Essays):** Create News Feed paragraphs and Rabbit Hole layers.

**Detailed AI Prompts:**

**Prompt 1: Extract and Categorize Entities (Power Players)**

- **Goal:** Identify key entities relevant to tracing power within an article.
- **Input:** `article_id`, `article_content`, `article_title`, `domain`.
- **Instructions:** Analyze text to find people, organizations, locations, government bodies, key concepts acting as agents. Focus on relevance to power/influence. Assign types (`Person`, `Organization`, `Government`, `Location`, `Political Concept`, `Economic Concept`, `Technology`, `Geopolitical Entity`, `Other`). Count mentions (`mention_count`). Flag if mentioned in an influential context (`is_influential_context`).
- **Output:** List of entities [{`entity_name`, `entity_type`, `mention_count`, `is_influential_context`}] for the article.

**Prompt 2: Extract Context for Entity Influence Score**

- **Goal:** Extract phrases providing qualitative context for entity influence.
- **Input:** `article_id`, `article_content`, `list_of_entities` (from Prompt 1).
- **Instructions:** For each entity, find mentions in content. Extract surrounding sentence/phrase (max 30 words). Prioritize context indicating financial scale, decision power, network position, impact, or political action.
- **Output:** Dictionary `{ entity_id_or_tuple: [list_of_contextual_strings] }`.

**Guidance 3: Generating Text Embeddings (API Call Configuration)**

- **Goal:** Generate high-quality embeddings optimized for clustering and similarity.
- **Tool:** Gemini Embedding API (e.g., `gemini-embedding-exp-03-07`).
- **Configuration:**
  - **Input:** `reader.articles.content` (handle 8K token limit).
  - **Task Type:** Primarily `CLUSTERING`; consider `SEMANTIC_SIMILARITY` for relationship analysis[cite: 15, 42, 43].
  - **Model:** Recommended Gemini embedding model[cite: 9, 14].
  - **Dimensionality:** Align with chosen model and `reader.embeddings.embedding` schema definition (e.g., 768 or 3072)[cite: 17, 19].
- **Output:** Vector stored in `reader.embeddings.embedding`.

**Prompt 4: Interpret Cluster Content and Suggest Theme/Name**

- **Goal:** Propose themes/names and identify framing for an article cluster.
- **Input:** `cluster_id`, `article_summaries`/`titles` (top N in cluster), `top_entities` (in cluster).
- **Instructions:** Review inputs to find the central theme/event. Suggest 2-3 concise `cluster_theme` names (max 10 words). Identify dominant narrative `framing` (e.g., "Economic Opportunity", "National Security"). Note any `ambiguity` or conflicting views.
- **Output:** Dictionary `{ suggested_themes: [], dominant_framing: [], observed_ambiguity: "" }`.

**Prompt 5: Enhanced Content Generation for Fracture (News Feed & Rabbit Hole)**

- **Goal:** Generate structured content for Fracture UI, integrating platform philosophy.
- **Core Philosophy Reminder:** Trace Power, Recognize Framing, Interpret Ambiguity, See Sparks, Hold Competing Truths[cite: 126, 129].
- **Input:** `reader.articles` record, related `entities`, `cluster` info, potentially similar article IDs via `embeddings`, external web search access.
- **Instructions Part A (News Feed - `essays.type='news_feed'`):**
  1.  Summarize article core (2-3 sentences, ~50-70 words).
  2.  Integrate bracketed data point[cite: 148].
  3.  Mention key `entity` + hint of role/influence.
  4.  End with a "spark" (question, omission, contradiction)[cite: 133, 145].
  5.  Identify 1-2 primary narrative frames.
  6.  **Output:** `content` (paragraph), `tags` (topic + frame tags).
- **Instructions Part B (Rabbit Hole - `essays.type='rabbit_hole'`, `layer_depth=1-5`):**
  1.  **Layer 1 (Recap):** Event + immediate context (~100 words)[cite: 96].
  2.  **Layer 2 (Theories):** 2-3 plausible interpretations with tagged political/economic theories (~150 words)[cite: 97, 98, 147, 277].
  3.  **Layer 3 (Correlations):** Patterns via data, entity links, embedding similarity, external trends (~150 words)[cite: 99].
  4.  **Layer 4 (Angles/Ambiguities):** Speculative questions, highlight contradictions/gaps, potential hidden agendas (~150 words)[cite: 100, 101].
  5.  **Layer 5 (Opinions/Counter-Narratives):** Challenging perspective, niche academic/expert takes, critique of mainstream (~150 words)[cite: 102, 103].
- **Output Storage:** Store each piece in `essays` table with correct `type`, `layer_depth`, `article_id` link. Populate `essays.tags`. Link entities mentioned via `essay_entities`.
- **Success Criteria:** Reflect Fracture principles, leverage database context, encourage critical thinking.

---

### 6. Gemini API Technical Considerations

_(Summarized from `Gemini News Analysis Research Project_.pdf`)_

- **Relevant Models:**
  - `gemini-embedding-exp-03-07`: State-of-the-art for text embeddings, crucial for clustering/similarity, high dimensionality (3072D), 8K token input limit, MRL support[cite: 9, 14, 15, 16, 17, 18, 19, 33, 95]. _Experimental status requires monitoring_[cite: 96].
  - Gemini 1.5 Flash / 2.0 Flash: Fast, versatile, large 1M token context window, suitable for generation, reasoning, potentially analyzing multiple articles or long documents[cite: 10, 11, 12].
- **Capabilities:** Strong semantic embedding for clustering news[cite: 9, 15, 16]. Large context windows useful for complex analysis[cite: 11, 12]. Multimodal capabilities exist but are not the primary focus here[cite: 6, 10].
- **Limitations (Free Tier):**
  - **Rate Limits:** Critical constraint, especially for embeddings (`gemini-embedding-exp-03-07`: 5 RPM / 100 RPD)[cite: 30, 31, 73, 76, 77, 95]. Processing thousands of daily articles requires efficient batching or upgrading[cite: 26, 31, 77, 95]. Generative models (Flash) have higher limits (15 RPM / 1,500 RPD)[cite: 73, 76].
  - **Token Limits:** 8K input limit for the embedding model needs handling for long articles (truncation/segmentation)[cite: 18, 33, 77].
- **Scalability & Cost:** Free tier limits likely insufficient for real-time processing of thousands of articles daily[cite: 77, 95]. Scaling requires moving to paid tiers, incurring costs based on token usage (input/output)[cite: 79, 80]. Cost estimation depends heavily on article length and API call volume[cite: 80]. Optimization is key if scaling[cite: 80].
- **Alternatives:** Open-source libraries (spaCy, NLTK, Gensim, Flair) and platforms (Hugging Face) offer alternatives[cite: 85, 86, 87]. These provide flexibility and avoid rate limits but may require more setup, lack the same embedding quality, or have different strengths (e.g., spaCy strong in NER)[cite: 88, 90, 91, 92, 93].

---

### 7. Development Strategy

The recommended approach involves iterative development:

1.  **Phase 1: Foundational MVP:** Focus on core data ingestion (Scraper -> Reader basic link), embedding generation (`gemini-embedding-exp-03-07`), and basic clustering (KMeans/DBSCAN). Test embedding quality and manage API limits.
2.  **Phase 2: Enhance Interpretation:** Add entity extraction/scoring, basic relationship analysis, and initial "Scroll Right" layer generation (e.g., Recap, Theories). Integrate alternative NLP tools (e.g., spaCy for NER) if needed.
3.  **Phase 3: Advanced Analysis & UI:** Implement full "Scroll Right" layers, complex reasoning, hypothesis generation, framing analysis, cross-referencing, and user interaction features. Refine UI based on Fracture concepts.

---

### 8. Success Metrics

Success is measured not just by usage, but by the platform's impact on users' cognitive abilities and civic engagement[cite: 169, 256, 285].

- **Core Axes:**
  - **Cognitive Shift:** Increased skepticism (not cynicism), ability to distinguish fact/speculation, capacity to hold competing truths[cite: 170, 285, 296].
  - **Narrative Recognition:** Identifying cross-domain patterns, recognizing framing tactics, tracing influence[cite: 171, 286, 297, 298].
  - **Civic Empowerment:** Increased confidence, contributing interpretations, sharing nuanced takes[cite: 172, 286, 299, 300].
- **Measurable Indicators (Examples):**
  - **Depth:** % engaging 3+ Rabbit Hole layers, time spent in Interpret view[cite: 175, 287, 301].
  - **Literacy:** Use of cluster navigation, quiz accuracy on related topics[cite: 176, 287, 301].
  - **Framing:** Engagement with opposing theory tags, user submissions of angles/flags[cite: 177, 288, 302, 303].
  - **Civic:** Use of input features ("Disagree?", "Suggest Angle"), glossary engagement[cite: 178, 288, 303].
- **Strategic Metrics:** Institutional adoption (education, media), cross-ideological trust scores, integrity of shared "sparks" linked to full interpretation [cite: 179-181, 289, 304-306].
- **Resilience Metrics:** Theory Recognition Index (longitudinal user understanding), Civic Spillover (reported changes in media habits), handling of misuse cases, Stanford Test (qualitative assessment by critical thinkers) [cite: 181, 290, 291, 307-311].

---

This document synthesizes the available information to provide a detailed blueprint for the Fracture project. Further refinement may be needed as development progresses and specific implementation challenges arise.
