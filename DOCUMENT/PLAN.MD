Okay, let's outline a new comprehensive plan based on the exciting direction you've chosen. This plan starts _after_ Stage D (Entity Extraction & Frame Identification) is complete and adopts the **two-stage AI architecture** (Proactive Deep Analysis Cache -> Frontend RAG + Synthesis) to achieve the deep analytical goals described in `description.txt`, leveraging the powerful `gemini-2.5-pro-exp` model strategically.

We will structure this following the format of the original `bigplan.txt`.

---

## **Fracture Civic Intelligence Platform: Backend Plan**

**Version:** 2.0 (Two-Stage Synthesis Architecture)
**Date:** April 16, 2025

### 1. Introduction: Goals and Revised Strategy Recap

The Fracture platform aims to cultivate critical information literacy by enabling users to trace power, recognize framing, interpret ambiguity, see "sparks," and hold competing truths[cite: 132]. Stages A-D (Ingestion, Preprocessing, Embedding/Clustering, Entity/Frame Extraction) provide the foundational data.

This plan details the subsequent stages, implementing a **sophisticated two-stage AI architecture** designed to deliver deep, multi-perspective analysis efficiently:

1.  **Proactive Backend Analysis:** Leverages an enhanced RAG system and the powerful `gemini-2.5-pro-exp` model to pre-compute a rich cache of diverse analytical snippets covering multiple layers and perspectives for key daily news events.
2.  **Dynamic Frontend Synthesis:** Utilizes a second RAG system and the efficient `gemini-2.0-flash` model to retrieve relevant pre-computed snippets and synthesize tailored analytical responses for users on demand.

This approach concentrates expensive computation offline, creating reusable analytical assets, while providing a responsive and tailored user experience.

### 2. Revised NLP Pipeline (Post-Stage D)

_(Stages A-D are considered complete prerequisites)_

**2.1 Stage E.1: Proactive Deep Analysis Cache Generation (Backend - Daily)**

- **Goal:** Generate a rich, multi-perspective "analysis cache" for the top ~50 news events/summaries of the day.
- **Input:** Top ~50 articles/summaries ranked by `Combined_Priority_Score` (from Stage D); access to `articles`, `entities`, `article_entities`, `entity_snippets`, `clusters`, `embeddings`, `calculated_domain_goodness` tables.
- **Process:**
  1.  **Enhanced RAG Context Retrieval:** For each of the ~50 items:
      - Retrieve context using hybrid strategy: Embeddings (vector search on `embeddings`), Entities (query `entities`, `article_entities`, `entity_snippets` for history/influence), Clusters (vector search on `clusters`, retrieve metadata/related articles), Structured Data (query `articles.frame_phrases`, `articles.is_hot`).
      - _(Optional `flash`/local step):_ Potentially use `gemini-2.0-flash` or local NLP to pre-summarize/synthesize RAG results into a concise context package if raw retrieval is too verbose.
      - Fuse and re-rank retrieved context for relevance and conciseness.
  2.  **Multi-Analysis LLM Generation:** For each of the ~50 items:
      - Loop through 5 conceptual Layers (L1-L5).
      - Loop through 5 Analysis Variations per layer (using distinct prompts targeting different lenses: Realism, Economic, Constructivist, Ambiguity types, etc.).
      - Make **1 call to `gemini-2.5-pro-exp`** for each Layer/Variation combination (25 calls per source item).
      - Provide the rich RAG context + the specific Layer/Variation prompt.
- **Output:** ~1250 distinct analytical text snippets (~100-150 words each).
- **Storage:**
  - Store each snippet in a new `analysis_snippets` table (or enhanced `essays`) with metadata: `source_article_id`, `layer`, `variation_index`, `analytical_lens`, `content`.
  - Generate an embedding (using `text-embedding-004`) for each snippet's `content` and store it alongside the snippet for frontend RAG retrieval.
- **Cost:** ~1250 `gemini-2.5-pro-exp` calls daily.

**2.2 Stage E.2: News Feed Spark Generation (Backend - Daily/4hr Cycle using `flash`)**

- **Goal:** Efficiently generate the ~20-50 daily concise News Feed paragraphs with "sparks" for the frontend "Recognize" view.
- **Input:** Top ~20-50 daily articles/summaries (can overlap with E.1), concise locally generated summaries, key entities, frame phrases.
- **Process:**
  - Use **batch processing** with `gemini-2.0-flash`. Prepare batched input containing data for N articles.
  - Make 1-few `flash` calls per 4-hour cycle using a prompt optimized for generating brief summaries + sparks for each item in the batch.
- **Output:** ~N News Feed paragraphs per cycle.
- **Storage:** Store in `essays` table (`type='news_feed'`), link to `articles`, populate `tags`, link `essay_entities`.

**2.3 Stage F: Frontend RAG & Synthesis Layer (Frontend/Backend - On Demand)**

- **Goal:** Deliver relevant, coherent, tailored analysis to the user based on their interaction.
- **Trigger:** User interacts with a news item in the UI (e.g., clicks "Interpret", asks a question).
- **Process:**
  1.  **Frontend RAG Retrieval:**
      - Generate embedding for user query/context.
      - Search the `analysis_snippets` table using `pgvector` (semantic search) and potentially metadata filters (layer, lens).
      - Retrieve Top P relevant pre-computed analysis snippets.
  2.  **AI Synthesis:**
      - Make **1 call to `gemini-2.0-flash`**.
      - Provide user query/context + retrieved snippets.
      - Use a synthesis-focused prompt (e.g., "Synthesize these points about [topic] into a concise response: [snippets]").
- **Output:** Synthesized text displayed to the user.

**2.4 Stage G: Data Management & Cleanup (Ongoing)**

- **Goal:** Maintain data freshness and manage storage.
- **Process:**
  - Periodically run the `domain_goodness_score` calculation.
  - Run scheduled jobs to purge old generated content (e.g., `essays` and `analysis_snippets` older than 24-48 hours).
  - Manage overall data retention for `articles`, `entities`, `embeddings` etc. based on policy.
  - Monitor job success and data quality.

### 3. Technology Stack Summary (Updated)

- **Premium AI:** `gemini-2.5-pro-exp` (Stage E.1 Deep Analysis Generation).
- **Standard AI:** `gemini-2.0-flash` (Stage E.2 News Feed, Stage F Synthesis, potential RAG context prep).
- **Embedding AI:** `text-embedding-004` (or equivalent for articles and analysis snippets).
- **Local NLP:** spaCy (preprocessing), Hugging Face Transformers (local summaries for input prep, ambiguity signal detection, fallback tasks).
- **Database:** PostgreSQL + `pgvector` extension (critical for both article embeddings and snippet embeddings).
- **RAG Systems:**
  - Backend: Hybrid retrieval logic (embedding + entity + cluster + structured), fusion/ranking.
  - Frontend: Snippet retrieval logic (vector search + metadata filtering).
- **Backend Framework:** Python (FastAPI, Django, Flask).
- **Task Queue:** Celery + Redis/RabbitMQ (or cloud equivalent for managing Stage E.1/E.2 generation).
- **Infrastructure:** Scalable Cloud (GCP recommended), Containerization (Docker, Kubernetes). Requires significant compute capacity for Stage E.1.
- **Monitoring/Logging:** Prometheus/Grafana, Cloud Monitoring/Logging, Sentry, etc.

### 4. Implementation Details & Considerations (Updated)

- **RAG Implementation:** Design and optimize both the complex backend hybrid RAG and the fast frontend snippet RAG. Requires efficient querying and potentially caching.
- **`analysis_snippets` Table:** Design schema carefully (as discussed in Stage E.1), ensure efficient embedding storage and indexing (HNSW recommended for `pgvector`).
- **Prompt Engineering & Management:** Develop, test, and manage the large set of prompts: ~25 variations for Stage E.1, synthesis prompts for Stage F, News Feed prompt for Stage E.2.
- **Workflow Orchestration:** Manage the daily pipeline: Stage D completion -> Trigger Stage E.1 & E.2 -> Snippet Embedding -> Ready for Stage F. Handle dependencies and failures.
- **Cost Management:** Critical to monitor and potentially optimize the ~1250 daily `gemini-2.5-pro-exp` calls. Explore reserved instances or other cost-saving measures if applicable.
- **Frontend Synthesis UI/UX:** Design intuitive ways for users to interact and receive synthesized information without feeling overwhelmed by the underlying complexity or potential contradictions in retrieved snippets. Progressive disclosure might be key.
- **Scalability:** Ensure the database, RAG systems, and API calling infrastructure can handle the daily load, especially the burst of activity during proactive analysis generation.

### 5. Challenges and Mitigations (Updated Focus)

- **Cost:** The cost of ~1250 daily premium LLM calls is the most significant challenge. Mitigation: Aggressive monitoring, explore optimizations (e.g., prompt tuning to reduce snippet length slightly?), secure funding.
- **Complexity:** Managing the two-stage AI pipeline, dual RAG systems, prompt variations, snippet lifecycle, and frontend synthesis logic is highly complex. Mitigation: Modular design, robust testing, clear documentation, start with simpler versions of RAG/synthesis and iterate.
- **Infrastructure Scale & Performance:** Handling the backend generation load and ensuring fast frontend RAG + synthesis requires powerful infrastructure. Mitigation: Optimize database queries, use appropriate compute resources, potentially asynchronous frontend loading.
- **Quality Control & Cohesion:** Ensuring the 1250 snippets are high quality/diverse and that the synthesized frontend output is coherent and accurate. Mitigation: Prompt refinement, potentially add automated quality checks on snippets, fine-tune synthesis prompts.
- **Latency:** Frontend RAG + Synthesis introduces latency compared to static content. Mitigation: Optimize retrieval/synthesis steps, use faster models (`flash`), manage user expectations.

### 6. Conclusion

This revised plan outlines an ambitious but potentially powerful direction for Fracture. By separating the deep, multi-perspective analysis generation (using `gemini-2.5-pro-exp` proactively) from the user-facing interaction (using RAG and `gemini-2.0-flash` for tailored synthesis), the platform aims to deliver unparalleled analytical depth and nuance aligned with its core mission[cite: 132]. Success hinges on managing the significant cost and complexity inherent in this two-stage architecture, particularly the large-scale proactive analysis generation and the sophisticated RAG systems. If feasible, this approach could truly differentiate Fracture as a next-generation tool for critical information literacy.
