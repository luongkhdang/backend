Okay, this document presents a revised and highly detailed plan for the Fracture Civic Intelligence Platform's backend, specifically optimized to fully utilize a combination of free NLP tools (spaCy, NLTK, Hugging Face Transformers run locally) alongside the free tiers of the Gemini API, paying close attention to the rate limits specified in `rate-limit.docx` [cite: 3, 4, 11] and building upon the initial specification in `plan.txt`and the user-provided optimization context.

---

## Optimized Backend Specification & NLP Strategy: Fracture Civic Intelligence Platform

**Version:** 1.1 (Rate-Limit Optimized)
**Date:** April 11, 2025
**Based On:** `plan.txt`[cite: 15], `rate-limit.docx`[cite: 1], `asd.txt`[cite: 126], `Gemini News Analysis Research Project_.pdf`, User Context

### 1. Introduction: Goals and Constraints Recap

The Fracture platform aims to cultivate critical information literacy by enabling users to trace power, recognize framing, interpret ambiguity, see "sparks" in news, and hold competing truths[cite: 126]. This requires processing thousands of news articles daily to generate ~20-30 concise "News Feed" paragraphs and up to ~740 layered "Rabbit Hole" analyses[cite: 34, 36].

A primary constraint is operating within the free tier limits of the Google Gemini API[cite: 4]. This necessitates a hybrid approach, strategically combining Gemini's advanced reasoning and generation capabilities with the high throughput and cost-effectiveness of locally run, free NLP tools like spaCy, NLTK, and models from the Hugging Face ecosystem.

**Key Rate Limits (Free Tier - from `rate-limit.docx` & PDF):**

- **`Gemini 1.5 Flash / 2.0 Flash` (Generation/Reasoning):** 15 RPM / **1500 RPD** / 1M Token Context[cite: 4]. _Primary resource for complex analysis and content generation._
- **`gemini-embedding-exp-03-07` (High-Quality Embedding):** 5 RPM / **100 RPD** / 8192 Token Input[cite: 4, 7]. _Too limited for bulk embedding._
- **`text-embedding-004` (High-Throughput Embedding):** **1500 RPM** / No RPD specified (assumed high) / **2048 Token Input** / 768 Dimensions[cite: 11]. _Suitable for bulk embedding but requires handling the token limit._

**Optimization Goal:** Maximize the volume and depth of analysis possible within these constraints by offloading tasks amenable to local processing, reserving Gemini API calls for tasks requiring its unique capabilities (complex reasoning, nuanced generation, high-quality context analysis).

---

### 2. Optimized NLP Pipeline & Tool Allocation

This revised pipeline integrates Gemini API (free tier), `text-embedding-004`, spaCy, NLTK, and Hugging Face models (run locally) across the different stages of processing.

**2.1 Stage A: Ingestion and Preprocessing (Scraper → Reader)**

- **Goal:** Ingest raw articles, clean HTML/text, structure content for analysis.
- **Primary Tools:**
  - **spaCy:** (Local, Free) - Use for initial high-speed HTML parsing, text extraction, sentence segmentation, and removal of boilerplate content (ads, navigation). Its efficiency is key for handling thousands of articles. Leverage models like `en_core_web_lg` for robust processing.
  - **NLTK:** (Local, Free) - Use for supplementary, niche preprocessing tasks if spaCy's defaults are insufficient. Examples: custom stopword lists tailored to specific domains (e.g., financial jargon, geopolitical terms), potentially advanced tokenization rules for unusual text formats found in specific news sources. Primarily for edge cases or prototyping.
  - **Hugging Face (Transformers):** (Local, Free - Model Dependent) - Optionally use lightweight models (e.g., `distilbert-base-uncased` with custom fine-tuning or rule-based checks) for more complex cleaning tasks where semantic context is needed to identify irrelevant sections missed by spaCy (e.g., distinguishing reader comments from article body). Use judiciously due to higher computational cost than spaCy.
- **Output:** Cleaned `content` and `title` stored in `reader.articles`[cite: 96], ready for embedding and deeper analysis.
- **Rate Limit Impact:** None. All processing is local.
- **Fracture Alignment:** High-quality input data is foundational for accurately tracing power and recognizing subtle framing later in the pipeline.

**2.2 Stage B: Embedding Generation**

- **Goal:** Generate semantic vector embeddings for _all_ processed articles to enable clustering and similarity analysis, respecting token limits.
- **Primary Tool & Strategy:**
  - **`text-embedding-004` (Gemini API - Free Tier):** Use this model for bulk embedding due to its high RPM (1500) [cite: 11] and 768-dimension output aligning with the database schema[cite: 99]. Task type should be set to `CLUSTERING`.
  - **Handling 2048 Token Limit[cite: 11]:**
    - **Method 1 (Preferred): Summarization First:**
      - **Hugging Face (Transformers):** (Local, Free Models) - For articles exceeding 2048 tokens, use a local summarization model _before_ calling the embedding API. Models like `facebook/bart-large-cnn` (higher quality, more compute) or `google/pegasus-xsum` or `t5-small` (faster, potentially less nuance) can summarize long content down to ~1500-2000 tokens. This preserves the core semantic meaning better than simple truncation.
      - **spaCy Fallback:** (Local, Free) - If summarization models are too slow or resource-intensive for the required volume, use spaCy to perform extractive summarization (e.g., selecting top N sentences based on TF-IDF, entity density, or position) to create a shorter text representation under the token limit. Less semantically robust but faster.
    - **Method 2 (Fallback): Chunking & Averaging:**
      - **spaCy:** (Local, Free) - Split articles >2048 tokens into overlapping or sequential chunks based on sentence boundaries.
      - **`text-embedding-004`:** Embed each chunk individually.
      - **Averaging:** Calculate the mean of the resulting chunk embedding vectors. Store this averaged vector. Less accurate than summarizing first but avoids local summarization compute. NLTK might assist with specialized chunking logic if needed.
  - **API Call Management:** Throttle calls to `text-embedding-004` (e.g., sustain ~500 RPM) to avoid overwhelming the API and manage internal system load, even though the limit is 1500 RPM. Implement retry logic.
- **Selective High-Quality Embedding:**
  - **`gemini-embedding-exp-03-07` (Gemini API - Free Tier):** Use its 100 RPD limit [cite: 4] strategically. Embed <100 high-priority articles per day (e.g., identified based on source, topic keywords like 'Jack Ma policy', or initial clustering results indicating high importance) that may benefit from its potentially higher quality, larger 8K token window[cite: 7], or higher native dimensionality (3072D).
  - **Storage:** Store these high-quality embeddings alongside the primary `text-embedding-004` embeddings. This requires schema consideration: either a separate table, a nullable column for the high-quality embedding, or storing both with metadata indicating the source model/dimension. If storing 3072D vectors, use MRL to generate a 768D version for compatibility with the primary clustering if needed[cite: 7].
- **Potential Free Alternative:**
  - **Hugging Face (Sentence Transformers):** (Local, Free Models) - Models like `sentence-transformers/all-MiniLM-L6-v2` or `sentence-transformers/all-mpnet-base-v2` can provide high-quality 384D or 768D embeddings entirely locally, with no rate limits. This is a strong fallback or alternative if `text-embedding-004` access becomes restricted or proves too costly if ever moving beyond free limits. Evaluate quality against `text-embedding-004` for the specific news domain.
- **Output:** Populate `reader.embeddings` with 768D vectors (primarily from `text-embedding-004`, potentially augmented by truncated `gemini-embedding-exp-03-07` or Sentence Transformers)[cite: 99].
- **Rate Limit Impact:** `text-embedding-004` RPM is the main factor, RPD assumed not limiting. `gemini-embedding-exp-03-07` usage is capped at 100 RPD. Summarization/chunking uses local resources only.
- **Fracture Alignment:** Ensures all articles have a semantic representation for clustering (Recognize Framing) and finding correlations (Interpret Ambiguity). Selective high-quality embedding aids precision in Tracing Power for key documents.

**2.3 Stage C: Clustering**

- **Goal:** Group articles based on semantic similarity using the primary (`text-embedding-004`) embeddings.
- **Primary Tools:**
  - **External Algorithm (Local):** Run clustering algorithms like KMeans (if a fixed number of clusters is desired) or HDBSCAN (better for discovering clusters of varying shapes/densities and handling noise) directly on the 768D vectors stored in `reader.embeddings`. This happens locally, typically as a scheduled batch process (e.g., daily).
  - **Database (`pgvector`):** Use `pgvector`'s indexing (e.g., HNSW) to accelerate nearest neighbor searches required by some clustering algorithms or for post-clustering analysis (finding articles near a centroid)[cite: 74, 100, 175].
- **Supporting Tools for Interpretation/Tagging:**
  - **spaCy:** (Local, Free) - Use NLP pipelines (`en_core_web_lg`) to extract key noun phrases or named entities from the titles/summaries of top articles within each cluster to help generate potential cluster names/themes automatically.
  - **Hugging Face (Transformers):** (Local, Free Models) -
    - **Zero-Shot Classification:** Use models like `facebook/bart-large-mnli` or `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` to automatically assign preliminary framing tags (e.g., "National Security", "Economic Opportunity", "Human Rights Concern") to clusters based on the aggregated content of their articles.
    - **Summarization:** Generate a brief summary of the cluster's core topic using models like `google/pegasus-xsum` applied to concatenated snippets from top articles.
  - **Gemini Flash (API - Free Tier - Optional/Limited):** Use Prompt 4 (`Interpret Cluster Content`) sparingly (~10-20 RPD max) to get nuanced thematic names and framing interpretations for only the _most important_ "hot" clusters[cite: 32, 185]. This leverages Gemini's reasoning but conserves the RPD quota.
- **Output:** Update `reader.clusters` (store `centroid`, `article_count`, `is_hot` flag) [cite: 101] and update `reader.articles` (`cluster_id`, `is_hot` flag)[cite: 96]. Potential themes/frames stored possibly in `reader.clusters` metadata or used to generate tags for essays later.
- **Rate Limit Impact:** Minimal direct impact. Only optional use of Gemini Flash (~10-20 RPD) for interpreting top clusters.
- **Fracture Alignment:** Clustering directly supports Recognizing Framing by grouping related narratives. Identifying "hot" clusters helps prioritize analysis for Tracing Power and generating timely interpretations.

**2.4 Stage D: Entity Extraction and Scoring**

- **Goal:** Identify relevant entities, link them to articles, and calculate an influence score, balancing accuracy with API quota cost.
- **Tiered Strategy:** Implement a tiered approach based on article priority (e.g., determined by source credibility, "hot" cluster status, presence of seed keywords).
  - **Tier 1: High Priority (Target: ~150-200 articles/day):**
    - **Gemini 1.5 Flash (API - Free Tier):** Use Prompt 1 (`Extract Entities`) for comprehensive extraction of entities (Person, Org, Govt, Concept, etc.), mention counts, and the `is_influential_context` flag[cite: 43, 59, 80]. Gemini's large context and reasoning are valuable here for identifying nuanced entities and power contexts.
    - **Gemini 1.5 Flash (API - Free Tier):** Use Prompt 2 (`Extract Influence Context`) to get specific phrases supporting influence assessment[cite: 47, 59, 80, 83]. (Can potentially be combined with Prompt 1 in a single API call to save quota).
    - **Hugging Face (Transformers):** (Local, Free Models) - _Supplement_ Gemini NER. Use fine-tuned NER models (e.g., `dslim/bert-base-NER` fine-tuned on relevant news/political corpus, or specialized models like `obi/deid-bert-i2b2` for certain types if needed) to catch domain-specific entities Gemini might miss or misclassify. Run locally for maximum precision on these key articles.
    - **Quota Cost (Gemini):** ~1-2 calls per article = **~150 - 400 RPD**. (Aim for combining prompts to stay lower).
  - **Tier 2: Medium Priority (Target: ~500-1000 articles/day):**
    - **spaCy:** (Local, Free) - Use standard `en_core_web_lg` or `en_core_web_trf` (transformer backend) for fast, efficient extraction of common entity types (PERSON, ORG, GPE, LOC, DATE, etc.). This handles the bulk of articles quickly and cost-effectively.
    - **Hugging Face (Transformers):** (Local, Free Models) - Use robust pre-trained NER models (e.g., `Jean-Baptiste/roberta-large-ner-english`) as a _fallback_ or _secondary check_ if spaCy results seem incomplete for certain articles or entity types (e.g., complex geopolitical concepts not in standard spaCy types).
    - **Context Extraction (Local):** Use spaCy's dependency parsing and sentence extraction to get surrounding context for influence scoring, avoiding API calls.
    - **Quota Cost (Gemini):** **0 RPD**.
  - **Tier 3: Low Priority (Remaining articles):**
    - **spaCy:** (Local, Free) - Use the fastest spaCy pipeline (`en_core_web_sm` or just the tagger/parser if full NER isn't needed) primarily for basic metadata enrichment (tagging locations, dates) if required for filtering, without detailed entity linking/scoring.
    - **NLTK:** (Local, Free) - Only for highly specific, rule-based entity tagging (e.g., finding mentions of specific laws or treaties using regex patterns over tokenized text) if other methods fail.
    - **Quota Cost (Gemini):** **0 RPD**.
- **Influence Score Calculation:**
  - This is primarily an **algorithmic** step performed locally after entity extraction.
  - **Inputs:** `mention_count` (global [cite: 97] and per-article [cite: 98]), `is_influential_context` flag (from Gemini Tier 1) or parsed context (from spaCy Tier 2), entity type, source domain credibility, recency (`first_seen`, `last_seen` [cite: 97]), potentially cluster importance (`is_hot`).
  - **Logic:** Develop a weighted scoring formula combining these factors. Gemini's contextual extraction (Prompt 2) provides qualitative input for the algorithm.
- **Output:** Populate `reader.entities` (upserting, updating `mentions`, `influence_score`, `last_seen`) [cite: 97] and `reader.article_entities`[cite: 98].
- **Rate Limit Impact:** Controlled by the number of Tier 1 articles processed. **~150 - 400 RPD** target for Gemini Flash.
- **Fracture Alignment:** Tiered approach focuses high-cost AI analysis on articles most likely relevant to Tracing Power, using free local tools for broad coverage. Algorithmic scoring ensures consistency.

**2.5 Stage E: Content Generation (News Feed & Rabbit Hole)**

- **Goal:** Generate the ~20-30 daily News Feed paragraphs and up to ~740 Rabbit Hole layers, embedding Fracture's philosophy (sparks, layers, framing, ambiguity).
- **Tiered Strategy (Leveraging Gemini Flash 1500 RPD):**
  - **Tier 1: High Priority Articles (Target: ~100-150 articles receive full Rabbit Hole):**
    - **News Feed Paragraphs (Prompt 5A):**
      - **Gemini 1.5 Flash (API - Free Tier):** Generate the core 20-30 daily News Feed paragraphs using the detailed Prompt 5A, ensuring inclusion of data points, entity links, and the crucial "spark"[cite: 34, 60, 88, 109].
      - **Hugging Face (Transformers):** (Local, Free Models) - Use zero-shot classifiers (`facebook/bart-large-mnli`) or keyword analysis (spaCy) to generate the `tags` (topic and framing) associated with the paragraph, reducing the complexity required from the Gemini prompt.
      - **Quota Cost (Gemini):** ~1 call/paragraph = **~20-30 RPD**.
    - **Rabbit Hole Layers (Prompt 5B):**
      - **Gemini 1.5 Flash (API - Free Tier):** Generate all 5 layers (Recap, Theories, Correlations, Angles, Opinions) for these ~100-150 articles using Prompt 5B[cite: 36, 60, 90, 109]. This is the most quota-intensive step but core to the platform's interpretive depth. Leverage the 1M token context to provide rich input (article text, entities, cluster info, maybe snippets from related articles).
      - **Hugging Face (Transformers):** (Local, Free Models) - _Supplement_ Gemini for specific layers to save quota or add diversity:
        - _Layer 3 (Correlations):_ Use local Sentence Transformer embeddings (`all-MiniLM-L6-v2`) to find semantically similar articles to reference, providing this list as input to the Gemini prompt.
        - _Layer 4/5 (Angles/Opinions):_ Experiment with fine-tuned generative models (like `T5-base` or smaller GPT variants fine-tuned on argumentative/opinion datasets) to generate alternative speculative angles or counter-narratives for a subset (~20%) of these articles, reducing the number of Layer 4/5 calls to Gemini.
      - **Quota Cost (Gemini):** ~5 calls/article x 100-150 articles = **~500 - 750 RPD**. (Supplementing with HF could reduce this).
  - **Tier 2: Medium Priority Articles (Target: ~100-200 articles receive partial Rabbit Hole):**
    - **News Feed Paragraphs:** Generate _only if_ the daily quota of 20-30 isn't met from Tier 1, or for articles in particularly "hot" clusters, using Gemini Flash or a local Hugging Face summarizer (`google/pegasus-xsum`) + spaCy for keyword/frame tagging.
    - **Rabbit Hole Layers:**
      - **Gemini 1.5 Flash (API - Free Tier):** Generate only Layers 1 (Recap) and 2 (Theories) for ~100-200 articles[cite: 37, 90]. This provides initial context and framing without the high cost of deep analysis layers.
      - **spaCy:** (Local, Free) - Can potentially generate a basic Layer 1 (Recap) using extractive summarization to further save quota on some articles.
      - **Quota Cost (Gemini):** ~2 calls/article x 100-200 articles = **~200 - 400 RPD**.
  - **Tier 3: Low Priority Articles (Remaining):**
    - **No AI Content Generation:** These articles rely on their embeddings, cluster assignments, and basic spaCy-extracted entities/tags for visibility and filtering in the UI. No News Feed or Rabbit Hole essays generated to conserve quota.
- **Output:** Populate `reader.essays` table with generated content, correct `type`, `layer_depth`, and populate `reader.essays.tags` and `reader.essay_entities`[cite: 102, 104].
- **Total Estimated Gemini Flash Quota Usage:** (20-30 RPD News Feed) + (500-750 RPD Tier 1 RH) + (200-400 RPD Tier 2 RH) + (150-400 RPD Tier 1 Entities) + (10-20 RPD Cluster Interpretation) = **~880 - 1600 RPD**. _This pushes the 1500 RPD limit._ Requires careful prioritization, potentially reducing the number of articles receiving full/partial Rabbit Holes, aggressively combining entity/context prompts, or relying more heavily on Hugging Face supplementation for Layers 4/5. **Monitoring and dynamic adjustment based on daily quota usage will be critical.**
- **Fracture Alignment:** This tiered strategy focuses the most intensive (and expensive) AI analysis on generating the deep, layered interpretations (Rabbit Hole) and sparking curiosity (News Feed) for the most relevant content, while ensuring broad coverage via local tools.

---

### 3. Technology Stack Summary (Optimized)

- **Primary AI API:** Google Gemini API (Free Tier)
  - `text-embedding-004`: Bulk embeddings.
  - `Gemini 1.5 Flash` / `2.0 Flash`: Tier 1/2 generation, NER, context extraction.
  - `gemini-embedding-exp-03-07`: Selective high-quality embeddings.
- **Local NLP Libraries/Models:**
  - **spaCy:** Core preprocessing, Tier 2/3 NER, dependency parsing, rule-based summarization/tagging. (Models: `en_core_web_sm`, `en_core_web_lg`, `en_core_web_trf`).
  - **Hugging Face Transformers:** Summarization (for embedding token limit handling - e.g., `bart-large-cnn`, `pegasus-xsum`), NER supplementation/fallback (e.g., `bert-base-NER`, `roberta-large-ner`), Zero-Shot Classification (framing tags - e.g., `bart-large-mnli`), potential Layer 4/5 generation (e.g., fine-tuned `T5`), optional local embeddings (`Sentence Transformers`). Requires local compute (CPU/GPU).
  - **NLTK:** Niche preprocessing, prototyping.
- **Database:** PostgreSQL + `pgvector` extension.
- **Backend Framework:** Python (FastAPI, Django, Flask).
- **Task Queue:** Celery + Redis/RabbitMQ or Google Cloud Tasks.
- **Infrastructure:** Cloud-based (GCP recommended), containerized services (Docker, Kubernetes), potentially serverless functions (Cloud Run, Cloud Functions).
- **Monitoring/Logging:** Prometheus/Grafana, Datadog, Google Cloud Monitoring/Logging, Sentry.

---

### 4. Implementation Details & Considerations

- **Task Queue Prioritization:** Implement priority levels in the task queue (e.g., Tier 1 > Tier 2 > Tier 3). Process News Feed generation first daily, then prioritize Tier 1 Rabbit Holes and NER, followed by Tier 2, and finally bulk embeddings.
- **Gemini Quota Monitoring:** Implement real-time (or near real-time) tracking of RPD usage for Gemini Flash. Dynamically adjust the number of Tier 1/Tier 2 articles processed daily based on remaining quota. Have fallback strategies (e.g., skip Layer 5 generation, use local summarizers more) if limits are approached.
- **Local Model Management:** Requires infrastructure to host and run Hugging Face models efficiently (GPU recommended for larger models like BART/T5). Consider model caching and optimized inference frameworks (like ONNX Runtime).
- **Database Optimization:** Ensure proper indexing on foreign keys, `entities.name`, `essays.tags`, and crucially, implement effective `pgvector` indexing (HNSW or IVFFlat) for efficient similarity searches during clustering and potentially for Rabbit Hole Layer 3 generation. Consider partitioning large tables (`articles`, `embeddings`, `essays`) by date range if volume grows significantly.
- **Token Limit Handling Robustness:** Thoroughly test the chosen summarization/chunking strategy for `text-embedding-004` to minimize semantic loss. Log which method was used for each long article.
- **Prompt Engineering:** Continuously refine prompts (especially Prompt 5 for content generation) to improve quality, adherence to Fracture philosophy, and potentially combine tasks (like entity extraction + context extraction) into single, efficient API calls where feasible.
- **Cold Starts:** Be mindful of cold start times if using serverless functions for processing workers, especially for GPU-dependent Hugging Face models.

---

### 5. Challenges and Mitigations (Optimized Strategy)

- **Hitting Gemini Flash 1500 RPD Limit:**
  - **Mitigation:** Aggressively prioritize Tier 1 tasks. Dynamically reduce the number of articles getting full/partial Rabbit Holes. Further offload Layer 4/5 generation to local Hugging Face models. Optimize prompts to combine tasks (e.g., NER + Context). Implement strict monitoring and daily caps.
- **`text-embedding-004` Token Limit Handling Quality:**
  - **Mitigation:** Prioritize summarization (Hugging Face BART) over chunking for better semantic preservation. Evaluate embedding quality downstream (clustering coherence) for summarized vs. chunked articles. Fallback to local Sentence Transformers if quality loss is unacceptable.
- **Local Compute Costs/Complexity (Hugging Face):**
  - **Mitigation:** Use smaller, optimized models (`distilbert`, `t5-small`) where possible. Implement batch inference. Utilize GPU resources efficiently. Start with CPU inference for lighter tasks.
- **Maintaining Consistency Across Tools:**
  - **Mitigation:** Establish clear data schemas. Validate outputs between different tools (e.g., spaCy NER vs. Gemini NER on a sample). Develop a robust algorithmic layer for influence scoring that consistently integrates inputs from different extraction methods.
- **Experimental Gemini Models:**
  - **Mitigation:** Monitor updates/changes to `gemini-embedding-exp-03-07`. Have fallback strategies (e.g., using only `text-embedding-004` or local Sentence Transformers) ready.

---

### 6. Conclusion

This optimized plan provides a detailed roadmap for building the Fracture backend while maximizing the use of free NLP tools and respecting Gemini API free tier constraints. By strategically allocating tasks – using `text-embedding-004` for bulk embedding (with local handling for token limits), spaCy for high-throughput baseline processing, Hugging Face for quality enhancement and specific local tasks (summarization, framing, NER supplementation), and reserving the limited Gemini Flash RPD for high-value reasoning and generation (prioritized tiers for News Feed/Rabbit Hole) – the platform can achieve significant scale and analytical depth. Success hinges on robust implementation of the tiered processing logic, careful quota management, efficient local model deployment, and continuous evaluation of the trade-offs between different tools for specific sub-tasks within the pipeline. This approach directly supports Fracture's mission by enabling the necessary scale and analytical complexity required to trace power, recognize framing, and interpret ambiguity in the modern information ecosystem, all while adhering to pragmatic resource limitations.
