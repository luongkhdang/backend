Okay, here is a reworked and highly detailed set of AI prompts and guidance, tailored to the optimized plan that leverages free local NLP tools (spaCy, NLTK, Hugging Face) alongside the free tiers of the Gemini API (`text-embedding-004`, `Gemini Flash`, `gemini-embedding-exp-03-07`) to operate within rate limits while fulfilling Fracture's complex analytical and philosophical goals.

---

## Reworked Detailed AI Prompts & Guidance (Optimized Plan v1.1)

This section outlines the specific instructions and configurations for the AI models and local NLP tools within the optimized Fracture Reader pipeline.

### 1. Guidance: Text Embedding Generation

**Goal:** Generate semantic vector embeddings for clustering and similarity analysis for _all_ processed articles, managing API limits and token constraints.

**Primary Strategy (Bulk Embedding):**

- **Target Tool:** `text-embedding-004` (Gemini API - Free Tier)
- **Rationale:** High throughput (1500 RPM) allows processing thousands of articles daily. 768-dimension output aligns with the primary `reader.embeddings.embedding` schema field.
- **API Call Configuration:**
  - **Input:** Cleaned `article.content` and potentially `article.title`.
  - **Task Type:** `CLUSTERING` (optimize for grouping related articles).
  - **Token Limit Handling (Input > 2048 Tokens):**
    - **Preferred Pre-processing (Local):** Use a local Hugging Face summarization model (e.g., `facebook/bart-large-cnn` for quality, `google/pegasus-xsum` or `t5-small` for speed) to condense the article text to <2048 tokens _before_ sending to the API. Log that summarization occurred.
    - **Fallback Pre-processing (Local):** If summarization is too slow/resource-intensive, use local spaCy for extractive summarization (e.g., select top N sentences by relevance/entity density) or split the text into overlapping chunks, embed each chunk with `text-embedding-004`, and average the resulting vectors. Log the method used. Less semantically accurate but avoids summarization compute.
  - **Rate Limit Management:** Throttle API calls (e.g., sustain ~500 RPM) using the task queue or AI Interaction Layer to ensure stability. Implement exponential backoff and retry logic for transient API errors.
- **Output:** A 768-dimension `VECTOR` stored in `reader.embeddings.embedding`.

**Selective Strategy (High-Quality Embedding):**

- **Target Tool:** `gemini-embedding-exp-03-07` (Gemini API - Free Tier)
- **Target Articles:** <100 high-priority articles per day (identified by source, keywords, cluster analysis, etc.).
- **Rationale:** Leverage potentially superior embedding quality and larger 8K token input window for critical articles where nuance is paramount (e.g., detailed policy analysis, tracing power in specific documents).
- **API Call Configuration:**
  - **Input:** Cleaned `article.content` and potentially `article.title`. Handles up to 8192 tokens directly.
  - **Task Type:** Consider `SEMANTIC_SIMILARITY` or `RETRIEVAL_DOCUMENT` if the primary use for these specific embeddings is finding nuanced connections or enabling precise search, rather than broad clustering. Use `CLUSTERING` if comparing directly within the main clusters.
  - **Rate Limit Management:** Strictly adhere to the **100 RPD** and 5 RPM limits via the AI Interaction Layer. Queue these tasks with high priority but ensure the daily cap isn't exceeded.
- **Output & Storage:**
  - Generate embedding (native dimension 3072, 1536, or 768).
  - **Schema Consideration:** Store this embedding separately if its dimension or intended use differs significantly from the primary 768D embedding. Options:
    - A separate table `reader.high_quality_embeddings`.
    - A nullable `embedding_hq VECTOR(3072)` column in `reader.embeddings`.
    - Use MRL to generate a 768D version from the 3072D output and store it alongside the primary embedding with metadata indicating its source (`model: gemini-exp-03-07`). This ensures compatibility for some analyses while retaining the option for higher fidelity. **Requires careful schema design and query logic.**

**Alternative Strategy (Fully Local - Fallback):**

- **Target Tool:** Local Hugging Face Sentence Transformers (e.g., `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`).
- **Rationale:** Provides high-quality embeddings (often 384D or 768D) with no API calls, rate limits, or costs (beyond local compute). A strong fallback if API access changes or becomes infeasible.
- **Implementation:** Integrate model loading and inference within the Processing Worker or a dedicated local service. Requires local CPU/GPU resources.
- **Output:** A 384D or 768D vector stored in `reader.embeddings.embedding`.

---

### 2. Prompt: Tier 1 Entity Extraction & Influence Context

**Goal:** Perform high-accuracy entity extraction and identify contextual clues for influence scoring on the most important articles, leveraging Gemini's reasoning capabilities.

- **Target Tool:** `Gemini 1.5 Flash` / `2.0 Flash` (Gemini API - Free Tier)
- **Target Articles:** Tier 1 (High Priority, ~150-200 articles/day).
- **Input:**
  - `article_id`: ID of the processed article.
  - `article_title`: Title from `reader.articles`.
  - `cleaned_article_content`: Processed text content from `reader.articles`.
  - `predefined_entity_types`: List [ "Person", "Organization", "Government", "Location", "Political Concept", "Economic Concept", "Technology", "Geopolitical Entity", "Event", "Policy/Law", "Financial Instrument" ] (Expand as needed).
  - `known_entities` (Optional): List of entities already identified in related articles for consistency.
- **Instructions:**
  "Analyze the provided `article_title` and `cleaned_article_content` for the Fracture platform. Your goal is to identify key entities relevant to tracing power and influence, and extract context related to that influence. Adhere strictly to the output format.

  1.  **Entity Extraction:**

      - Identify all significant named entities (people, organizations, locations, specific policies, events, concepts acting as agents) mentioned in the text.
      - For each unique entity found, assign the most appropriate type from the `predefined_entity_types` list. Be specific (e.g., 'Government Agency' instead of just 'Organization' if possible). If no predefined type fits, use 'Other' and briefly describe.
      - Accurately count the total number of times each unique entity (name-type pair) is mentioned (`mention_count`).
      - Focus especially on entities related to the platform's core topics (China-US trade, Vietnam-US trade, US political economy, global influence, key figures like Jack Ma, Pham Nhat Vuong).

  2.  **Influence Context Extraction:**

      - For **each mention** of each identified entity, examine the immediate surrounding sentence(s).
      - Determine if the context suggests significant influence, action, or power dynamics (e.g., making decisions, controlling significant assets, lobbying, being the subject of major policy, causing notable impact).
      - If such context exists, extract the **key phrase or clause** (max ~25 words) that best demonstrates this influence. Examples: 'announced a $50B merger', 'lobbied against the proposed regulation', 'controls 40% of the regional market', 'whose policy affects millions', 'signed the landmark treaty', 'faces investigation over...'.
      - Do **not** extract generic mentions (e.g., 'X Corp is a company', 'Jack Ma was mentioned'). Focus only on phrases indicating power/influence.

  3.  **Output Format:** Return the results as a JSON object with a single key "entities". The value should be a list of objects, where each object represents a unique identified entity and has the following structure:
      ```json
      {
        "entity_name": "string",
        "entity_type": "string (from predefined list or 'Other')",
        "mention_count": integer,
        "influence_context_phrases": [
          "string (phrase indicating influence)",
          "string (another phrase indicating influence)",
          ... // Include one phrase per influential mention found
        ]
      }
      ```
      - If an entity is mentioned multiple times but only some mentions have influential context, include only the phrases from the influential mentions in the `influence_context_phrases` list. If no mentions have influential context, this list should be empty `[]`. Ensure `entity_name` and `entity_type` accurately reflect the unique entity."

- **Quota Considerations:** Aim to combine Extraction and Context requests into a single prompt/call per article where possible to conserve the ~150-400 RPD allocated for this tier. The large context window of Flash models supports providing the full article content.
- **Output Usage:** The structured JSON output will be parsed by the Processing Worker to update `reader.entities` and `reader.article_entities`. The `influence_context_phrases` list provides qualitative input for the algorithmic calculation of the `influence_score`.

---

### 3. Guidance: Tier 2 & 3 Entity Extraction (Local Tools)

- **Goal:** Perform fast, efficient entity extraction for the bulk of articles using local tools.
- **Target Tools:**
  - **spaCy (Primary for Tier 2/3):** Use `en_core_web_lg` or `en_core_web_trf` for robust NER on Tier 2 articles. Use `en_core_web_sm` or just tagger/parser for basic entity tagging (locations, dates) on Tier 3 if full NER isn't needed. Leverage multi-processing/batching for speed.
  - **Hugging Face (Transformers - Tier 2 Supplement/Fallback):** Use local NER models (e.g., `dslim/bert-base-NER`, `Jean-Baptiste/roberta-large-ner-english`, potentially fine-tuned versions) to either re-process Tier 2 articles where spaCy's output seems insufficient or to specifically target entity types spaCy struggles with (e.g., complex concepts).
- **Influence Context (Tier 2):** Use spaCy's dependency parsing to identify verbs associated with entities and extract surrounding clauses as a proxy for influence context, feeding this simpler context into the scoring algorithm.
- **Implementation:** Integrate spaCy/Hugging Face model loading and inference within the Processing Worker logic.
- **Rate Limit Impact:** None (local processing).

---

### 4. Prompt: Optional Cluster Interpretation (Limited Use)

- **Goal:** Generate nuanced themes and identify framing for the _most important_ "hot" clusters, using Gemini's reasoning sparingly.
- **Target Tool:** `Gemini 1.5 Flash` / `2.0 Flash` (Gemini API - Free Tier)
- **Target Clusters:** Top ~5-10 "hot" clusters per day.
- **Input:**
  - `cluster_id`: ID of the hot cluster.
  - `top_article_snippets`: List of representative text snippets (e.g., titles + first sentences) from the top 5-10 articles closest to the cluster centroid.
  - `top_entities_in_cluster`: List of the most frequent/influential entities found within the articles of this cluster.
  - `preliminary_hf_frames` (Optional): List of framing tags suggested by local Hugging Face zero-shot models.
- **Instructions:**
  "You are an AI analyst for the Fracture platform. Analyze the provided information about a 'hot' news cluster to identify its core theme, potential narrative frames, and any inherent ambiguities.

  **Context:**

  - `cluster_id`: [Insert Cluster ID]
  - `top_article_snippets`: [Insert List of Snippets]
  - `top_entities_in_cluster`: [Insert List of Entities]
  - `preliminary_hf_frames` (Optional): [Insert List of HF Frames]

  **Tasks:**

  1.  **Synthesize Core Theme:** Based on the snippets and entities, describe the central topic or event of this cluster in a concise phrase (max 10 words). Propose 1-2 options if multiple interpretations exist.
  2.  **Identify Dominant Framing:** What is the primary lens or narrative perspective being used in these snippets? Consider the preliminary HF frames if provided, but use your reasoning. Examples: 'Economic Competition', 'National Security Concern', 'Technological Race', 'Political Strategy', 'Social Impact'. Identify 1-2 dominant frames.
  3.  **Surface Ambiguity/Tension:** Does the information suggest underlying conflicts, contradictions, unanswered questions, or multiple competing viewpoints? Briefly describe any observed ambiguity or tension within the cluster's theme.
  4.  **Suggest Refined Tags:** Based on your analysis, suggest 3-5 concise tags suitable for categorizing essays related to this cluster (e.g., 'US-China Tech', 'Trade War', 'Semiconductors', 'Geopolitics', 'Frame:National Security').

  **Output Format:** Return a JSON object:

  ````json
  {
    "suggested_themes": ["string", ...],
    "dominant_framing": ["string", ...],
    "observed_ambiguity": "string (brief description or null)",
    "suggested_tags": ["string", ...]
  }
  ```"
  ````

- **Quota Considerations:** Use extremely sparingly, max **~10-20 RPD**. Primary framing/tagging should rely on local tools.
- **Output Usage:** The output can inform manual review, refine automatically generated tags, or provide richer context when generating essays related to this cluster.

---

### 5. Prompts: Tiered Content Generation (News Feed & Rabbit Hole)

**Guiding Principle:** All generated content must actively embody Fracture's core philosophy: Trace Power, Recognize Framing, Interpret Ambiguity, See Sparks, Hold Competing Truths.

**Prompt 5A: Generate News Feed Paragraph (Tier 1 Focus)**

- **Goal:** Create a concise, engaging summary with a "spark" for the daily News Feed.
- **Target Tool:** `Gemini 1.5 Flash` / `2.0 Flash` (Gemini API - Free Tier)
- **Target Articles:** Core 20-30 articles daily (Tier 1, high-priority Tier 2).
- **Input:**
  - `article_id`, `article_title`, `cleaned_article_content`.
  - `key_entities`: List of important entities extracted for this article (name, type).
  - `cluster_info`: Cluster ID and potentially its theme/framing.
  - `relevant_data_point` (Optional): A specific statistic or fact extracted earlier or found via search.
- **Instructions:**
  "Generate a 'News Feed' paragraph (2-3 sentences, max 70 words) for the Fracture platform based on the provided article context. The paragraph must be factual but end with a 'spark' – a question, noted omission, contradiction, or hint of underlying tension – designed to make the user curious and click through to the deeper analysis ('Rabbit Hole').

  **Input Context:**

  - Article Title: [Insert Title]
  - Article Content Snippet: [Insert Key Sentence(s) or Brief Summary]
  - Key Entities: [Insert List of Key Entities]
  - Cluster Theme (Optional): [Insert Cluster Theme]
  - Relevant Data Point (Optional): [Insert Data Point]

  **Requirements:**

  1.  **Summarize Core Event:** Briefly state the main point of the article.
  2.  **Integrate Context:** Weave in the `relevant_data_point` (use brackets: [Data Point]) OR mention a `key_entity` hinting at its role/power.
  3.  **Inject Spark:** Conclude with a compelling question, highlight a discrepancy, note silence from a key player, or point towards an ambiguity that invites further exploration. Examples: 'What does this mean for X?', 'Why the conflicting reports?', 'The official statement notably omitted Y.', 'Sources remain divided on the motive.'
  4.  **Tone:** Objective, concise, but provocative via the spark.

  **Output:** Provide ONLY the generated paragraph text."

- **Quota Considerations:** ~1 call/paragraph = **~20-30 RPD**.
- **Tagging:** Topic/Framing tags are generated separately using local Hugging Face zero-shot models or spaCy keyword analysis based on the article/cluster, then associated with the generated essay entry.

**Prompt 5B: Generate Rabbit Hole Layers (Tier 1 - Full 5 Layers)**

- **Goal:** Create the deep, layered analysis for high-priority articles, fully embodying Fracture's interpretive philosophy.
- **Target Tool:** `Gemini 1.5 Flash` / `2.0 Flash` (Gemini API - Free Tier)
- **Target Articles:** ~100-150 articles/day.
- **Input (Potentially requires multiple calls or one large context call):**
  - `article_id`, `article_title`, `cleaned_article_content`.
  - `extracted_entities`: Full list of entities and influence context (from Prompt 1 output).
  - `cluster_info`: Cluster ID, theme, framing tags.
  - `similar_article_ids` (Optional): List of related article IDs found via embedding similarity (local Sentence Transformers).
  - `target_layer`: Integer (1-5) specifying which layer to generate (if generating layer-by-layer).
- **Instructions (Provide all context, specify `target_layer`):**
  "Generate Layer [target_layer] of the 'Rabbit Hole' analysis for the Fracture platform, based on the provided article context and adhering to the specific goal of this layer. Ensure the output reflects Fracture's philosophy (Trace Power, Recognize Framing, Interpret Ambiguity, Hold Competing Truths).

  **Input Context:**

  - Article Title: [Insert Title]
  - Article Content Summary/Key Points: [Insert Summary]
  - Extracted Entities & Influence Context: [Insert JSON from Prompt 1 output]
  - Cluster Info: [Insert Cluster ID, Theme, Frames]
  - Similar Articles (Optional): [Insert List of related IDs/Titles]

  **Layer-Specific Instructions:**

  - **If `target_layer` == 1 (Recap & Context):** (~100 words) Restate the core event from the article. Provide immediate historical or topical context using the `pub_date`, `domain`, `cluster_info`, or links to `similar_article_ids`. Ground the user in the basic facts and immediate relevance.
  - **If `target_layer` == 2 (Theories & Interpretations):** (~150 words) Offer 2-3 distinct, plausible interpretations of the event. Explicitly apply relevant theoretical lenses (e.g., Realism, Liberalism, Constructivism, Critical Theory, Elite Theory, Game Theory - choose appropriately) and _state the theory name_. Structure this as presenting competing analytical viewpoints based on established frameworks. Aim for structured disagreement.
  - **If `target_layer` == 3 (Correlations & Patterns):** (~150 words) Connect this event to broader patterns. Reference correlations found via `cluster_info` or `similar_article_ids` (embedding similarity). Trace patterns related to the key `extracted_entities` (e.g., recurring actions, co-occurrences across articles, `influence_score` trends if available). Cite external data points or trends where possible (e.g., '[US ITC data shows...]'). Focus on data-driven connections.
  - **If `target_layer` == 4 (Intriguing Angles & Ambiguities):** (~150 words) Pose provocative "What if?" or "Is it really about C?" questions based on the analysis so far. Explicitly highlight contradictions, significant omissions, unanswered questions, or narrative gaps found in the source article or related information ('What's _not_ being discussed?'). Surface potential hidden agendas or less obvious power dynamics. Encourage critical questioning.
  - **If `target_layer` == 5 (Opinions & Counter-Narratives):** (~150 words) Present a challenging perspective that counters the dominant narrative(s). This could involve citing specific (real or illustrative) academic critiques, expert opinions diverging from the consensus, insights from alternative theoretical frameworks (e.g., post-structuralist view), or perspectives from marginalized groups potentially affected by the event. Clearly frame this as a specific viewpoint (e.g., "From a critical perspective...", "However, some analysts argue..."). Equip the user to hold competing truths.

  **Output Format:** Return ONLY the generated text for the specified `target_layer`."

- **Quota Considerations:** Most intensive step. 5 calls/article x 100-150 articles = **~500 - 750 RPD**. Explore combining layers into fewer calls if context window allows, but distinct layers might yield better results. Supplementing Layers 3/4/5 with local Hugging Face models for ~20% of these articles is crucial to stay within limits.
- **Local Supplementation:** For Layer 3 (Correlations), use local embeddings to find similar articles _before_ calling Gemini. For Layers 4/5 (Angles/Opinions), experiment with local fine-tuned T5/GPT models to generate this content for a subset of articles, feeding the output into the database instead of calling Gemini.

**Prompt 5C: Generate Rabbit Hole Layers (Tier 2 - Layers 1 & 2 ONLY)**

- **Goal:** Provide initial context and theoretical framing for medium-priority articles using minimal API quota.
- **Target Tool:** `Gemini 1.5 Flash` / `2.0 Flash` (Gemini API - Free Tier)
- **Target Articles:** ~100-200 articles/day.
- **Input:** Same as 5B, but context may be less rich (e.g., entities from spaCy). Specify `target_layer` = 1 or 2.
- **Instructions:**
  "Generate Layer [target_layer] (where target_layer is 1 or 2) of the 'Rabbit Hole' analysis for the Fracture platform.

  **Input Context:** [Provide Article Title, Summary, Entities (likely from spaCy), Cluster Info]

  **Layer-Specific Instructions:**

  - **If `target_layer` == 1 (Recap & Context):** (~100 words) Restate the core event. Provide immediate context using `pub_date`, `domain`, `cluster_info`. _Alternatively, for some articles, a locally generated spaCy extractive summary may be used instead of calling the API._
  - **If `target_layer` == 2 (Theories & Interpretations):** (~150 words) Offer 1-2 plausible interpretations applying relevant theoretical lenses (e.g., Realism, Liberalism) and state the theory name. Focus on providing basic analytical framing.

  **Output Format:** Return ONLY the generated text for the specified `target_layer`."

- **Quota Considerations:** ~2 calls/article x 100-200 articles = **~200 - 400 RPD**.
- **Local Supplementation:** Use spaCy extractive summaries for Layer 1 where appropriate to further reduce calls.

**Guidance 5D: Local NLP Tool Roles in Content Pipeline**

- **Goal:** Define tasks handled by local tools to support content generation and conserve API quota.
- **Tools & Tasks:**
  - **Hugging Face Summarizers (`bart-large-cnn`, `pegasus-xsum`, `t5-small`):**
    - Pre-process articles >2048 tokens for `text-embedding-004`.
    - Potentially generate basic summaries for Tier 2 News Feed items if Gemini quota is tight.
  - **Hugging Face Zero-Shot Classifiers (`bart-large-mnli`):**
    - Generate framing tags (e.g., "Economic", "Security") for News Feed items and potentially clusters.
  - **Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`):**
    - Find similar articles (input for Rabbit Hole Layer 3).
    - Fallback/alternative for primary embedding.
  - **Hugging Face Generative Models (Fine-tuned `T5`, etc.):**
    - Experimentally generate Rabbit Hole Layers 4 & 5 for a subset of Tier 1 articles to reduce Gemini usage. Requires fine-tuning effort.
  - **spaCy:**
    - Generate extractive summaries as a fallback for `text-embedding-004` pre-processing or potentially for Rabbit Hole Layer 1.
    - Extract keywords for topic tagging associated with essays.
    - Provide dependency parsing context for Tier 2 entity influence scoring.
- **Integration:** Outputs from these local tools (summaries, tags, similar article lists) should be stored appropriately or fed as input into the relevant Gemini prompts (e.g., Prompt 5B).

---

This reworked set of prompts and guidance provides a framework for implementing the optimized, mixed-tool NLP pipeline for Fracture, balancing the advanced capabilities of Gemini API free tiers with the efficiency and unlimited nature of local NLP tools to meet the project's demanding scale and analytical depth within its resource constraints. Continuous monitoring, testing, and refinement of both the prompts and the tiered logic will be essential.
