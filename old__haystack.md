Haystack Retrieval Strategy for Historical Context

The primary goal is to use Haystack's RAG capabilities to retrieve up to 20 relevant historical articles from the database. These articles should provide the necessary context to comprehensively address the combined "Intriguing_angles" and "Theories_and_interpretations" derived from all articles within the current group.

To achieve this, the Haystack retrieval process will identify and rank potential historical articles based on multiple signals:

Semantic Relevance: Utilizing vector embeddings (via pgvector) to find articles semantically similar to the group's combined angles and theories.
Contextual Relevance (Informed by Structured Data): Leveraging connections within your structured data tables to identify historical articles discussing relevant events, policies, or entity interactions. This includes finding articles that:
Mention significant events involving key entities linked to the current group.
Discuss policies or agreements involving these key entities.
Detail high-confidence relationships or frequent co-occurrences between these key entities (using entity_relationships and its metadata).
Belong to relevant clusters (articles.cluster_id).
Best Fit Ranking: Prioritizing and selecting the top 20 historical articles that, based on the above signals, are most likely to contain information needed to answer the group's specific "Intriguing_angles" and "Theories_and_interpretations".
Final Prompt Assembly for Gemini

The input prompt for the Gemini API will be constructed to approach the 200,000 token target, containing:

Core Prompt: The specific instructions for the essay generation task.
Current Group Article Context: Metadata for each article in the current group:
title, domain, pub_date, content
the top 5 most influential entities and associated snippets (entity_snippets.snippet) for the top influential entities mentioned within that specific historical article.
Only the "Intriguing_angles" and "Theories_and_interpretations" extracted from the frame_phrases array.
Selected Historical Article Context: For each of the top 20 retrieved historical articles:
title, domain, pub_date, content
Only the "Intriguing_angles" and "Theories_and_interpretations" extracted from the frame_phrases array.
the top 5 most influential entities and associated snippets (entity_snippets.snippet) for the top influential entities mentioned within that specific historical article.

This comprehensive input aims to provide Gemini with sufficient context from both the current group and relevant history to generate insightful essays aligned with your application's objectives.
