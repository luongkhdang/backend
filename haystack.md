Haystack by Deepset
use gemini api as node

in step 5 (@step5.py), we are going to use Haystack (a rag system) to gather more data to write a report/essay for each group.

**Rich RAG & Sophisticated Context Processing (Backend):** For each group, deep historical context is retrieved via RAG.

1. in the folder src/steps/step5/ , the file group.json contain 50 groups of article.
   group.json structure as follow:

{
"processed_date": "2025-04-21 07:00 PM",
"article_groups": {
"group_1": {
"group_rationale": "Examines the framing of US foreign policy decisions, particularly concerning aid allocation and sanctions, revealing underlying narratives and potential biases in reporting. Aligns strongly with Recognize Framing and Trace Power.",
"article_ids": [
129078,
127985,
128814,
128747,
129124,
129093,
128101
]
},
"group_2":{
...
}
}

1. now, in step 5, for each group, we want to use @generator.py in the folder src/gemini/modules/ to gerenate essays for that topic using the the articles metadata (`title`, `domain`, `pub_date`, `frame_phrases` (we only want to utilize `Theories_and_interpretations` and `Intriguing_angles`, ignore the rest), top 5 influencial entities for the specific article, entity's type, and all of its snippets), and historical articles (from RAG (Haystack))

example of `frame_phrases`:
{"Political cost-benefit", ..,"Theories_and_interpretations: Applying Public Choice Theory, the article highlights how individual congressional Republicans face conflicting incentives. While there's pressure to cut Medicaid spending (driven by conservative budget hawks), doing so could harm their own constituents who benefit from the ACA's Medicaid expansion. The text states, \"the changes that could save the most money would impose heavy costs on many of their own voters.\" This suggests that representatives are weighing the benefits of adhering to party ideology against the potential electoral consequences of harming their constituents, a classic example of rational actors pursuing their self-interest within a political system. This theory suggests that the final decision will likely be a compromise reflecting the relative power and influence of these competing interests.","Intriguing_angles: The article repeatedly emphasizes the partisan split in Medicaid expansion, noting that states and districts with larger expansion populations tend to lean Democratic. However, it also acknowledges that a significant number of Republicans represent districts with above-average Medicaid enrollment. This creates a tension: is the focus on the partisan divide intended to justify cuts targeting primarily Democratic areas, or is it a genuine attempt to highlight the difficult choices facing Republicans who must balance fiscal conservatism with the needs of their constituents? What are the potential long-term consequences of framing healthcare access as a partisan issue, and how might this affect future policy decisions?"}

Historical articles (max 20. need to rate and select the top 20 what would suplement and answer all Intriguing_angles and Theories_and_interpretations of the articles in the group. We will need to append all Intriguing_angles and Theories_and_interpretations in to a group Intriguing_angles and Theories_and_interpretations) selected by Haystack RAG with the following:

- Embedding // Haystack natively supports PostgreSQL with pgvector for storing and querying embeddings, making it ideal for your data structure (articles, entities, clusters, embeddings)
- Cluster
- Event
- Best fit to answer the group's Intriguing_angles and Theories_and_interpretations
- Structured Data Retrieval:
  - Entity Relationships //Identify relationships between entities by finding articles where multiple entities co-occur frequently, indicating potential alliances or conflicts. Rank by relevance (e.g., using confidence_score for relationships, event_date recency, or influence_score for entities). Deduplicate overlapping information (e.g., similar event mentions).
  - Retrieve events involving key entities to construct a historical timeline.
  - Fetch policies or agreements linked to key entities, including their descriptions and effective dates.
  - Retrieve explicit relationships between entities, prioritizing high-confidence connections.
  - Fetch co-occurrence contexts from entity_relationships where entities are mentioned together in specific contexts
  - Fetch events and policies linked to entities within the news group’s cluster.

The goal is to fill 200,000 input token context to produce meaningful essays that align to the application core.
